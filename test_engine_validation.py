"""
Engine 文件验证测试脚本
=======================
功能：验证 .engine / .pt 模型能否在项目中正常加载推理。
加载模式完全匹配 inference_engine.py 的实际逻辑：
  - TensorRT：TRT profile 提取 batch → batch 填充 warmup → 清理缓存
  - PyTorch：单帧 warmup

测试项：
  1. 元数据完整性（names / stride / batch / imgsz / task）
  2. Profile batch 一致性
  3. YOLO 加载 & Warmup 推理
  4. 类别名验证（model.names vs 元数据 names）
  5. 真实推理结果验证（检测框 / 置信度 / 类别分布）
  6. 各 GPU 显存占用记录（加载前后 & 增量）

用法：
  python test_engine_validation.py
  python test_engine_validation.py --model-dir models
  python test_engine_validation.py --model hl_traffic.engine --gpu 0
  python test_engine_validation.py --model mudflat_geocage.engine --gpu 1
"""

import argparse
import json
import os
import pickle
import sys
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ── 常量 ──
DEFAULT_MODEL_DIR = "models"
DEFAULT_IMGSZ = 640
PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️ WARN"

# ANSI 颜色
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


# ══════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════

def _get_current_trt_version() -> str:
    """获取当前环境的 TensorRT 版本"""
    try:
        import tensorrt as trt
        return trt.__version__
    except Exception:
        return "未安装"


def _find_pt_fallback(engine_path: str) -> str:
    """查找 .engine 对应的 .pt 回退文件"""
    pt_path = engine_path.replace(".engine", ".pt")
    if os.path.exists(pt_path):
        return pt_path
    return ""


def _fmt_gib(bytes_val: int) -> str:
    """字节 → GiB 字符串"""
    return f"{bytes_val / (1 << 30):.2f} GiB"


def _colored(status: str) -> str:
    """状态标签着色"""
    if PASS in status:
        return f"{GREEN}{status}{RESET}"
    elif FAIL in status:
        return f"{RED}{status}{RESET}"
    return f"{YELLOW}{status}{RESET}"


def _read_engine_metadata(engine_path: str) -> dict:
    """读取 Ultralytics engine 元数据（兼容 pickle 和 JSON 格式）"""
    with open(engine_path, "rb") as f:
        data = f.read()
    # pickle 格式（旧版）
    magic = b"UlTralYtiCsEnGiNe"
    idx = data.rfind(magic)
    if idx >= 0:
        return pickle.loads(data[idx + len(magic):])
    # JSON 格式（ultralytics 8.3+，export_model_pt_engine.py 使用的格式）
    meta_len = int.from_bytes(data[:4], byteorder="little")
    return json.loads(data[4:4 + meta_len].decode("utf-8"))


def diagnose_engine_file(engine_path: str) -> dict:
    """
    深度诊断 engine 文件结构（不反序列化 TRT 引擎，安全）。
    返回文件结构分析结果：
      - 文件总大小 / 元数据偏移 / TRT 数据大小
      - 元数字段列表 / TRT 版本
      - 若元数据损坏则返回前 64 字节 hex
    """
    info = {"file_size": 0, "has_meta": False, "meta_size": 0,
            "trt_data_offset": 0, "trt_data_size": 0,
            "meta_raw_hex": "", "error": ""}
    try:
        with open(engine_path, "rb") as f:
            data = f.read()
        info["file_size"] = len(data)

        # 尝试 JSON 格式
        meta_len = int.from_bytes(data[:4], byteorder="little")
        if 0 < meta_len < len(data) - 4:
            meta_raw = data[4:4 + meta_len]
            try:
                meta = json.loads(meta_raw.decode("utf-8"))
                info["has_meta"] = True
                info["meta_size"] = meta_len
                info["trt_data_offset"] = 4 + meta_len
                info["trt_data_size"] = len(data) - 4 - meta_len
                info["meta_fields"] = list(meta.keys())
                info["meta_trt_version"] = meta.get("trt_version", "无此字段")
            except Exception as e:
                info["error"] = f"JSON 解析失败: {e}"
                info["meta_raw_hex"] = meta_raw[:64].hex()
        else:
            info["error"] = f"无法识别的元数据格式 (meta_len={meta_len})"
    except Exception as e:
        info["error"] = str(e)
    return info


def _get_gpu_memory(gpu_id: int) -> dict:
    """获取指定 GPU 显存信息"""
    free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
    used_bytes = total_bytes - free_bytes
    return {
        "gpu_id": gpu_id,
        "free_bytes": free_bytes,
        "used_bytes": used_bytes,
        "total_bytes": total_bytes,
        "free_gib": free_bytes / (1 << 30),
        "used_gib": used_bytes / (1 << 30),
        "total_gib": total_bytes / (1 << 30),
        "used_pct": used_bytes / total_bytes * 100,
    }


def _log_memory(mem: dict, tag: str = ""):
    """格式化输出显存信息"""
    tag_str = f" [{tag}]" if tag else ""
    print(
        f"    GPU {mem['gpu_id']}{tag_str}: "
        f"{mem['used_gib']:.2f} / {mem['total_gib']:.2f} GiB "
        f"({mem['used_pct']:.1f}%)"
    )


# ══════════════════════════════════════════════════════════
#  验证函数
# ══════════════════════════════════════════════════════════

def validate_metadata(engine_path: str) -> dict:
    """
    验证项 1：检查 engine 元数据完整性。
    返回 {"status": bool, "metadata": dict, "checks": dict}
    """
    checks = {}
    metadata = {}

    try:
        metadata = _read_engine_metadata(engine_path)
        checks["元数据可读"] = True
    except Exception as e:
        return {"status": False, "metadata": {}, "checks": {"元数据可读": False, "错误": str(e)}}

    # 检查必填字段 — 宽松校验，罕见值仅 WARN 不 FAIL
    def _check_int_or_list_int(v):
        """stride 可能是 int 或 [int]（JSON 反序列化后 list）"""
        if isinstance(v, int):
            return True, v > 0
        if isinstance(v, (list, tuple)) and len(v) == 1 and isinstance(v[0], (int, float)):
            return True, v[0] > 0
        return False, False

    field_checks = {
        "names": lambda v: (isinstance(v, dict), len(v) > 0),
        "stride": _check_int_or_list_int,
        "batch": lambda v: (isinstance(v, int), v > 0),
        "imgsz": lambda v: (isinstance(v, (list, tuple)), len(v) == 2),
        "task": lambda v: (isinstance(v, str), len(v) > 0),
    }

    for field, validator in field_checks.items():
        val = metadata.get(field)
        if val is None:
            checks[f"字段 {field}"] = False
            continue
        type_ok, value_ok = validator(val)
        checks[f"字段 {field}"] = type_ok and value_ok

    return {"status": True, "metadata": metadata, "checks": checks}


def validate_profile_consistency(metadata: dict, real_batch: int) -> dict:
    """
    验证项 2：对比 TRT profile batch（从已加载的 YOLO model 提取）与元数据 batch 是否一致。
    """
    meta_batch = metadata.get("batch", -1)
    return {
        "status": real_batch == meta_batch,
        "profile_batch": real_batch,
        "meta_batch": meta_batch,
    }


def validate_load_and_infer(engine_path: str, gpu_id: int, batch_size: int, imgsz: int) -> dict:
    """
    按 inference_engine.py 的 _load_on_gpu 实际模式加载 engine 并验证：
      - YOLO 加载
      - TRT profile batch 提取（覆盖配置值）
      - Warmup 推理（TRT→batch填充，PyTorch→单帧）
      - 推理结果验证（boxes / conf / class_id 字段完整性）
      - 类别名验证（model.names 与元数据对比）
      - Warmup 后清理缓存，记录稳态显存
    """
    result = {
        "status": False, "is_tensorrt": False,
        "mem_before": {}, "mem_after": {},
        "infer_time_ms": 0.0, "real_batch": batch_size,
        "model_names": {}, "detections": [], "error": "",
    }

    try:
        mem_before = _get_gpu_memory(gpu_id)
        result["mem_before"] = mem_before

        is_tensorrt = engine_path.endswith(".engine")
        result["is_tensorrt"] = is_tensorrt

        torch.cuda.set_device(gpu_id)
        t0 = time.time()

        # ── 1. YOLO 加载 engine（与 inference_engine._load_on_gpu 一致）──
        model = YOLO(engine_path)

        # ── 2. 从 TRT profile 提取真实 batch（与 inference_engine 一致）──
        real_batch = batch_size
        if is_tensorrt:
            try:
                trt_engine = model.engine.engine  # ICudaEngine
                input_name = trt_engine.get_tensor_name(0)
                opt_shape = trt_engine.get_tensor_profile_shape(input_name, 0)[1]
                engine_batch = opt_shape[0]
                if engine_batch != real_batch:
                    real_batch = engine_batch
            except Exception:
                pass  # 保持默认 batch
        result["real_batch"] = real_batch

        # ── 3. Warmup 推理（与 inference_engine._load_on_gpu 一致）──
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        if is_tensorrt:
            # TRT：batch 填充，模拟 _predict_slices 逻辑
            model.predict(
                [dummy] * real_batch,
                verbose=False, batch=real_batch,
            )
        else:
            # PyTorch：单帧
            model.predict(dummy, verbose=False)

        # ── 4. Warmup 后清理临时 buffer（与 inference_engine 一致）──
        if is_tensorrt:
            torch.cuda.empty_cache()
        time.sleep(0.3)

        # ── 5. 运行真实推理并验证结果 ──
        # 构造含模拟目标的图片，确保有检测输出
        test_frame = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        cv2.rectangle(test_frame, (50, 50), (200, 200), (200, 200, 200), -1)
        cv2.rectangle(test_frame, (400, 300), (550, 450), (180, 180, 180), -1)

        pred_results = model.predict(
            test_frame, conf=0.25, imgsz=imgsz, verbose=False,
        )

        infer_time = (time.time() - t0) * 1000
        result["infer_time_ms"] = infer_time

        # ── 6. 提取检测结果 ──
        detections = []
        for pred in pred_results:
            boxes = pred.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                detections.append({
                    "bbox": boxes.xyxy[i].cpu().numpy().tolist(),
                    "conf": float(boxes.conf[i].cpu().numpy()),
                    "cls_id": int(boxes.cls[i].cpu().numpy()),
                })
        result["detections"] = detections

        # ── 7. 获取加载后的模型类别名 ──
        try:
            result["model_names"] = dict(model.names) if hasattr(model, "names") else {}
            if not result["model_names"] and hasattr(model, "model") and hasattr(model.model, "names"):
                result["model_names"] = dict(model.model.names)
        except Exception:
            pass

        # ── 8. 清理 + 记录稳态显存 ──
        del model
        if is_tensorrt:
            torch.cuda.empty_cache()
        time.sleep(0.3)
        mem_after = _get_gpu_memory(gpu_id)
        result["mem_after"] = mem_after
        result["status"] = True

    except Exception as e:
        err_msg = str(e)

        # 检测 TRT 版本不匹配 — 补充诊断信息
        if is_tensorrt and ("NoneType" in err_msg or "create_execution_context" in err_msg):
            try:
                import tensorrt as trt_local
                trt_version = trt_local.__version__
            except Exception:
                trt_version = "?"
            err_msg += (
                f"\n  │     当前环境 TRT 版本: {trt_version}"
                f"\n  │     引擎可能使用了不同的 TRT 版本构建"
                f"\n  │     请确认 export_model_pt_engine.py 与此脚本运行在相同环境"
                f"\n  │     解决: python export_model_pt_engine.py --model {os.path.basename(engine_path).replace('.engine','')} --force"
            )
        elif is_tensorrt:
            try:
                import tensorrt as trt_local
                err_msg += f"\n  │     当前 TRT 版本: {trt_local.__version__}"
            except Exception:
                pass

        result["error"] = err_msg
        try:
            result["mem_after"] = _get_gpu_memory(gpu_id)
        except Exception:
            pass

    return result


# ══════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════

def validate_single_engine(model_path: str, gpu_id: int, imgsz: int):
    """对单个模型文件执行全部验证项（支持 .engine 和 .pt）"""
    filename = os.path.basename(model_path)
    is_engine = filename.endswith(".engine")
    size_mb = os.path.getsize(model_path) / (1 << 20)

    # ── 检测可用 GPU ──
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count == 0 and is_engine:
        print(f"\n{CYAN}{'='*60}{RESET}")
        print(f"{RED}错误: 未检测到 CUDA GPU，无法加载 TensorRT engine{RESET}")
        print(f"{CYAN}{'='*60}{RESET}")
        return False

    effective_gpu = gpu_id if gpu_id < gpu_count else 0
    gpu_name = torch.cuda.get_device_name(effective_gpu) if gpu_count > 0 else "N/A"
    total_mem = _get_gpu_memory(effective_gpu)["total_gib"] if gpu_count > 0 else 0

    model_type = "TensorRT" if is_engine else "PyTorch"
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"   模型: {filename}")
    print(f"   类型: {model_type}")
    print(f"   大小: {size_mb:.1f} MiB")
    if gpu_count > 0:
        print(f"   GPU:  {effective_gpu} ({gpu_name}, {total_mem:.1f} GiB)")
    print(f"{CYAN}{'='*60}{RESET}")

    all_pass = True

    # ── 1. 元数据验证（仅 .engine 文件）──
    metadata = {}
    meta_names = {}
    effective_batch = 8
    if is_engine:
        print(f"\n  ├─ [1/3] 元数据完整性验证 ...")
        meta_result = validate_metadata(model_path)
        metadata = meta_result["metadata"]
        meta_readable = meta_result["checks"].get("元数据可读", True)

        if not meta_readable:
            err = meta_result["checks"].get("错误", "未知错误")
            print(f"  │   {FAIL} 元数据读取失败: {err}")
        else:
            for check_name, ok in meta_result["checks"].items():
                if check_name == "元数据可读":
                    continue
                icon = PASS if ok else WARN
                print(f"  │   {icon} {check_name}")

            meta_batch = metadata.get("batch", "?")
            meta_imgsz = metadata.get("imgsz", "?")
            meta_task = metadata.get("task", "?")
            meta_stride = metadata.get("stride", "?")
            meta_names = metadata.get("names", {})
            meta_pt = metadata.get("pt", "?")
            meta_trt_ver = metadata.get("trt_version", "")
            print(f"  │   {PASS if meta_batch != '?' else WARN}  batch={meta_batch}")
            print(f"  │   {PASS if meta_imgsz != '?' else WARN}  imgsz={meta_imgsz}")
            print(f"  │   {PASS if meta_task != '?' else WARN}  task={meta_task}")
            print(f"  │   {PASS if meta_stride != '?' else WARN}  stride={meta_stride}")
            print(f"  │   {PASS if meta_names else WARN}  names={len(meta_names)} classes")
            print(f"  │   {'   '} pt={meta_pt}")
            if meta_trt_ver:
                cur_trt = _get_current_trt_version()
                match = PASS if meta_trt_ver == cur_trt else FAIL
                print(f"  │   {match} 构建 TRT={meta_trt_ver}  |  当前 TRT={cur_trt}")

        effective_batch = metadata.get("batch", 8) if isinstance(metadata.get("batch"), int) else 8

    # ── 2. 加载 + 推理验证（核心）──
    step_label = "2/2" if not is_engine else "2/3"
    print(f"\n  ├─ [{step_label}] 加载 / 类别 / 推理验证 ...")
    infer_result = validate_load_and_infer(model_path, effective_gpu, effective_batch, imgsz)

    if infer_result["status"]:
        real_batch = infer_result["real_batch"]
        is_trt = infer_result["is_tensorrt"]
        engine_type = "TensorRT" if is_trt else "PyTorch"
        batch_note = f" (TRT profile={real_batch})" if is_trt and real_batch != effective_batch else ""
        print(f"  │   {PASS} YOLO 加载 {engine_type} 模型成功")
        print(f"  │   {PASS} Warmup 推理完成 ({infer_result['infer_time_ms']:.1f}ms, "
              f"batch={real_batch}{batch_note})")

        # ── Profile batch 一致性验证（仅 TRT，从已加载的 model 提取）──
        if is_engine:
            meta_batch_val = metadata.get("batch", None)
            if meta_batch_val is not None and isinstance(meta_batch_val, int):
                profile_result = validate_profile_consistency(metadata, real_batch)
                if profile_result["status"]:
                    print(f"  │   {PASS} profile batch={profile_result['profile_batch']} "
                          f"vs 元数据 batch={profile_result['meta_batch']} -- 一致")
                else:
                    print(f"  │   {WARN} profile batch={profile_result['profile_batch']} "
                          f"vs 元数据 batch={profile_result['meta_batch']} -- 不一致(非致命)")
            else:
                print(f"  │   {WARN} 元数据 batch 不可用，跳过 profile 检查")

        # ── 类别名验证 ──
        model_names = infer_result["model_names"]
        if meta_names and model_names:
            meta_keys = set(str(k) for k in meta_names.keys())
            model_keys = set(str(k) for k in model_names.keys())
            if meta_keys == model_keys:
                print(f"  │   {PASS} 类别名匹配: {len(model_names)} classes "
                      f"({', '.join(str(model_names[k]) for k in sorted(model_names.keys(), key=int))})")
            else:
                missing = meta_keys - model_keys
                extra = model_keys - meta_keys
                msg = []
                if missing:
                    msg.append(f"元数据有但模型无: {sorted(missing)}")
                if extra:
                    msg.append(f"模型有但元数据无: {sorted(extra)}")
                print(f"  │   {WARN} 类别名不匹配: {'; '.join(msg)}")
        elif model_names and not meta_names:
            print(f"  │   {WARN} 模型有类别名但元数据为空: {len(model_names)} classes")
        elif meta_names and not model_names:
            print(f"  │   {WARN} 元数据有类别名但模型加载后为空: {len(meta_names)} classes")
        else:
            print(f"  │   {WARN} 无类别名信息")

        # ── 推理结果验证 ──
        detections = infer_result["detections"]
        if detections:
            # 按 cls_id 分组统计
            cls_counts = {}
            for d in detections:
                cid = d["cls_id"]
                cls_counts[cid] = cls_counts.get(cid, 0) + 1
            cls_summary = []
            for cid in sorted(cls_counts.keys()):
                name = model_names.get(cid, str(cid)) if model_names else str(cid)
                cls_summary.append(f"{name}={cls_counts[cid]}")
            confs = [d["conf"] for d in detections]
            print(f"  │   {PASS} 推理输出正常: {len(detections)} 个检测框 "
                  f"(conf={min(confs):.2f}~{max(confs):.2f})")
            print(f"  │   {'   '} 类别分布: {', '.join(cls_summary)}")
        else:
            print(f"  │   {WARN} 推理完成但无检测框（可能 conf 阈值过高或模型输出异常）")

        # ── 显存记录 ──
        mb = infer_result["mem_before"]
        ma = infer_result["mem_after"]
        _log_memory(mb, "加载前")
        _log_memory(ma, "加载后")

        used_delta = ma["used_gib"] - mb["used_gib"]
        delta_str = f"+{used_delta:.2f}" if used_delta >= 0 else f"{used_delta:.2f}"
        print(f"  │   显存增量: {delta_str} GiB")
        all_pass = True  # 加载推理成功即通过
    else:
        # 分多行打印错误，避免一行太长
        err_lines = infer_result['error'].split('\n')
        for i, line in enumerate(err_lines):
            prefix = "  │   " if i == 0 else "  │     "
            print(f"  │   {FAIL} {line}" if i == 0 else f"  │     {line}")

        # ── 失败时自动诊断文件结构 ──
        if filename.endswith(".engine"):
            diag = diagnose_engine_file(model_path)
            if not diag["has_meta"]:
                print(f"  │     📄 文件诊断: 元数据读取失败 - {diag['error']}")
                print(f"  │     文件大小: {diag['file_size']} bytes")
            else:
                print(f"  │     📄 文件诊断:")
                print(f"  │       文件总大小: {diag['file_size']} bytes ({diag['file_size']/(1<<20):.1f} MiB)")
                print(f"  │       元数据偏移: +{diag['trt_data_offset']} bytes ({diag['meta_size']} bytes JSON)")
                print(f"  │       TRT 数据区: {diag['trt_data_size']} bytes")
                print(f"  │       元数字段: {diag.get('meta_fields', 'N/A')}")
                print(f"  │       构建 TRT 版本: {diag.get('meta_trt_version', 'N/A')}")
                if diag.get("meta_trt_version", "").replace(".", "").isdigit():
                    cur = _get_current_trt_version()
                    if diag["meta_trt_version"] != cur:
                        print(f"  │       ⚠️ 构建 TRT={diag['meta_trt_version']} ≠ 当前 TRT={cur}")
                    else:
                        print(f"  │       ✅ TRT 版本匹配 ({cur}) — 问题不在版本")

        # 检测是否存在 .pt 回退方案
        if filename.endswith(".engine"):
            pt_path = model_path.replace(".engine", ".pt")
            if os.path.exists(pt_path):
                pt_size = os.path.getsize(pt_path) / (1 << 20)
                print(f"  │     💡 存在 .pt 回退模型: {os.path.basename(pt_path)} ({pt_size:.1f}MiB)")
                print(f"  │     在 inference_engine.py 中会自动回退 PyTorch 推理")
        all_pass = False

    # ── 结论 ──
    model_label = "模型" if not filename.endswith(".engine") else "engine"
    print(f"\n  └─ {'='*40}")
    if all_pass:
        print(f"  {GREEN}✅ {filename} 验证通过（{model_label}可正常加载推理）{RESET}")
    else:
        print(f"  {RED}❌ {filename} 验证失败（{model_label}无法加载推理）{RESET}")
    print(f"{CYAN}{'='*60}{RESET}\n")

    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description="验证 TensorRT .engine / .pt 模型能否在项目中正常加载推理"
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help=f"模型目录（默认: {DEFAULT_MODEL_DIR}）")
    parser.add_argument("--model", default="",
                        help="指定单个 engine/pt 文件（如 hl_traffic.engine），不指定则验证全部")
    parser.add_argument("--gpu", type=int, default=0, help="GPU 编号（默认: 0）")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help=f"输入尺寸（默认: {DEFAULT_IMGSZ}）")
    parser.add_argument("--prefix", default="",
                        help="engine 文件名前缀筛选（如 hl_）")
    args = parser.parse_args()

    # ── 环境信息 ──
    trt_ver = _get_current_trt_version()
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    print(f"{CYAN}{'='*60}{RESET}")
    print(f"  模型验证工具")
    print(f"  CUDA: {'可用' if torch.cuda.is_available() else '不可用'}, GPU 数: {gpu_count}")
    print(f"  TensorRT 版本: {trt_ver}")
    for i in range(gpu_count):
        mem = _get_gpu_memory(i)
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)} ({mem['total_gib']:.1f} GiB)")
    print(f"{CYAN}{'='*60}{RESET}\n")

    if not torch.cuda.is_available():
        print(f"{YELLOW}警告: CUDA 不可用，TensorRT engine 无法加载{RESET}")
        print(f"{YELLOW}      仅支持测试 .pt 模型（PyTorch 推理）{RESET}")

    # ── 收集模型文件 ──
    if not os.path.isdir(args.model_dir):
        print(f"{RED}错误: 模型目录不存在: {args.model_dir}{RESET}")
        sys.exit(1)

    if args.model:
        # 支持指定 .engine 或 .pt
        model_files = [args.model]
    else:
        model_files = sorted(os.listdir(args.model_dir))
        # 收集 .engine 和 .pt（优先 engine）
        engine_files = [f for f in model_files if f.endswith(".engine")]
        pt_files = [f for f in model_files if f.endswith(".pt") and f.replace(".pt", ".engine") not in engine_files]
        model_files = engine_files + pt_files
        if args.prefix:
            model_files = [f for f in model_files if f.startswith(args.prefix)]

    if not model_files:
        print(f"{YELLOW}未找到 .engine / .pt 文件（目录: {args.model_dir}）{RESET}")
        sys.exit(0)

    print(f"发现 {len(model_files)} 个模型文件:\n")
    for i, f in enumerate(model_files, 1):
        ftype = "TensorRT" if f.endswith(".engine") else "PyTorch"
        print(f"  [{i}] {f} ({ftype})")

    # ── 逐个验证 ──
    results = {}
    for fname in model_files:
        model_path = os.path.join(args.model_dir, fname)
        ok = validate_single_engine(model_path, args.gpu, args.imgsz)
        results[fname] = ok

    # ── 汇总 ──
    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed

    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"  📊 验证汇总")
    print(f"  {'='*40}")
    print(f"  总计: {len(results)}  |  {GREEN}通过: {passed}{RESET}  |  {RED}失败: {failed}{RESET}")
    print()
    for fname, ok in results.items():
        icon = f"{GREEN}✅{RESET}" if ok else f"{RED}❌{RESET}"
        ftype = " (TRT)" if fname.endswith(".engine") else " (PT)"
        print(f"  {icon}  {fname}{ftype}")

    if failed > 0:
        print(f"\n  💡 提示: .engine 文件加载失败通常是因为 TRT 版本不匹配")
        print(f"     当前 TRT: {_get_current_trt_version()}")
        print(f"     重新导出: python export_model_pt_engine.py --force")
    print(f"{CYAN}{'='*60}{RESET}")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
