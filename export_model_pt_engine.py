"""
将 models/ 下的 .pt 模型导出为 TensorRT .engine
"""

import argparse
import os
import time
import json
import onnx
import tensorrt as trt
from ultralytics import YOLO

DEFAULT_MODEL_DIR = "models"
DEFAULT_MAX_BATCH = 8
DEFAULT_IMGSZ = 640


# ── TensorRT 版本兼容 ──

def _create_trt_network(builder):
    """
    兼容 TensorRT 10/11 创建 Network.
    TensorRT 11 移除了 EXPLICIT_BATCH 标志（已为默认）。
    """
    try:
        # TensorRT 10.x -
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        return builder.create_network(flag)
    except AttributeError:
        # TensorRT 11+ — 显式 batch 已是默认行为
        return builder.create_network()


# ── 元数据嵌入 ──

def _extract_ultralytics_metadata(pt_path: str, imgsz: int, max_batch: int):
    """从 .pt 模型提取 Ultralytics 所需的元数据（names, stride 等）"""
    model = YOLO(pt_path)

    # 获取 class names
    names = {}
    try:
        names = model.names if hasattr(model, "names") else {}
        if not names:
            # 尝试从 model.model 获取
            if hasattr(model, "model") and hasattr(model.model, "names"):
                names = model.model.names
    except Exception:
        names = {}

    # 获取 stride（确保是 int，YOLO.predict 内部会做 int(stride)）
    stride = 32
    try:
        if hasattr(model, "model") and hasattr(model.model, "stride"):
            s = model.model.stride
            if hasattr(s, "tolist"):
                stride = s.tolist()
            elif hasattr(s, "cpu"):
                stride = s.cpu().numpy().tolist()
            else:
                stride = int(s)
            # stride 可能是 [32] 或 32，统一为 int
            stride = int(stride[0]) if isinstance(stride, (list, tuple)) else int(stride)
    except Exception:
        stride = 32

    # 获取任务类型（detect / segment / pose 等）
    task_type = getattr(model, "task", "detect")

    # 记录构建时的 TRT 版本（用于后续加载时诊断版本不匹配）
    try:
        import tensorrt as _trt_ver
        trt_version = _trt_ver.__version__
    except Exception:
        trt_version = "unknown"

    metadata = {
        "names": names,
        "stride": stride,
        "batch": max_batch,
        "imgsz": (imgsz, imgsz),
        "pt": True,
        "task": task_type,
        "trt_version": trt_version,
    }
    return metadata


def _embed_metadata_to_engine(engine_path: str, metadata: dict):
    """用 Ultralytics 原生格式嵌入元数据（4字节长度前缀 + JSON + TRT引擎数据）"""
    meta_json = json.dumps(metadata).encode("utf-8")
    meta_len = len(meta_json)

    # 读取已保存的 TRT 引擎数据
    with open(engine_path, "rb") as f:
        trt_data = f.read()

    # 以 Ultralytics 原生格式重写：[4字节长度][JSON元数据][TRT引擎]
    with open(engine_path, "wb") as f:
        f.write(meta_len.to_bytes(4, byteorder="little"))
        f.write(meta_json)
        f.write(trt_data)
def _fix_onnx_spatial_dims(onnx_path: str, imgsz: int):
    """
    将 ONNX 中 input 'images' 的 H, W 维度从动态改为固定值 imgsz。
    batch 维度保持动态不变。
    """
    model = onnx.load(onnx_path)
    inp = model.graph.input[0]
    shape = inp.type.tensor_type.shape
    for idx in (2, 3):  # H, W
        dim = shape.dim[idx]
        dim.dim_param = ""
        dim.dim_value = imgsz
    onnx.save(model, onnx_path)


def _set_memory_pool(config, workspace_gb: int):
    """
    设置 builder 内存池。
    TRT 10.x cu12 绑定中 WORKSPACE 仍然有效。
    """
    mem_bytes = workspace_gb * (1 << 30)
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, mem_bytes)
        print(f"TRT: 设置内存池 = {workspace_gb} GiB")
    except (AttributeError, Exception) as e:
        print(f"TRT: 无法设置内存池 ({e})，使用默认值")


def _build_trt_engine(
    onnx_path: str,
    engine_path: str,
    max_batch: int,
    imgsz: int,
    enable_fp16: bool,
    workspace_gb: int = 8,
) -> bool:
    """
    从 ONNX 构建 TensorRT engine（直接 API，兼容 TRT 10/11）。
    使用显式优化轮廓固定空间维度、仅 batch 动态。
    """
    logger = trt.Logger(trt.Logger.INFO)
    print(f"TRT 版本: {trt.__version__}")
    builder = trt.Builder(logger)
    network = _create_trt_network(builder)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for err in range(parser.num_errors):
                print(f"ONNX 解析错误: {parser.get_error(err)}")
            return False

    # 获取网络输入
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    print(f"TRT 网络输入: {input_name} | shape before profile: {input_tensor.shape}")

    # 创建优化轮廓：batch 动态，空间固定
    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_name,
        min=(1, 3, imgsz, imgsz),
        opt=(max_batch, 3, imgsz, imgsz),
        max=(max_batch, 3, imgsz, imgsz),
    )

    config = builder.create_builder_config()
    config.add_optimization_profile(profile)
    _set_memory_pool(config, workspace_gb)

    if enable_fp16:
        try:
            # TRT 10+
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("TRT: 启用 FP16")
            else:
                print("TRT: 硬件不支持快速 FP16，跳过")
                enable_fp16 = False
        except Exception:
            # TRT 11+ 可能有所不同
            config.set_flag(trt.BuilderFlag.FP16)
            print("TRT: 尝试启用 FP16")

    print(f"TRT: 开始构建 engine (profile: batch=1~{max_batch}, imgsz={imgsz})...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("TRT: build_serialized_network 返回 None，构建失败")
        return False

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"TRT: engine 序列化保存完成")
    return True


def export_engine(pt_path: str, max_batch: int, imgsz: int, device: int,
                  enable_fp16: bool = True) -> bool:
    """.pt → .engine（三步法：ONNX → 固定空间 → TRT API 构建 + 嵌入元数据）"""
    engine_path = pt_path.replace(".pt", ".engine")
    onnx_path = pt_path.replace(".pt", ".onnx")
    filename = os.path.basename(pt_path)
    t0 = time.time()

    # ---- 第 1 步：导出 ONNX（batch 动态，空间固定） ----
    print(f"[1/4] 导出 ONNX: {filename.replace('.pt', '.onnx')} ...")
    try:
        model = YOLO(pt_path)
        # ONNX 用 dynamic=True 导出全部维度为动态，第 2 步再固定 H,W
        model.export(
            format="onnx",
            half=enable_fp16,
            batch=max_batch,
            dynamic=True,
            imgsz=imgsz,
            device=device,
        )
        if not os.path.exists(onnx_path):
            onnx_path = os.path.join(
                os.path.dirname(pt_path),
                os.path.basename(pt_path).replace(".pt", ".onnx"),
            )
        if not os.path.exists(onnx_path):
            print(f"ONNX 文件未找到: {onnx_path}")
            return False
        onnx_size = os.path.getsize(onnx_path) / (1 << 20)
        print(f"ONNX 导出成功: {os.path.basename(onnx_path)} ({onnx_size:.1f}MB)")
    except Exception as e:
        print(f"ONNX 导出失败: {e}")
        return False

    # ---- 第 2 步：固定 ONNX 空间维度（H, W → imgsz） ----
    print(f"[2/4] 固定 ONNX 空间维度为 {imgsz}x{imgsz}（保留 batch 动态）...")
    try:
        _fix_onnx_spatial_dims(onnx_path, imgsz)
        check = onnx.load(onnx_path)
        inp_shape = [d.dim_value if d.dim_value else -1 for d in check.graph.input[0].type.tensor_type.shape.dim]
        print(f"    修改后 input shape: {inp_shape}")
        del check
    except Exception as e:
        print(f"ONNX 修改失败: {e}")
        return False

    # ---- 第 3 步：TensorRT API 构建 engine（兼容 TRT 10/11） ----
    print(f"[3/4] TensorRT API 构建 engine ...")
    print(f"     profile: batch=1~{max_batch}, imgsz={imgsz}, fp16={enable_fp16}")
    try:
        ok = _build_trt_engine(
            onnx_path, engine_path,
            max_batch=max_batch,
            imgsz=imgsz,
            enable_fp16=enable_fp16,
            workspace_gb=8,
        )
        if not ok:
            print("TensorRT 构建失败")
            return False
    except Exception as e:
        elapsed = time.time() - t0
        print(f"TensorRT 构建异常: {filename} ({elapsed:.1f}s) | error={e}")
        return False

    # ---- 第 4 步：嵌入 Ultralytics 元数据 ----
    print(f"[4/4] 嵌入 Ultralytics 元数据（names, stride 等）...")
    try:
        metadata = _extract_ultralytics_metadata(pt_path, imgsz, max_batch)
        _embed_metadata_to_engine(engine_path, metadata)
        print(f"    元数据嵌入完成: names={len(metadata.get('names', {}))} classes")
    except Exception as e:
        print(f"元数据嵌入警告（engine 仍可用，但 class names 可能丢失）: {e}")

    elapsed = time.time() - t0
    size_mb = os.path.getsize(engine_path) / (1 << 20) if os.path.exists(engine_path) else 0
    print(f"Engine 导出成功: {os.path.basename(engine_path)} ({elapsed:.1f}s, {size_mb:.1f}MB)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="导出 YOLO TensorRT engine（先 ONNX + 手动 TensorRT 构建，动态 batch 固定 imgsz）"
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="模型目录")
    parser.add_argument("--prefix", default="hl_",
                        help="模型文件名前缀筛选（如 hl_），只处理此前缀的模型；设为空字符串''则处理全部")
    parser.add_argument("--model", default="", help="指定单个模型文件名（如 traffic.pt），不指定则处理全部")
    parser.add_argument("--max-batch", type=int, default=DEFAULT_MAX_BATCH,
                        help="最大 batch（engine 支持 1~此值）")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help="输入尺寸")
    parser.add_argument("--device", type=int, default=0, help="GPU 编号")
    parser.add_argument("--force", action="store_true", help="强制重导出（覆盖已有 .engine）")
    parser.add_argument("--no-fp16", action="store_true", help="禁用 FP16，使用 FP32")
    parser.add_argument("--workspace", type=int, default=8,
                        help="TensorRT workspace 大小 (GB)")
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        print(f"模型目录不存在: {args.model_dir}")
        return

    # 收集要处理的 pt 文件
    if args.model:
        pt_files = [args.model] if args.model.endswith(".pt") else [args.model + ".pt"]
    else:
        pt_files = sorted(os.listdir(args.model_dir))
        pt_files = [f for f in pt_files if f.endswith(".pt")]
        if args.prefix:
            pt_files = [f for f in pt_files if f.startswith(args.prefix)]
            if not pt_files:
                print(f"未找到以 '{args.prefix}' 开头的 .pt 文件")
                return

    exported = []
    failed = []

    for filename in pt_files:
        if not filename.endswith(".pt"):
            continue
        pt_path = os.path.join(args.model_dir, filename)
        engine_path = pt_path.replace(".pt", ".engine")

        if os.path.exists(engine_path) and not args.force:
            print(f"跳过（engine 已存在）: {filename} | 使用 --force 可强制重导出")
            exported.append(filename)
            continue

        # 先尝试 FP16，失败则自动回退 FP32
        fp16_attempts = [not args.no_fp16] if args.no_fp16 else [True, False]
        succeeded = False

        for use_fp16 in fp16_attempts:
            label = "FP16" if use_fp16 else "FP32"
            print(f"\n{'='*50}")
            print(f"尝试 {label} 导出: {filename}")
            print('='*50)
            if export_engine(pt_path, args.max_batch, args.imgsz, args.device,
                             enable_fp16=use_fp16):
                succeeded = True
                exported.append(filename)
                break
            if not args.no_fp16 and use_fp16:
                print(f"→ FP16 失败，自动回退 FP32 重试...")

        if not succeeded:
            failed.append(filename)

    print(f"\n{'='*50}")
    print(f"总结: 成功={len(exported)} | 失败={len(failed)}")
    if failed:
        print(f"失败列表: {failed}")
    if exported and args.force:
        print("提示: engine 已重导出，请重启服务使新 engine 生效。")


if __name__ == "__main__":
    main()
