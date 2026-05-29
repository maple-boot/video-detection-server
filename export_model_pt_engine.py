"""将 models/ 下的 .pt 模型导出为 TensorRT .engine（使用 Ultralytics 内置导出，保留动态 batch + 元数据）"""

import argparse
import os
import time
from ultralytics import YOLO

DEFAULT_MODEL_DIR = "models"
DEFAULT_MAX_BATCH = 8
DEFAULT_IMGSZ = 640


def export_engine(pt_path: str, max_batch: int, imgsz: int, device: int,
                  enable_fp16: bool = True) -> bool:
    """.pt → .engine，使用 Ultralytics 内置导出（自动嵌入元数据，支持动态 batch）"""
    engine_path = pt_path.replace(".pt", ".engine")
    filename = os.path.basename(pt_path)

    print(f"导出 Engine: {filename} | dynamic_batch=1~{max_batch} | imgsz={imgsz} | device={device} ...")
    t0 = time.time()

    try:
        model = YOLO(pt_path)
        model.export(
            format="engine",
            half=enable_fp16,
            batch=max_batch,
            dynamic=True,
            imgsz=imgsz,
            device=device,
            workspace=8,  # GB，与原有 --workspace 8192MB 一致
        )
        elapsed = time.time() - t0
        size_mb = os.path.getsize(engine_path) / (1 << 20) if os.path.exists(engine_path) else 0
        print(f"Engine 导出成功: {os.path.basename(engine_path)} ({elapsed:.1f}s, {size_mb:.1f}MB)")
        return True
    except Exception as e:
        print(f"Engine 导出失败: {filename} | error={e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="导出 YOLO TensorRT engine（Ultralytics 内置导出，动态 batch，自动嵌入元数据）"
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="模型目录")
    parser.add_argument("--model", default="", help="指定单个模型文件名（如 traffic.pt），不指定则处理全部")
    parser.add_argument("--max-batch", type=int, default=DEFAULT_MAX_BATCH,
                        help="最大 batch（engine 支持 1~此值）")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help="输入尺寸")
    parser.add_argument("--device", type=int, default=0, help="GPU 编号")
    parser.add_argument("--force", action="store_true", help="强制重导出（覆盖已有 .engine）")
    parser.add_argument("--no-fp16", action="store_true", help="禁用 FP16，使用 FP32")
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

        if export_engine(pt_path, args.max_batch, args.imgsz, args.device,
                         enable_fp16=not args.no_fp16):
            exported.append(filename)
        else:
            failed.append(filename)

    print(f"\n总结: 成功={len(exported)} | 失败={len(failed)}")
    if failed:
        print(f"失败列表: {failed}")
    if exported and args.force:
        print("提示: engine 已重导出，请重启服务使新 engine 生效。")


if __name__ == "__main__":
    main()
