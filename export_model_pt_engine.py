import os
import time
from ultralytics import YOLO

model_dir = "models"
exported = []
failed = []

for filename in sorted(os.listdir(model_dir)):
    if not filename.endswith(".pt"):
        continue

    pt_path = os.path.join(model_dir, filename)
    engine_path = pt_path.replace(".pt", ".engine")

    if os.path.exists(engine_path):
        print(f"跳过（已存在）: {filename}")
        continue

    try:
        print(f"转换中: {filename} ...")
        t0 = time.time()
        model = YOLO(pt_path)
        model.export(
            format="engine",
            device=0,
            half=True,
            workspace=4,
            batch=8,           # ← SAHI 最大并行切片数
            imgsz=640,         # ← 与 SAHI slice_size 一致
        )
        elapsed = time.time() - t0
        print(f"完成: {filename} → {filename.replace('.pt', '.engine')} ({elapsed:.1f}s)")
        exported.append(filename)
    except Exception as e:
        print(f"失败: {filename} | error={e}")
        failed.append(filename)

print(f"\n总结: 成功={len(exported)} | 失败={len(failed)}")
if failed:
    print(f"失败列表: {failed}")
