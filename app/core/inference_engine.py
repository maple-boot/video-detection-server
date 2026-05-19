import os
import time
import threading
import numpy as np
from ultralytics import YOLO
from app.utils.logger import get_task_logger, get_system_logger

logger = get_system_logger()


class InferenceEngine:
    """YOLO 推理引擎，支持 TensorRT、SAHI 切片检测"""

    def __init__(self, config: dict):
        self.config = config
        self._models = {}
        self._lock = threading.Lock()
        logger.info("推理引擎初始化完成")

    def load_model(self, algorithm_id: str, model_path: str, classes_path: str = "") -> bool:
        """加载模型，优先加载 TensorRT engine"""
        with self._lock:
            if algorithm_id in self._models:
                logger.info(f"模型已加载，跳过 | algorithm_id={algorithm_id}")
                return True

            try:
                # 优先加载 TensorRT engine
                engine_path = model_path.replace(".pt", ".engine")
                actual_path = engine_path if os.path.exists(engine_path) else model_path

                if not os.path.exists(actual_path):
                    logger.error(f"模型文件不存在: {actual_path}")
                    return False

                is_tensorrt = actual_path.endswith(".engine")
                logger.info(f"加载模型 | algorithm_id={algorithm_id} | path={actual_path} | tensorrt={is_tensorrt}")

                model = YOLO(actual_path)
                classes = []
                if classes_path and os.path.exists(classes_path):
                    with open(classes_path, "r", encoding="utf-8") as f:
                        classes = [line.strip() for line in f if line.strip()]

                self._models[algorithm_id] = {
                    "model": model,
                    "classes": classes,
                    "model_path": actual_path,
                    "load_time": time.time(),
                    "is_tensorrt": is_tensorrt,
                }
                logger.info(f"模型加载成功 | algorithm_id={algorithm_id} | classes={len(classes)}")
                return True

            except Exception as e:
                logger.error(f"模型加载失败 | algorithm_id={algorithm_id} | error={e}")
                return False

    def detect(self, algorithm_id: str, frame: np.ndarray,
               conf: float = 0.75, imgsz: int = 640) -> tuple:
        """执行检测，返回 (detections, inference_time, raw_results)"""
        if algorithm_id not in self._models:
            return [], 0, []

        try:
            model_info = self._models[algorithm_id]
            model = model_info["model"]
            classes = model_info["classes"]

            with self._lock:
                t0 = time.time()
                results = model.predict(
                    frame,
                    conf=conf,
                    imgsz=imgsz,
                    verbose=False,
                    device="0",
                )
                inference_time = (time.time() - t0) * 1000

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    conf_val = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    class_name = classes[cls_id] if cls_id < len(classes) else str(cls_id)
                    detections.append({
                        "bbox": box.tolist(),
                        "confidence": conf_val,
                        "class_id": cls_id,
                        "class_name": class_name,
                    })

            return detections, inference_time, results
        except Exception as e:
            logger.error(f"推理异常 | algorithm_id={algorithm_id} | error={e}")
            return [], 0, []

    def detect_sahi(self, algorithm_id: str, frame: np.ndarray,
                    conf: float = 0.75, imgsz: int = 640,
                    slice_size: int = 640, overlap_ratio: float = 0.1,
                    iou_threshold: float = 0.5) -> tuple:
        """SAHI 切片检测"""
        if algorithm_id not in self._models:
            return [], 0, []

        try:
            model_info = self._models[algorithm_id]
            model = model_info["model"]
            classes = model_info["classes"]
            is_tensorrt = model_info.get("is_tensorrt", False)

            h, w = frame.shape[:2]

            if h <= slice_size and w <= slice_size:
                return self.detect(algorithm_id, frame, conf, imgsz)

            step = int(slice_size * (1 - overlap_ratio))
            y_positions = list(range(0, max(1, h - slice_size + 1), step))
            x_positions = list(range(0, max(1, w - slice_size + 1), step))

            if y_positions[-1] + slice_size < h:
                y_positions.append(h - slice_size)
            if x_positions[-1] + slice_size < w:
                x_positions.append(w - slice_size)

            slices = []
            slice_coords = []
            for y in y_positions:
                for x in x_positions:
                    y_end = min(y + slice_size, h)
                    x_end = min(x + slice_size, w)
                    slice_img = frame[y:y_end, x:x_end]

                    sh, sw = slice_img.shape[:2]
                    if sh < slice_size or sw < slice_size:
                        padded = np.full((slice_size, slice_size, 3), 114, dtype=np.uint8)
                        padded[:sh, :sw] = slice_img
                        slice_img = padded

                    slices.append(slice_img)
                    slice_coords.append((x, y))

            actual_count = len(slices)

            with self._lock:
                t0 = time.time()
                # TensorRT：填充到 batch=8 的整数倍
                if is_tensorrt:
                    batch_size = 8
                    if actual_count % batch_size != 0:
                        pad_count = batch_size - (actual_count % batch_size)
                        dummy = np.full((slice_size, slice_size, 3), 114, dtype=np.uint8)
                        slices.extend([dummy] * pad_count)
                # 批量推理
                all_results = model.predict(
                    slices,
                    conf=conf,
                    imgsz=imgsz,
                    verbose=False,
                    device="0",
                    batch=8 if is_tensorrt else len(slices),
                )
                # 只取实际切片的结果，丢弃填充的
                all_results = all_results[:actual_count]
                inference_time = (time.time() - t0) * 1000
            # 映射坐标回原图
            all_detections = []
            for idx, result in enumerate(all_results):
                x_off, y_off = slice_coords[idx]
                boxes = result.boxes
                if boxes is None:
                    continue

                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    conf_val = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())

                    x1 = max(0, min(float(box[0]) + x_off, w))
                    y1 = max(0, min(float(box[1]) + y_off, h))
                    x2 = max(0, min(float(box[2]) + x_off, w))
                    y2 = max(0, min(float(box[3]) + y_off, h))

                    class_name = classes[cls_id] if cls_id < len(classes) else str(cls_id)

                    all_detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf_val,
                        "class_id": cls_id,
                        "class_name": class_name,
                    })

            merged = self._nms_merge(all_detections, iou_threshold)

            logger.debug(
                f"SAHI | slices={actual_count} | padded={len(slices)} | tensorrt={is_tensorrt} | "
                f"raw={len(all_detections)} | merged={len(merged)} | time={inference_time:.1f}ms"
            )

            return merged, inference_time, []

        except Exception as e:
            logger.error(f"SAHI 检测异常 | algorithm_id={algorithm_id} | error={e}")
            return [], 0, []

    @staticmethod
    def _nms_merge(detections: list, iou_threshold: float = 0.5) -> list:
        if not detections:
            return []

        class_groups = {}
        for det in detections:
            cls_id = det["class_id"]
            if cls_id not in class_groups:
                class_groups[cls_id] = []
            class_groups[cls_id].append(det)

        merged = []
        for cls_id, dets in class_groups.items():
            dets.sort(key=lambda x: x["confidence"], reverse=True)
            keep = []
            while dets:
                best = dets.pop(0)
                keep.append(best)
                remaining = []
                for det in dets:
                    iou = InferenceEngine._calculate_iou(best["bbox"], det["bbox"])
                    if iou < iou_threshold:
                        remaining.append(det)
                dets = remaining
            merged.extend(keep)

        return merged

    @staticmethod
    def _calculate_iou(box1: list, box2: list) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        if inter_area == 0:
            return 0.0

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter_area / (box1_area + box2_area - inter_area)

    def unload_model(self, algorithm_id: str):
        with self._lock:
            if algorithm_id in self._models:
                del self._models[algorithm_id]
                logger.info(f"模型已卸载 | algorithm_id={algorithm_id}")

    def get_loaded_models(self) -> list:
        return list(self._models.keys())

    def clear_cache(self):
        with self._lock:
            self._models.clear()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("GPU 显存缓存已清空")
            except ImportError:
                pass
            logger.info("所有模型已卸载")
