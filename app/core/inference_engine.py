import os
import time
import threading
import numpy as np
from ultralytics import YOLO
from app.utils.logger import get_task_logger, get_system_logger

logger = get_system_logger()


class InferenceEngine:
    """YOLO 推理引擎，支持 PyTorch、TensorRT、SAHI 切片检测，线程安全"""

    def __init__(self, config: dict):
        self.config = config
        self._models = {}
        self._lock = threading.Lock()
        self._model_dir = ""
        logger.info("推理引擎初始化完成")

    def load_model(self, algorithm_id: str, model_path: str, classes_path: str = "") -> bool:
        """加载模型"""
        with self._lock:
            if algorithm_id in self._models:
                logger.info(f"模型已加载，跳过 | algorithm_id={algorithm_id}")
                return True

            try:
                if not os.path.exists(model_path):
                    logger.error(f"模型文件不存在: {model_path}")
                    return False

                model = YOLO(model_path)
                classes = []
                if classes_path and os.path.exists(classes_path):
                    with open(classes_path, "r", encoding="utf-8") as f:
                        classes = [line.strip() for line in f if line.strip()]

                self._models[algorithm_id] = {
                    "model": model,
                    "classes": classes,
                    "model_path": model_path,
                    "load_time": time.time(),
                }
                logger.info(f"模型加载成功 | algorithm_id={algorithm_id} | path={model_path} | classes={len(classes)}")
                return True

            except Exception as e:
                logger.error(f"模型加载失败 | algorithm_id={algorithm_id} | error={e}")
                return False

    def detect(self, algorithm_id: str, frame: np.ndarray,
               conf: float = 0.75, imgsz: int = 640) -> tuple:
        """执行检测，返回 (detections, inference_time, raw_results)"""
        with self._lock:
            if algorithm_id not in self._models:
                logger.warning(f"模型未加载: {algorithm_id}")
                return [], 0, []

            try:
                model_info = self._models[algorithm_id]
                model = model_info["model"]
                classes = model_info["classes"]

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
                    slice_size: int = 640, overlap_ratio: float = 0.2,
                    iou_threshold: float = 0.5) -> tuple:
        """
        SAHI 切片检测
        将大图切成多个小块分别检测，再合并结果

        Args:
            algorithm_id: 算法ID
            frame: 输入图像
            conf: 置信度阈值
            imgsz: YOLO 输入尺寸
            slice_size: 切片大小
            overlap_ratio: 切片重叠比例
            iou_threshold: NMS 合并阈值

        Returns:
            (detections, inference_time, raw_results)
        """
        with self._lock:
            if algorithm_id not in self._models:
                logger.warning(f"模型未加载: {algorithm_id}")
                return [], 0, []

            try:
                model_info = self._models[algorithm_id]
                model = model_info["model"]
                classes = model_info["classes"]

                h, w = frame.shape[:2]

                # 如果图像小于切片尺寸，直接检测
                if h <= slice_size and w <= slice_size:
                    return self.detect(algorithm_id, frame, conf, imgsz)

                t0 = time.time()

                # 计算切片参数
                step = int(slice_size * (1 - overlap_ratio))
                all_detections = []

                # 生成切片坐标
                y_positions = list(range(0, max(1, h - slice_size + 1), step))
                x_positions = list(range(0, max(1, w - slice_size + 1), step))

                # 确保覆盖边缘
                if y_positions[-1] + slice_size < h:
                    y_positions.append(h - slice_size)
                if x_positions[-1] + slice_size < w:
                    x_positions.append(w - slice_size)

                # 对每个切片进行检测
                for y in y_positions:
                    for x in x_positions:
                        # 提取切片
                        y_end = min(y + slice_size, h)
                        x_end = min(x + slice_size, w)
                        slice_img = frame[y:y_end, x:x_end]

                        # 如果切片尺寸不足，pad 到 slice_size
                        sh, sw = slice_img.shape[:2]
                        if sh < slice_size or sw < slice_size:
                            padded = np.full((slice_size, slice_size, 3), 114, dtype=np.uint8)
                            padded[:sh, :sw] = slice_img
                            slice_img = padded

                        # YOLO 检测
                        results = model.predict(
                            slice_img,
                            conf=conf,
                            imgsz=imgsz,
                            verbose=False,
                            device="0",
                        )

                        # 提取结果并映射回原始坐标
                        for result in results:
                            boxes = result.boxes
                            if boxes is None:
                                continue

                            for i in range(len(boxes)):
                                box = boxes.xyxy[i].cpu().numpy()
                                conf_val = float(boxes.conf[i].cpu().numpy())
                                cls_id = int(boxes.cls[i].cpu().numpy())

                                # 坐标映射回原始图
                                x1 = float(box[0]) + x
                                y1 = float(box[1]) + y
                                x2 = float(box[2]) + x
                                y2 = float(box[3]) + y

                                # 裁剪到图像边界
                                x1 = max(0, min(x1, w))
                                y1 = max(0, min(y1, h))
                                x2 = max(0, min(x2, w))
                                y2 = max(0, min(y2, h))

                                class_name = classes[cls_id] if cls_id < len(classes) else str(cls_id)

                                all_detections.append({
                                    "bbox": [x1, y1, x2, y2],
                                    "confidence": conf_val,
                                    "class_id": cls_id,
                                    "class_name": class_name,
                                })

                inference_time = (time.time() - t0) * 1000

                # NMS 合并重叠检测
                merged_detections = self._nms_merge(all_detections, iou_threshold)

                logger.debug(
                    f"SAHI 检测完成 | 切片数={len(y_positions) * len(x_positions)} | "
                    f"原始检测={len(all_detections)} | 合并后={len(merged_detections)} | "
                    f"耗时={inference_time:.1f}ms"
                )

                return merged_detections, inference_time, []

            except Exception as e:
                logger.error(f"SAHI 检测异常 | algorithm_id={algorithm_id} | error={e}")
                return [], 0, []

    @staticmethod
    def _nms_merge(detections: list, iou_threshold: float = 0.5) -> list:
        """NMS 合并重叠检测框"""
        if not detections:
            return []

        # 按类别分组
        class_groups = {}
        for det in detections:
            cls_id = det["class_id"]
            if cls_id not in class_groups:
                class_groups[cls_id] = []
            class_groups[cls_id].append(det)

        merged = []
        for cls_id, dets in class_groups.items():
            # 按置信度降序排序
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
        """计算两个框的 IoU"""
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
        """卸载模型"""
        with self._lock:
            if algorithm_id in self._models:
                del self._models[algorithm_id]
                logger.info(f"模型已卸载 | algorithm_id={algorithm_id}")

    def get_loaded_models(self) -> list:
        """获取已加载的模型列表"""
        return list(self._models.keys())

    def clear_cache(self):
        """清空模型缓存，释放显存"""
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
