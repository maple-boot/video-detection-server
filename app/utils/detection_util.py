import math
import cv2
import numpy as np
from app.utils.logger import get_task_logger


class DetectionUtils:
    """检测结果工具类：IoU、pHash、去重过滤"""

    def __init__(self, task_id: str = "system"):
        self.logger = get_task_logger(task_id)

    @staticmethod
    def calculate_iou(box1: list, box2: list) -> float:
        """计算两个框的 IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def is_valid_box(box: list) -> bool:
        """检查检测框是否有效（跳过 NaN）"""
        return not any(math.isnan(v) for v in box)

    @staticmethod
    def nms_merge(detections: list, iou_threshold: float = 0.5) -> list:
        """按类别的 NMS 合并，保留置信度最高的框"""
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
                    iou = DetectionUtils.calculate_iou(best["bbox"], det["bbox"])
                    if iou < iou_threshold:
                        remaining.append(det)
                dets = remaining
            merged.extend(keep)

        return merged

    @staticmethod
    def calculate_box_ratio(box: list, img_width: int, img_height: int) -> float:
        """计算识别框占图片面积比例"""
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        img_area = img_width * img_height
        return box_area / img_area if img_area > 0 else 0.0

    @staticmethod
    def phash(image: np.ndarray, hash_size: int = 8) -> np.ndarray:
        """计算图片的感知哈希 (pHash)"""
        resized = cv2.resize(image, (hash_size * 4, hash_size * 4))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
        gray = np.float32(gray)
        dct = cv2.dct(gray)
        dct_low = dct[:hash_size, :hash_size]
        median = np.median(dct_low)
        return (dct_low > median).flatten().astype(np.uint8)

    @staticmethod
    def hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
        """计算两个哈希的汉明距离"""
        return np.sum(hash1 != hash2)

    def filter_detections(self, detections: list, img_width: int, img_height: int,
                          conf_threshold: float = 0.8, max_box_ratio: float = 0.4,
                          allowed_classes: list = None) -> list:
        """过滤检测结果：置信度、框面积、类别白名单"""
        filtered = []
        for det in detections:
            if det["confidence"] < conf_threshold:
                continue

            ratio = self.calculate_box_ratio(det["bbox"], img_width, img_height)
            if ratio > max_box_ratio:
                self.logger.debug(f"框面积过大，舍弃 | ratio={ratio:.2f} | class={det['class_name']}")
                continue

            if allowed_classes and det["class_name"] not in allowed_classes:
                continue

            filtered.append(det)

        return filtered

    def is_duplicate_frame(self, frame1: np.ndarray, frame2: np.ndarray,
                           threshold: int = 10) -> bool:
        """通过 pHash 判断两帧是否重复"""
        if frame1 is None or frame2 is None:
            return False
        hash1 = self.phash(frame1)
        hash2 = self.phash(frame2)
        distance = self.hamming_distance(hash1, hash2)
        return distance < threshold
