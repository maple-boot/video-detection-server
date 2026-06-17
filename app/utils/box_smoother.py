"""
检测框平滑器 — 消除跳帧推理导致的识别框抖动和闪烁。

核心思路：
- 检测帧：用 EMA 平滑目标位置（alpha 越小越平滑）
- 跳过的帧：保持上次平滑后的位置不变
- 消失轨迹保留 grace_period 帧，用 IoU 回退匹配 track_id 重分配
- 噪声过滤：中心偏移 < noise_threshold 像素时忽略

用法：
    smoother = BoxSmoother(alpha=0.2, noise_threshold=12)
    # 检测帧
    smoothed_dets = smoother.update_on_detection(detections, detection_interval)
    # 跳帧
    display_dets = smoother.get_display_boxes()
"""

import numpy as np


def _iou(box_a, box_b):
    """计算两个 [x1,y1,x2,y2] 框的 IoU"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class BoxSmoother:
    """
    检测框 EMA 平滑器，解决跳帧推理导致的框抖动和闪烁。
    """

    def __init__(self, alpha=0.2, iou_threshold=0.3, grace_period=15, noise_threshold=12):
        """
        Args:
            alpha: EMA 平滑系数 (0,1]。越小越平滑。默认 0.2。
            iou_threshold: IoU 回退匹配阈值（track_id 重分配时）
            grace_period: 消失轨迹保留帧数
            noise_threshold: 噪声过滤阈值（像素），偏移小于此值不移动框
        """
        self.alpha = alpha
        self.iou_threshold = iou_threshold
        self.grace_period = grace_period
        self.noise_threshold = noise_threshold
        # track_id -> {"bbox": [x1,y1,x2,y2], "confidence":..., "class_name":..., "class_id":...}
        self._tracks = {}
        self._stale = {}  # track_id -> ([x1,y1,x2,y2], ttl)

    def update_on_detection(self, detections, detection_interval=2):
        """
        检测帧调用：对每个 track 的 bbox 做 EMA 平滑 + 噪声过滤。
        原地修改 bbox 和 confidence，返回 detections。
        """
        current_ids = set()
        unmatched_dets = []

        for i, det in enumerate(detections):
            track_id = det.get("track_id")
            if track_id is None:
                continue
            current_ids.add(track_id)
            bbox = det["bbox"]
            if any(np.isnan(v) for v in bbox):
                continue

            if track_id in self._tracks:
                # 先判断是噪声还是真实移动
                prev = self._tracks[track_id]["bbox"]
                prev_cx = (prev[0] + prev[2]) / 2.0
                prev_cy = (prev[1] + prev[3]) / 2.0
                det_cx = (bbox[0] + bbox[2]) / 2.0
                det_cy = (bbox[1] + bbox[3]) / 2.0
                cx_diff = abs(det_cx - prev_cx)
                cy_diff = abs(det_cy - prev_cy)
                prev_w = prev[2] - prev[0]
                prev_h = prev[3] - prev[1]
                det_w = bbox[2] - bbox[0]
                det_h = bbox[3] - bbox[1]
                w_diff = abs(det_w - prev_w)
                h_diff = abs(det_h - prev_h)

                if (cx_diff < self.noise_threshold and
                        cy_diff < self.noise_threshold and
                        w_diff < self.noise_threshold and
                        h_diff < self.noise_threshold):
                    # 偏移很小，视为噪声，不移动框
                    det["bbox"] = list(prev)
                else:
                    # 真实移动，EMA 平滑
                    a = self.alpha
                    smoothed = [
                        a * bbox[0] + (1 - a) * prev[0],
                        a * bbox[1] + (1 - a) * prev[1],
                        a * bbox[2] + (1 - a) * prev[2],
                        a * bbox[3] + (1 - a) * prev[3],
                    ]
                    self._tracks[track_id]["bbox"] = smoothed
                    det["bbox"] = smoothed

                # EMA 平滑 confidence（防止在阈值边界跳变导致框闪烁）
                old_conf = self._tracks[track_id].get("confidence", 0)
                new_conf = det.get("confidence", 0)
                a = self.alpha
                smoothed_conf = a * new_conf + (1 - a) * old_conf
                self._tracks[track_id]["confidence"] = smoothed_conf
                det["confidence"] = smoothed_conf

                self._tracks[track_id]["class_name"] = det.get("class_name", "")
                self._tracks[track_id]["class_id"] = det.get("class_id", 0)
            else:
                unmatched_dets.append((i, det))

        # IoU 回退匹配
        stale_to_remove = []
        for idx, det in unmatched_dets:
            bbox = det["bbox"]
            best_iou_val = 0
            best_stale_id = None
            for stale_id, (stale_bbox, ttl) in self._stale.items():
                if stale_id in current_ids:
                    continue
                iou_val = _iou(bbox, stale_bbox)
                if iou_val > best_iou_val:
                    best_iou_val = iou_val
                    best_stale_id = stale_id

            if best_stale_id is not None and best_iou_val >= self.iou_threshold:
                stale_bbox = self._stale[best_stale_id][0]
                a = self.alpha
                smoothed = [
                    a * bbox[0] + (1 - a) * stale_bbox[0],
                    a * bbox[1] + (1 - a) * stale_bbox[1],
                    a * bbox[2] + (1 - a) * stale_bbox[2],
                    a * bbox[3] + (1 - a) * stale_bbox[3],
                ]
                self._tracks[det["track_id"]] = {
                    "bbox": smoothed,
                    "confidence": det.get("confidence", 0),
                    "class_name": det.get("class_name", ""),
                    "class_id": det.get("class_id", 0),
                }
                det["bbox"] = smoothed
                stale_to_remove.append(best_stale_id)
            else:
                # 首次出现，直接使用
                self._tracks[det["track_id"]] = {
                    "bbox": list(bbox),
                    "confidence": det.get("confidence", 0),
                    "class_name": det.get("class_name", ""),
                    "class_id": det.get("class_id", 0),
                }
                det["bbox"] = list(bbox)

        for sid in stale_to_remove:
            self._stale.pop(sid, None)

        # 本帧未出现的轨迹 → 转入保留期
        lost_ids = set(self._tracks.keys()) - current_ids
        for tid in lost_ids:
            self._stale[tid] = (list(self._tracks[tid]["bbox"]), self.grace_period)
            del self._tracks[tid]

        self._decay_stale()
        return detections

    def get_display_boxes(self):
        """
        跳帧调用：返回上次平滑后的所有 bbox（保持不动）。
        """
        display = []
        for tid, track in self._tracks.items():
            display.append({
                "track_id": tid,
                "bbox": list(track["bbox"]),
                "confidence": track.get("confidence", 0),
                "class_name": track.get("class_name", ""),
                "class_id": track.get("class_id", 0),
            })
        return display

    def _decay_stale(self):
        """保留期递减"""
        expired = []
        for tid, (bbox, ttl) in self._stale.items():
            ttl -= 1
            if ttl <= 0:
                expired.append(tid)
            else:
                self._stale[tid] = (bbox, ttl)
        for tid in expired:
            del self._stale[tid]

    def reset(self):
        """重置所有平滑状态"""
        self._tracks.clear()
        self._stale.clear()