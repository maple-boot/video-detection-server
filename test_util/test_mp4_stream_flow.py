"""
MP4 文件 YOLO 推理测试 — 按实时流(StreamWorker)处理流程
保留：自适应检测间隔、SAHI切片、ByteTrack追踪、PIL中文绘框
去掉：结果上报、坐标信息、MinIO上传、HTTP回调

用法:
    python test_mp4_stream_flow.py --input input.mp4 --model_path models/best.pt --algorithm_id 1
    python test_mp4_stream_flow.py --input input.mp4 --model_path models/best.pt --algorithm_id 1 --output result.mp4
    python test_mp4_stream_flow.py --input input.mp4 --model_path models/best.pt --algorithm_id 1 --sahi
"""

import os
import sys
import time
import argparse
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.engine.results import Results, Boxes

# ─────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="MP4 文件 YOLO 推理测试（实时流处理流程）")
    parser.add_argument("--input", type=str, required=True, help="输入 MP4 文件路径")
    parser.add_argument("--output", type=str, default=None, help="输出 MP4 文件路径（默认: input_stamped.mp4）")
    parser.add_argument("--model_path", type=str, required=True, help="YOLO 模型路径 (.pt 或 .engine)")
    parser.add_argument("--classes_path", type=str, default="", help="类别文件路径")
    parser.add_argument("--algorithm_id", type=str, default="1", help="算法ID（用于日志标识）")
    parser.add_argument("--conf", type=float, default=0.75, help="检测置信度阈值")
    parser.add_argument("--report_conf", type=float, default=0.8, help="过滤置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸")
    parser.add_argument("--sahi", action="store_true", help="启用 SAHI 切片检测")
    parser.add_argument("--slice_size", type=int, default=640, help="SAHI 切片大小")
    parser.add_argument("--overlap_ratio", type=float, default=0.1, help="SAHI 切片重叠比例")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="SAHI NMS 合并 IoU 阈值")
    parser.add_argument("--device", type=str, default="0", help="推理设备（如 0, 0,1, cpu）")
    parser.add_argument("--warmup_frames", type=int, default=15, help="预热帧数（跳过脏数据）")
    parser.add_argument("--max_box_ratio", type=float, default=0.4, help="最大框面积比例（过滤过大框）")
    parser.add_argument("--smooth_alpha", type=float, default=0.2, help="检测框EMA平滑系数 (0,1]，越小越平滑（0=关闭平滑, 0.2=推荐值）")
    parser.add_argument("--smooth_noise", type=int, default=12, help="检测框噪声过滤阈值（像素），偏移小于此值不移动框")
    return parser.parse_args()


# ─────────────────────────────────────────────
# TrackArgs — ByteTrack 追踪参数（与 StreamWorker 一致）
# ─────────────────────────────────────────────
class TrackArgs:
    def __init__(self):
        self.track_high_thresh = 0.5
        self.track_low_thresh = 0.1
        self.new_track_thresh = 0.5
        self.track_buffer = 10
        self.match_thresh = 0.7
        self.fuse_score = True
        self.min_box_area = 10
        self.mot20 = False


# ─────────────────────────────────────────────
# SAHI 切片
# ─────────────────────────────────────────────
def create_slices(frame, slice_size, overlap_ratio):
    """SAHI 切片（与 InferenceEngine._create_slices 一致）"""
    h, w = frame.shape[:2]
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
    return slices, slice_coords


def nms_merge(detections, iou_threshold=0.5):
    """按类别的 NMS 合并（与 DetectionUtils.nms_merge 一致）"""
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
                x1 = max(best["bbox"][0], det["bbox"][0])
                y1 = max(best["bbox"][1], det["bbox"][1])
                x2 = min(best["bbox"][2], det["bbox"][2])
                y2 = min(best["bbox"][3], det["bbox"][3])
                intersection = max(0, x2 - x1) * max(0, y2 - y1)
                area1 = (best["bbox"][2] - best["bbox"][0]) * (best["bbox"][3] - best["bbox"][1])
                area2 = (det["bbox"][2] - det["bbox"][0]) * (det["bbox"][3] - det["bbox"][1])
                union = area1 + area2 - intersection
                iou = intersection / union if union > 0 else 0.0
                if iou < iou_threshold:
                    remaining.append(det)
            dets = remaining
        merged.extend(keep)
    return merged


# ─────────────────────────────────────────────
# 绘框（与 StreamWorker._draw_boxes 一致，PIL 中文支持）
# ─────────────────────────────────────────────
def get_chinese_font(size=16):
    font_paths = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/msyh.ttc",       # Windows 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",      # Windows 黑体
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


TRACK_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255),
    (255, 255, 0), (0, 255, 255), (255, 0, 255),
    (128, 255, 0), (255, 128, 0), (0, 128, 255),
    (128, 0, 255), (255, 128, 128), (128, 255, 128),
]


def draw_boxes(frame, detections, conf_threshold=0.75):
    """绘制检测框（PIL，含 track_id + 中文支持）"""
    if not detections:
        return frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = get_chinese_font(16)
    for det in detections:
        if det.get("confidence", 0) < conf_threshold:
            continue
        box = det["bbox"]
        if any(np.isnan(v) for v in box):
            continue
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        track_id = det.get("track_id", 0)
        color = TRACK_COLORS[track_id % len(TRACK_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        class_name = det.get("class_name", "")
        conf_val = det.get("confidence", 0)
        label = f"ID:{track_id} {class_name} {conf_val:.2f}"
        text_bbox = draw.textbbox((x1, y1 - 25), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        if y1 - text_h - 5 > 0:
            draw.rectangle([x1, y1 - text_h - 5, x1 + text_w + 5, y1], fill=color)
            draw.text((x1 + 2, y1 - text_h - 3), label, fill=(255, 255, 255), font=font)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────
# 检测 + 追踪
# ─────────────────────────────────────────────
def run_sahi_detect(model, frame, classes, conf, imgsz, slice_size, overlap_ratio, iou_threshold, device):
    """SAHI 切片检测（与 InferenceEngine.detect_sahi 逻辑一致）"""
    h, w = frame.shape[:2]
    if h <= slice_size and w <= slice_size:
        # 图片小于切片大小，直接全图检测
        results = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False, device=device)
        detections = parse_results(results, classes)
        return detections, results

    slices, slice_coords = create_slices(frame, slice_size, overlap_ratio)

    all_results = []
    for s in slices:
        results = model.predict(s, conf=conf, imgsz=imgsz, verbose=False, device=device)
        all_results.extend(results)

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

    merged = nms_merge(all_detections, iou_threshold)
    return merged, []


def parse_results(results, classes):
    """解析 YOLO 检测结果"""
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
    return detections


def track_from_detections(tracker, detections, frame, classes):
    """从检测结果构造 Boxes 进行 ByteTrack 追踪（与 StreamWorker._track_from_detections 一致）"""
    try:
        if not detections:
            empty_boxes = Boxes(torch.zeros((0, 6), dtype=torch.float32), frame.shape[:2]).numpy()
            tracks = tracker.update(empty_boxes, frame.shape[:2])
        else:
            det_array = []
            for det in detections:
                bbox = det["bbox"]
                det_array.append([bbox[0], bbox[1], bbox[2], bbox[3], det["confidence"], det["class_id"]])
            det_tensor = torch.tensor(det_array, dtype=torch.float32)
            boxes = Boxes(det_tensor, frame.shape[:2]).numpy()
            tracks = tracker.update(boxes, frame.shape[:2])

        if len(tracks) == 0:
            return detections

        tracked = []
        for track in tracks:
            x1, y1, x2, y2 = float(track[0]), float(track[1]), float(track[2]), float(track[3])
            track_id = int(track[4])
            conf = float(track[5])
            cls_id = int(track[6])
            class_name = classes[cls_id] if cls_id < len(classes) else str(cls_id)
            tracked.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls_id,
                "class_name": class_name,
                "track_id": track_id,
            })
        return tracked
    except Exception as e:
        print(f"[WARN] ByteTrack 追踪异常: {e}")
        return detections


def track_with_raw_results(tracker, detections, frame, classes, raw_results):
    """用 YOLO 原始结果做 ByteTrack 追踪（与 StreamWorker._track 一致）"""
    try:
        if raw_results and len(raw_results) > 0 and raw_results[0].boxes is not None and len(raw_results[0].boxes) > 0:
            boxes = raw_results[0].boxes.cpu().numpy()
            tracks = tracker.update(boxes, frame.shape[:2])
        else:
            empty_boxes = Boxes(torch.zeros((0, 6), dtype=torch.float32), frame.shape[:2]).numpy()
            tracks = tracker.update(empty_boxes, frame.shape[:2])
    except Exception as e:
        print(f"[WARN] ByteTrack 更新异常: {e}")
        return detections

    if len(tracks) == 0:
        return detections

    tracked = []
    for track in tracks:
        x1, y1, x2, y2 = float(track[0]), float(track[1]), float(track[2]), float(track[3])
        track_id = int(track[4])
        conf = float(track[5])
        cls_id = int(track[6])
        class_name = classes[cls_id] if cls_id < len(classes) else str(cls_id)
        tracked.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "class_id": cls_id,
            "class_name": class_name,
            "track_id": track_id,
        })
    return tracked


# ─────────────────────────────────────────────
# 检测框平滑（EMA + IoU 匹配回退）
# 解决跳帧推理导致的识别框跨帧抖动问题
# ─────────────────────────────────────────────
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
    
    核心思路：
    - 检测帧：用 EMA 平滑目标位置（alpha 越小越平滑）
    - 跳过的帧：保持上次平滑后的位置不变（不做插值，避免跳变）
    - 消失轨迹保留 grace_period 帧，用 IoU 回退匹配 track_id 重分配
    """
    def __init__(self, alpha=0.35, iou_threshold=0.3, grace_period=15, noise_threshold=8):
        """
        Args:
            alpha: EMA 平滑系数 (0,1]。越小越平滑。
                   0.35 = 每次移动35%的距离，兼顾平滑和响应。
            iou_threshold: IoU 回退匹配阈值
            grace_period: 消失轨迹保留帧数
            noise_threshold: 噪声过滤阈值（像素）。当检测框中心偏移小于此值时，
                             视为检测噪声，不移动框位置。
        """
        self.alpha = alpha
        self.iou_threshold = iou_threshold
        self.grace_period = grace_period
        self.noise_threshold = noise_threshold
        # track_id -> {"bbox": [x1,y1,x2,y2], "confidence":..., "class_name":..., "class_id":...}
        self._tracks = {}
        self._stale = {}  # track_id -> ([x1,y1,x2,y2], ttl)

    def update_on_detection(self, detections, detection_interval):
        """
        检测帧调用：对每个 track 的 bbox 做 EMA 平滑。
        原地修改 bbox。
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
                # 计算中心点偏移
                prev_cx = (prev[0] + prev[2]) / 2.0
                prev_cy = (prev[1] + prev[3]) / 2.0
                det_cx = (bbox[0] + bbox[2]) / 2.0
                det_cy = (bbox[1] + bbox[3]) / 2.0
                cx_diff = abs(det_cx - prev_cx)
                cy_diff = abs(det_cy - prev_cy)
                # 宽高也检查，防止单纯 size 抖动
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
                # IoU 匹配：继承历史位置，EMA 平滑
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
        不做插值，框在两次检测帧之间保持稳定位置。
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


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    args = parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"[ERROR] 输入文件不存在: {input_path}")
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_result{ext}"

    print("=" * 60)
    print("MP4 文件 YOLO 推理测试（实时流处理流程）")
    print("=" * 60)
    print(f"  输入文件:     {input_path}")
    print(f"  输出文件:     {output_path}")
    print(f"  模型路径:     {args.model_path}")
    print(f"  类别文件:     {args.classes_path or '(自动)'}")
    print(f"  算法ID:       {args.algorithm_id}")
    print(f"  置信度:       {args.conf}")
    print(f"  推理尺寸:     {args.imgsz}")
    print(f"  设备:         {args.device}")
    print(f"  SAHI:         {'启用' if args.sahi else '关闭'}")
    if args.sahi:
        print(f"  切片大小:     {args.slice_size}")
        print(f"  重叠比例:     {args.overlap_ratio}")
        print(f"  NMS IoU:      {args.iou_threshold}")
    print(f"  预热帧数:     {args.warmup_frames}")
    print(f"  平滑系数:     {args.smooth_alpha} ({'关闭' if args.smooth_alpha <= 0 else 'EMA平滑'})")
    print("=" * 60)

    # ── 1. 加载类别 ──
    classes = []
    classes_path = args.classes_path
    if not classes_path:
        auto_path = args.model_path.rsplit(".pt", 1)[0] + "_classes.txt"
        if os.path.exists(auto_path):
            classes_path = auto_path
    if classes_path and os.path.exists(classes_path):
        with open(classes_path, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
        print(f"[INFO] 加载 {len(classes)} 个类别: {classes_path}")
    else:
        print("[WARN] 未找到类别文件，将使用类别ID")

    # ── 2. 加载模型 ──
    # TensorRT 优先
    model_path = args.model_path
    if model_path.endswith(".pt"):
        engine_path = model_path.rsplit(".pt", 1)[0] + ".engine"
        if os.path.exists(engine_path):
            model_path = engine_path
            print(f"[INFO] 找到 TensorRT engine: {engine_path}")

    print(f"[INFO] 加载模型: {model_path}")
    model = YOLO(model_path)
    is_tensorrt = model_path.endswith(".engine")

    # 从 engine 元数据读取 batch
    batch_size = 8
    if is_tensorrt:
        import pickle
        import json
        try:
            with open(model_path, "rb") as f:
                data = f.read()
            magic = b"UlTralYtiCsEnGiNe"
            idx = data.rfind(magic)
            if idx >= 0:
                meta = pickle.loads(data[idx + len(magic):])
                if "batch" in meta:
                    batch_size = int(meta["batch"])
                    print(f"[INFO] Engine batch_size: {batch_size}")
        except Exception:
            pass

    # 预热
    print("[INFO] 模型预热中...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy, verbose=False, device=args.device)
    print("[INFO] 模型预热完成")

    # ── 3. 初始化视频捕获 ──
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {input_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] 视频属性: {width}x{height} @ {fps:.1f}fps | 总帧数: {total_frames}")

    # ── 4. 初始化视频写入器 ──
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"[ERROR] 无法创建输出视频: {output_path}")
        cap.release()
        sys.exit(1)

    # ── 5. 初始化 ByteTrack ──
    track_args = TrackArgs()
    tracker = BYTETracker(args=track_args, frame_rate=int(fps))
    print("[INFO] ByteTrack 追踪器已初始化")

    # 初始化框平滑器（EMA + 噪声过滤）
    box_smoother = BoxSmoother(alpha=args.smooth_alpha, noise_threshold=args.smooth_noise) if args.smooth_alpha > 0 else None
    if box_smoother:
        print(f"[INFO] 检测框平滑已启用 | alpha={args.smooth_alpha} | noise_threshold={args.smooth_noise}")

    # ── 6. 主处理循环（与 StreamWorker._run_loop 核心逻辑一致）──
    frame_count = 0
    skip_counter = 0
    detection_interval = 1  # 自适应检测间隔
    warmup_frames = args.warmup_frames

    # 检测统计
    total_infer_time = 0
    detect_frame_count = 0
    total_tracks_seen = set()

    # 最近一次检测帧的原始 detections（用于跳帧时 ByteTrack 追踪）
    last_raw_detections = []
    last_raw_results = []

    print("[INFO] 开始处理...")
    t_start_all = time.time()

    while True:
        t_start = time.time()

        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        skip_counter += 1

        # 预热阶段：跳过检测，直接写原始帧
        if warmup_frames > 0:
            warmup_frames -= 1
            writer.write(frame)
            if warmup_frames == 0:
                # 重置追踪器（与 StreamWorker 一致）
                tracker = BYTETracker(args=track_args, frame_rate=int(fps))
                if box_smoother:
                    box_smoother.reset()
                last_raw_detections = []
                last_raw_results = []
                print(f"[INFO] 预热完成，开始检测（跳过 {args.warmup_frames} 帧）")
            continue

        # ── 检测帧 or 跳帧 ──
        is_detection_frame = (skip_counter >= detection_interval)

        if is_detection_frame:
            # ── 执行检测 ──
            use_sahi = args.sahi and is_tensorrt

            if use_sahi:
                detections, raw_results = run_sahi_detect(
                    model, frame, classes,
                    conf=args.conf, imgsz=args.imgsz,
                    slice_size=args.slice_size,
                    overlap_ratio=args.overlap_ratio,
                    iou_threshold=args.iou_threshold,
                    device=args.device,
                )
                inference_time = 0
            else:
                t_infer = time.time()
                results = model.predict(
                    frame, conf=args.conf, imgsz=args.imgsz,
                    verbose=False, device=args.device,
                )
                inference_time = (time.time() - t_infer) * 1000
                detections = parse_results(results, classes)
                raw_results = results

            detect_frame_count += 1
            total_infer_time += inference_time

            # 自适应调整间隔
            frame_interval = 1000.0 / fps
            if inference_time < frame_interval * 0.8:
                detection_interval = 2
            elif inference_time < frame_interval * 1.5:
                detection_interval = 2
            elif inference_time < frame_interval * 3:
                detection_interval = 4
            elif inference_time < frame_interval * 5:
                detection_interval = 8
            else:
                detection_interval = 10

            skip_counter = 0

            # ByteTrack 追踪
            if use_sahi and detections:
                detections = track_from_detections(tracker, detections, frame, classes)
            elif raw_results:
                detections = track_with_raw_results(tracker, detections, frame, classes, raw_results)

            # 统计轨迹
            for det in detections:
                if det.get("track_id") is not None:
                    total_tracks_seen.add(det["track_id"])

            # 保存原始检测结果（不用于绘图，仅保留备查）
            last_raw_detections = detections
            last_raw_results = raw_results

            # ── 框平滑：检测帧更新目标位置 ──
            if box_smoother:
                box_smoother.update_on_detection(detections, detection_interval)
                # 检测帧的绘图数据 = 更新后的 detections（bbox 已是最新检测值）
                draw_detections = detections
            else:
                draw_detections = detections
        else:
            # ── 跳帧：从平滑器获取插值位置 ──
            if box_smoother:
                draw_detections = box_smoother.get_display_boxes()
            else:
                draw_detections = last_raw_detections

        # ── 绘框（draw_detections 已包含 confidence/class_name/track_id）──
        annotated = draw_boxes(frame.copy(), draw_detections, conf_threshold=args.conf)

        # ── 写入输出 ──
        writer.write(annotated)

        # ── 进度日志 ──
        t_total = (time.time() - t_start) * 1000
        if frame_count % 100 == 0:
            elapsed = time.time() - t_start_all
            progress = frame_count / total_frames * 100 if total_frames > 0 else 0
            avg_fps = frame_count / elapsed if elapsed > 0 else 0
            avg_infer = total_infer_time / detect_frame_count if detect_frame_count > 0 else 0
            print(
                f"[PROGRESS] Frame {frame_count}/{total_frames} ({progress:.1f}%) | "
                f"total={t_total:.1f}ms | avg_fps={avg_fps:.1f} | "
                f"avg_infer={avg_infer:.1f}ms | tracks={len(total_tracks_seen)} | "
                f"interval={detection_interval}"
            )

    # ── 7. 结束，上报剩余轨迹统计（不实际上报，仅统计）──
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    elapsed = time.time() - t_start_all
    avg_fps = frame_count / elapsed if elapsed > 0 else 0
    avg_infer = total_infer_time / detect_frame_count if detect_frame_count > 0 else 0

    print(f"  总帧数:         {frame_count}")
    print(f"  检测帧数:       {detect_frame_count}")
    print(f"  跳帧间隔(最终): {detection_interval}")
    print(f"  总耗时:         {elapsed:.1f}s")
    print(f"  平均 FPS:       {avg_fps:.1f}")
    print(f"  平均推理耗时:   {avg_infer:.1f}ms")
    print(f"  检测到轨迹数:   {len(total_tracks_seen)}")
    print(f"  输出文件:       {output_path}")
    print("=" * 60)

    # 释放资源
    cap.release()
    writer.release()
    print("[INFO] 资源已释放")


if __name__ == "__main__":
    main()