import os
import math
import time
import threading
import aiohttp
import cv2
import numpy as np
import secrets
import string
import torch
import traceback
from datetime import datetime
from enum import Enum
from PIL import ImageFont
from queue import Queue
from PIL import Image, ImageDraw, ImageFont
from app.core.video_capture import VideoCapture
from app.utils.ffmpeg_helper import FFmpegHelper
from app.utils.detection_util import DetectionUtils
from app.utils.geo_utils import GeoUtils
from app.utils.logger import get_task_logger, get_performance_logger
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.engine.results import Results, Boxes

_RANDOM_CHARS = string.ascii_letters + string.digits

class WorkerState(Enum):
    IDLE = "idle"
    INIT = "init"
    RUNNING = "running"
    RETRYING = "retrying"
    STOPPED = "stopped"

class TrackArgs:
    """ByteTrack 追踪参数"""
    def __init__(self, config: dict = None):
        track_config = config.get("tracker", {}) if config else {}
        self.track_high_thresh = track_config.get("track_high_thresh", 0.5)
        self.track_low_thresh = track_config.get("track_low_thresh", 0.1)
        self.new_track_thresh = track_config.get("new_track_thresh", 0.5)
        self.track_buffer = track_config.get("track_buffer", 50)
        self.match_thresh = track_config.get("match_thresh", 0.7)
        self.fuse_score = track_config.get("fuse_score", True)
        self.min_box_area = track_config.get("min_box_area", 10)
        self.mot20 = False

class StreamWorker:
    """RTMP 直播流 Worker — 完整链路：检测+跟踪+绘制+推流+上报+存储"""

    def __init__(self, task_id: str, algorithm_ids: list, stream_url: str,
                 push_url: str, config: dict, orm_helper=None, minio_helper=None,
                 inference_engine=None, platform_id: str = "", minio_name: str = "nbuav-crh", original_task_id: str = ""):
        self.task_id = task_id
        self.original_task_id = original_task_id or task_id.split("_")[0]
        self.algorithm_ids = algorithm_ids
        self.stream_url = stream_url
        self.push_url = push_url
        self.config = config
        self.orm_helper = orm_helper
        self.minio_helper = minio_helper
        self.inference_engine = inference_engine
        self.platform_id = platform_id
        self.minio_name = minio_name

        self.state = WorkerState.IDLE
        self._stop_event = __import__("threading").Event()
        self._retry_count = 0
        self._max_retry_interval = 10
        self._consecutive_success = 0
        self._frame_count = 0
        self._dropped_count = 0
        self._detection_interval = 1
        self._skip_counter = 0
        self._last_detections = {}

        self.video_capture = None
        self.ffmpeg_pusher = None
        self.detector_utils = DetectionUtils(task_id)
        self.geo_utils = GeoUtils(task_id)

        self.logger = get_task_logger(task_id)
        self.perf_logger = get_performance_logger(task_id)
        # ByteTrack 追踪器（每个算法独立）
        self._trackers = {}
        self._track_args = TrackArgs(config)
        # 活跃检测，追踪上报限制
        self._active_tracks = {}
        self._latest_annotated_frames = {}
        # 直播流重连配置
        self._max_reconnect = self.config.get("ffmpeg", {}).get("max_reconnect", 10)
        self._reconnect_count = 0
        # 流健康监控
        self._last_frame_time = time.time()
        self._stream_stall_warned = False
        self._stream_stall_threshold = self.config.get("ffmpeg", {}).get("stream_stall_threshold", 3.0)
        self._stream_reconnect_threshold = self.config.get("ffmpeg", {}).get("stream_reconnect_threshold", 8.0)
        self._stall_watchdog = None
        self._stall_watchdog_stop = threading.Event()
        # 连续检测配置
        self._min_report_frames = self.config.get("tracker", {}).get("min_report_frames", 5)
        # 回调上报队列
        self._report_queue = Queue()
        self._report_worker = threading.Thread(
            target=self._process_report_queue,
            daemon=True,
            name=f"report_worker_{task_id}"
        )
        self._report_worker.start()


    def start(self):
        """启动 Worker"""
        self.state = WorkerState.INIT
        self.logger.info(
            f"Worker 启动 | stream={self.stream_url} | push={self.push_url} | "
            f"algorithms={self.algorithm_ids}"
        )

        while not self._stop_event.is_set():
            try:
                self._run_loop()
                break
            except Exception as e:
                self.logger.error(f"Worker 运行异常: {e}\n{traceback.format_exc()}")
                if not self._stop_event.is_set():
                    self._handle_retry()

        self.state = WorkerState.STOPPED
        self.logger.info("Worker 已停止")

    def _run_loop(self):
        """主运行循环"""
        # 初始化视频捕获
        self.video_capture = VideoCapture(
            url=self.stream_url,
            config=self.config,
            task_id=self.task_id,
        )

        if not self.video_capture.is_alive():
            raise RuntimeError("视频捕获初始化失败")

        # 使用源流实际分辨率
        width = self.video_capture.width
        height = self.video_capture.height
        self.logger.info(f"视频捕获就绪 | 分辨率={width}x{height}")

        # 初始化 FFmpeg 推流
        self.ffmpeg_pusher = FFmpegHelper(
            push_url=self.push_url,
            width=width,
            height=height,
            fps=25,
            task_id=self.task_id,
            hwaccel=self.config.get("ffmpeg", {}).get("hwaccel", "cuda"),
        )

        if not self.ffmpeg_pusher.start():
            raise RuntimeError("FFmpeg 推流初始化失败")

        # 加载模型
        if not self._reload_models_if_needed():
            raise RuntimeError("模型加载失败，任务结束")

        self.state = WorkerState.RUNNING
        self._retry_count = 0
        self._consecutive_success = 0
        self.logger.info("Worker 进入运行状态")

        # 预热帧数（跳过 FFmpeg 解码器初始化期间的脏数据）
        warmup_frames = self.config.get("ffmpeg", {}).get("warmup_frames", 15)
        # 模型检测预热：用空白帧做一次推理，避免首次检测到物体时的 TensorRT 尖峰
        self._run_detection_warmup()
        # 主循环
        while not self._stop_event.is_set():
            t_start = time.time()

            # ★ 启动流中断看门狗：read() 是阻塞调用，源流中断时永不返回
            #   看门狗在超时后杀死 FFmpeg，使 read() 失败并触发现有重连逻辑
            self._start_stall_watchdog()

            # 读取最新帧，跳过积压
            success, frame = self.video_capture.read(skip_old=True)

            # ★ 取消看门狗（read() 已返回）
            self._stop_stall_watchdog()

            if not success:
                if self.video_capture.is_live:
                    self._reconnect_count += 1
                    if self._reconnect_count > self._max_reconnect:
                        self.logger.error(f"超过最大重连次数 {self._max_reconnect}，上报轨迹并结束任务")
                        self._flush_all_tracks()
                        raise RuntimeError("直播流重连失败，超过最大重试次数")

                    self.logger.warning(f"直播流读取失败，尝试重连 | {self._reconnect_count}/{self._max_reconnect}")
                    if not self.video_capture.restart():
                        raise RuntimeError("直播流重连失败")
                    if not self._reload_models_if_needed():
                        raise RuntimeError("模型重新加载失败，任务结束")
                    warmup_frames = self.config.get("ffmpeg", {}).get("warmup_frames", 10)
                    time.sleep(2)
                    continue
                else:
                    # 视频结束/意外断流，上报所有剩余轨迹；避免出现检测物品未成功上报的问题
                    self._flush_all_tracks()
                    break

            # ★ 流健康监控：检测源流是否长时间无帧
            now = time.time()
            idle_time = now - self._last_frame_time
            if idle_time > self._stream_stall_threshold:
                if not self._stream_stall_warned:
                    self.logger.warning(
                        f"源流疑似中断 | 无帧间隔={idle_time:.1f}s | "
                        f"阈值={self._stream_stall_threshold}s | "
                        f"frame_count={self._frame_count}"
                    )
                    self._stream_stall_warned = True
            else:
                self._stream_stall_warned = False
            self._last_frame_time = now

            self._frame_count += 1
            self._skip_counter += 1
            # 重连成功后，重置重连次数
            self._reconnect_count = 0

            # 预热阶段：跳过检测，直接推原始帧，避免脏数据导致yolo检测全屏结果
            if warmup_frames > 0:
                warmup_frames -= 1
                if not self.ffmpeg_pusher.write_frame(frame):
                    self.logger.warning("预热阶段推流写帧失败")
                    raise RuntimeError("推流写帧失败")
                if warmup_frames == 0:
                    for alg_id in self.algorithm_ids:
                        self._trackers[alg_id] = BYTETracker(args=self._track_args, frame_rate=25)
                    self._active_tracks.clear()
                    self._latest_annotated_frames.clear()
                    self._last_detections.clear()
                    self.logger.info(f"预热完成，开始检测 | 跳过帧数={self.config.get('ffmpeg', {}).get('warmup_frames', 15)}, 重置追踪器")
                continue

            # 对每个算法执行检测
            for alg_id in self.algorithm_ids:
                detections, inference_time = self._run_detection(alg_id, frame)

                # 过滤
                filtered = self.detector_utils.filter_detections(
                    detections,
                    img_width=frame.shape[1],
                    img_height=frame.shape[0],
                    conf_threshold=self.config.get("model", {}).get("report_conf", 0.8),
                    max_box_ratio=self.config.get("model", {}).get("max_box_ratio", 0.4),
                )

                # 绘制
                annotated_frame = self._draw_boxes(
                    frame.copy(), detections,
                    conf_threshold=self.config.get("model", {}).get("default_conf", 0.75),
                )

                # 推流
                if not self.ffmpeg_pusher.write_frame(annotated_frame):
                    self.logger.warning("推流写帧失败")
                    raise RuntimeError("推流写帧失败")

                # 更新活跃轨迹数据
                self._update_active_tracks(alg_id, detections, filtered, frame, annotated_frame)
                # 检查丢失的轨迹，执行上报
                current_track_ids = {det.get("track_id") for det in detections if det.get("track_id") is not None}
                self._check_dropped_tracks(alg_id, current_track_ids)

            # 性能日志
            t_total = (time.time() - t_start) * 1000
            if self._frame_count % 100 == 0:
                self.perf_logger.info(
                    f"Frame {self._frame_count} | total={t_total:.1f}ms | "
                    f"fps={self.video_capture.fps:.1f} | dropped={self._dropped_count}"
                )

            self._consecutive_success += 1

    def _run_detection(self, alg_id: str, frame: np.ndarray) -> tuple:
        """执行检测（带自适应间隔 + SAHI 切片 + ByteTrack）"""
        if self._skip_counter >= self._detection_interval:
            sahi_config = self.config.get("sahi", {})
            use_sahi = sahi_config.get("enabled", False)

            if use_sahi:
                result = self.inference_engine.detect_sahi(
                    alg_id, frame,
                    conf=self.config.get("model", {}).get("default_conf", 0.75),
                    imgsz=self.config.get("model", {}).get("imgsz", 640),
                    slice_size=sahi_config.get("slice_size", 640),
                    overlap_ratio=sahi_config.get("overlap_ratio", 0.2),
                    iou_threshold=sahi_config.get("iou_threshold", 0.5),
                )
            else:
                result = self.inference_engine.detect(
                    alg_id, frame,
                    conf=self.config.get("model", {}).get("default_conf", 0.75),
                    imgsz=self.config.get("model", {}).get("imgsz", 640),
                )

            if isinstance(result, tuple) and len(result) == 3:
                detections, inference_time, raw_results = result
            else:
                detections, inference_time, raw_results = [], 0, []

            # 自适应调整间隔（SAHI 慢时允许更大间隔）
            frame_interval = 1000.0 / 25
            if inference_time < frame_interval * 0.8:
                self._detection_interval = 1
            elif inference_time < frame_interval * 1.5:
                self._detection_interval = 2
            elif inference_time < frame_interval * 3:
                self._detection_interval = 4
            elif inference_time < frame_interval * 5:
                self._detection_interval = 8
            else:
                self._detection_interval = 10  # 188ms 检测 → 每10帧检测一次 针对慢检测的容错

            self._skip_counter = 0

            # ByteTrack 追踪
            if use_sahi and detections:
                # SAHI 模式：从检测结果构造 Boxes 对象给 ByteTrack
                detections = self._track_from_detections(alg_id, detections, frame)
            elif raw_results:
                detections = self._track(alg_id, detections, frame, raw_results)

            if detections:
                self._last_detections[alg_id] = detections
            return self._last_detections.get(alg_id, []), inference_time
        else:
            return self._last_detections.get(alg_id, []), 0

    def _track_from_detections(self, alg_id: str, detections: list, frame: np.ndarray) -> list:
        """从检测结果构造 Boxes 对象进行 ByteTrack 追踪"""
        tracker = self._trackers.get(alg_id)
        if tracker is None:
            return detections

        try:
            if not detections:
                empty_boxes = Boxes(torch.zeros((0, 6), dtype=torch.float32), frame.shape[:2]).numpy()
                tracks = tracker.update(empty_boxes, frame.shape[:2])
            else:
                # 构造 [x1, y1, x2, y2, conf, cls] 数组
                det_array = []
                for det in detections:
                    bbox = det["bbox"]
                    det_array.append([
                        bbox[0], bbox[1], bbox[2], bbox[3],
                        det["confidence"], det["class_id"]
                    ])
                det_tensor = torch.tensor(det_array, dtype=torch.float32)
                boxes = Boxes(det_tensor, frame.shape[:2]).numpy()
                tracks = tracker.update(boxes, frame.shape[:2])

            if len(tracks) == 0:
                return detections

            # 用追踪结果替换检测结果
            tracked_detections = []
            for track in tracks:
                x1, y1, x2, y2 = float(track[0]), float(track[1]), float(track[2]), float(track[3])
                track_id = int(track[4])
                conf = float(track[5])
                cls_id = int(track[6])

                # 多 GPU 推理引擎用 _pools，单 GPU 用 _models
                if hasattr(self.inference_engine, '_pools'):
                    pool = self.inference_engine._pools.get(alg_id)
                    classes = pool.classes if pool else []
                else:
                    classes = self.inference_engine._models.get(alg_id, {}).get("classes", [])
                class_name = classes[cls_id] if cls_id < len(classes) else str(cls_id)

                tracked_detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": class_name,
                    "track_id": track_id,
                })
            return tracked_detections
        except Exception as e:
            self.logger.warning(f"SAHI ByteTrack 追踪异常: {e}")
            return detections


    def _reload_models_if_needed(self) -> bool:
        """检查模型是否已加载，未加载则从数据库重新加载"""
        for alg_id in self.algorithm_ids:
            self._trackers[alg_id] = BYTETracker(args=self._track_args, frame_rate=25)
            self.logger.info(f"ByteTrack 追踪器初始化完成 | algorithm_id={alg_id}")

            loaded_models = self.inference_engine.get_loaded_models()
            if alg_id not in loaded_models:
                self.logger.warning(f"模型未加载，尝试重新加载 | algorithm_id={alg_id}")
                if self.orm_helper:
                    model_info = self.orm_helper.get_model_info(alg_id)
                    if model_info:
                        success = self.inference_engine.load_model(
                            alg_id, model_info.model_path, model_info.cls_path
                        )
                        if not success:
                            self.logger.error(f"模型重新加载失败 | algorithm_id={alg_id}")
                            return False
                    else:
                        self.logger.error(f"数据库中未找到模型信息 | algorithm_id={alg_id}")
                        return False
                else:
                    self.logger.error(f"无数据库连接，无法加载模型 | algorithm_id={alg_id}")
                    return False
            else:
                self.logger.debug(f"模型已加载 | algorithm_id={alg_id}")
        return True

    def _start_stall_watchdog(self):
        """启动流中断看门狗：超时后打断阻塞的 read()，触发现有重连逻辑"""
        self._stall_watchdog_stop.clear()
        threshold = self._stream_reconnect_threshold

        def watchdog():
            if self._stall_watchdog_stop.wait(timeout=threshold):
                return  # 正常返回，无需处理
            # 超时：源流中断，关闭管道 + 杀死 FFmpeg 解码进程
            # 不调 restart()——由主循环的统一重连逻辑处理
            self.logger.warning(
                f"源流中断超过 {threshold}s，看门狗正在终止 FFmpeg 解码进程 | "
                f"reconnect_count={self._reconnect_count}/{self._max_reconnect}"
            )
            if self.video_capture and self.video_capture._ffmpeg_handler:
                self.video_capture._ffmpeg_handler.stop()

        self._stall_watchdog = threading.Thread(
            target=watchdog, daemon=True,
            name=f"stall_watchdog_{self.task_id}"
        )
        self._stall_watchdog.start()

    def _stop_stall_watchdog(self):
        """取消看门狗（read() 正常返回时调用）"""
        self._stall_watchdog_stop.set()
        if self._stall_watchdog and self._stall_watchdog.is_alive():
            self._stall_watchdog.join(timeout=1)
        self._stall_watchdog = None

    def _run_detection_warmup(self):
        """用含模拟目标的图片做推理预热，避免首次检测到物体时的 TensorRT 尖峰"""
        sahi_config = self.config.get("sahi", {})
        use_sahi = sahi_config.get("enabled", False)

        for alg_id in self.algorithm_ids:
            loaded_models = self.inference_engine.get_loaded_models()
            if alg_id not in loaded_models:
                continue
            try:
                if use_sahi:
                    self.inference_engine.warmup_detect_with_targets(
                        alg_id,
                        imgsz=self.config.get("model", {}).get("imgsz", 640),
                        slice_size=sahi_config.get("slice_size", 640),
                        overlap_ratio=sahi_config.get("overlap_ratio", 0.1),
                        iou_threshold=sahi_config.get("iou_threshold", 0.5),
                    )
                else:
                    # 非 SAHI 模式：用空白帧做一次普通检测预热
                    warmup_frame = np.zeros((720, 960, 3), dtype=np.uint8)
                    self.inference_engine.detect(
                        alg_id, warmup_frame,
                        conf=self.config.get("model", {}).get("default_conf", 0.75),
                        imgsz=self.config.get("model", {}).get("imgsz", 640),
                    )
                    self.logger.info(f"模型预热完成 | algorithm_id={alg_id}")
            except Exception as e:
                self.logger.warning(f"模型预热跳过: {e}")

    def _draw_boxes(self, frame: np.ndarray, detections: list,
                    conf_threshold: float = 0.75) -> np.ndarray:
        """绘制检测框（含 track_id，支持中文）— PIL统一绘制，修复矩形框丢失bug"""
        if not detections:
            return frame

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img)
        font = self._get_chinese_font(16)

        for det in detections:
            if det["confidence"] < conf_threshold:
                continue
            box = det["bbox"]
            # 跳过包含 NaN 的无效检测框
            if any(math.isnan(v) for v in box):
                continue
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            track_id = det.get("track_id", 0)
            color = self._get_color(track_id)  # PIL中用的是RGB

            # PIL 绘制矩形框（统一在PIL上绘制，避免cv2/PIL混用丢失）
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # PIL 绘制中文文字+背景
            label = f"ID:{track_id} {det['class_name']} {det['confidence']:.2f}"
            text_bbox = draw.textbbox((x1, y1 - 25), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            if y1 - text_h - 5 > 0:
                draw.rectangle(
                    [x1, y1 - text_h - 5, x1 + text_w + 5, y1],
                    fill=color,
                )
                draw.text((x1 + 2, y1 - text_h - 3), label, fill=(255, 255, 255), font=font)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _get_chinese_font(size: int = 16):
        """获取中文字体，按优先级尝试多个路径"""
        font_paths = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]

        for path in font_paths:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue

        # 兜底：默认字体（不支持中文但不会报错）
        return ImageFont.load_default()

    # 选择过程中置信度最高的一次图片作为上传认证的结果
    def _update_active_tracks(self, alg_id: str, detections: list,
                              filtered: list, frame: np.ndarray, annotated_frame: np.ndarray):
        """更新活跃轨迹数据"""
        if alg_id not in self._active_tracks:
            self._active_tracks[alg_id] = {}
        if alg_id not in self._latest_annotated_frames:
            self._latest_annotated_frames[alg_id] = {}

        current_ids = set()

        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue
            current_ids.add(track_id)

            if track_id not in self._active_tracks[alg_id]:
                # 新轨迹，初始化
                self._active_tracks[alg_id][track_id] = {
                    "track_id": track_id,
                    "class_name": det["class_name"],
                    "class_id": det["class_id"],
                    "first_seen_frame": self._frame_count,
                    "last_seen_frame": self._frame_count,
                    "best_confidence": det["confidence"],
                    "best_bbox": det["bbox"],
                    "last_bbox": det["bbox"],
                    "detection_count": 1,
                }
                # 首帧即为最佳帧
                self._latest_annotated_frames[alg_id][track_id] = (frame.copy(), annotated_frame.copy())
            else:
                # 已有轨迹，更新
                track_data = self._active_tracks[alg_id][track_id]
                track_data["last_seen_frame"] = self._frame_count
                track_data["last_bbox"] = det["bbox"]
                track_data["detection_count"] += 1
                if det["confidence"] > track_data["best_confidence"]:
                    track_data["best_confidence"] = det["confidence"]
                    track_data["best_bbox"] = det["bbox"]
                    # 置信度更高时才更新图片
                    self._latest_annotated_frames[alg_id][track_id] = (frame.copy(), annotated_frame.copy())


    def _check_dropped_tracks(self, alg_id: str, current_track_ids: set):
        """检查 ByteTrack 丢弃的轨迹，满足条件才触发上报"""
        if alg_id not in self._active_tracks:
            return

        previous_ids = set(self._active_tracks[alg_id].keys())
        dropped_ids = previous_ids - current_track_ids

        for track_id in dropped_ids:
            track_data = self._active_tracks[alg_id].pop(track_id)
            frames = self._latest_annotated_frames.get(alg_id, {}).pop(track_id, None)

            # 判断是否满足最小上报帧数
            if track_data["detection_count"] < self._min_report_frames:
                self.logger.debug(
                    f"轨迹丢弃，未达上报阈值 | track_id={track_id} | "
                    f"class={track_data['class_name']} | "
                    f"存活帧数={track_data['detection_count']} | "
                    f"最小阈值={self._min_report_frames}"
                )
                continue

            self.logger.info(
                f"轨迹结束，准备上报 | track_id={track_id} | "
                f"class={track_data['class_name']} | "
                f"存活帧数={track_data['detection_count']} | "
                f"首帧={track_data['first_seen_frame']} | "
                f"末帧={track_data['last_seen_frame']}"
            )

            if frames:
                original_frame, annotated_frame = frames
                self._report_completed_track(alg_id, track_data, original_frame, annotated_frame)


    def _report_completed_track(self, alg_id, track_data, original_frame, annotated_frame):
        """放入队列，不阻塞主循环"""
        self._report_queue.put({
            "alg_id": alg_id,
            "track_data": track_data,
            "original_frame": original_frame.copy(),
            "annotated_frame": annotated_frame.copy(),
        })

    def _process_report_queue(self):
        """队列消费：逐个处理上报"""
        while True:
            item = self._report_queue.get()
            try:
                self._do_report(item)
            except Exception as e:
                self.logger.error(f"上报异常: {e}")
            finally:
                self._report_queue.task_done()

    def _do_report(self, item):
        """上报已完成的轨迹"""
        try:
            alg_id = item["alg_id"]
            track_data = item["track_data"]
            original_frame = item["original_frame"]
            annotated_frame = item["annotated_frame"]
            # 获取定位信息
            location = None
            if self.orm_helper:
                location = self.geo_utils.get_location(self.original_task_id, self.orm_helper)

            lng = location.get("longitude") if location else None
            lat = location.get("latitude") if location else None
            height_val = location.get("height") if location else None
            elevation = location.get("elevation") if location else None

            # 保存图片
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"frames/{self.task_id}"
            os.makedirs(save_dir, exist_ok=True)

            track_id = track_data["track_id"]
            original_path = os.path.join(save_dir, f"{timestamp}_track{track_id}_original.jpg")
            annotated_path = os.path.join(save_dir, f"{timestamp}_track{track_id}_annotated.jpg")

            cv2.imwrite(original_path, original_frame)
            cv2.imwrite(annotated_path, annotated_frame)

            # 上传 MinIO
            minio_object = ""
            orig_minio_object = ""
            minio_config_name = self.minio_name

            if self.platform_id and self.orm_helper:
                platform_cb = self.orm_helper.get_platform_callback_url(self.platform_id)
                if platform_cb:
                    minio_config_name = platform_cb.minio_config

            if self.minio_helper:
                minio_object = f"detection/{self.task_id}/{timestamp}_track{track_id}_annotated.jpg"
                orig_minio_object = f"detection/{self.task_id}/{timestamp}_track{track_id}_original.jpg"
                self.minio_helper.upload_file(annotated_path, minio_object, minio_config_name)
                self.minio_helper.upload_file(original_path, orig_minio_object, minio_config_name)

            # 更新图片记录（与 VideoProcessor 对齐：thumb_url=原图，image_url=标注图）
            if self.orm_helper and minio_object:
                try:
                    self.orm_helper.update_image_record(
                        task_out_bid=int(self.original_task_id),
                        thumb_url=orig_minio_object,
                        image_url=minio_object,
                    )
                except Exception as img_e:
                    self.logger.debug(f"更新图片记录跳过: {img_e}")

            # HTTP 回调（与 VideoProcessor 结构体一致）
            if self.platform_id and self.orm_helper:
                platform_cb = self.orm_helper.get_platform_callback_url(self.platform_id)
                callback_url = platform_cb.callback_url if platform_cb else None
                if callback_url:
                    # 构造 detections（与 VideoProcessor._build_report_detections 结构一致）
                    best_bbox = track_data["best_bbox"]
                    detections_for_report = [{
                        "xyxy": best_bbox,
                        "cls": track_data["class_id"],
                        "cls_name": track_data["class_name"],
                        "conf": round(track_data["best_confidence"], 4),
                        "objects": best_bbox,
                        "track_id": track_id,
                        "longitude": lng,
                        "latitude": lat,
                        "height": height_val,
                        "elevation": elevation,
                    }]

                    result_info = {
                        "message_id": "".join(secrets.choice(_RANDOM_CHARS) for _ in range(12)),
                        "task_id": self.original_task_id,
                        "algorithm_id": alg_id,
                        "timestamp": int(time.time() * 1000),
                        "photo_url": minio_object,
                        "thumb_url": orig_minio_object,
                        "detections": detections_for_report,
                    }

                    self.logger.info(f"推送识别结果：{result_info}")

                    threading.Thread(
                        target=self._send_callback_sync,
                        args=(callback_url, result_info),
                        daemon=True,
                    ).start()

        except Exception as e:
            self.logger.error(f"轨迹上报异常: {e}")


    def _flush_all_tracks(self):
        """视频结束时，上报所有剩余轨迹并重置追踪器"""
        for alg_id in list(self._active_tracks.keys()):
            current_ids = set()
            self._check_dropped_tracks(alg_id, current_ids)

        # 重置 ByteTracker，避免跨视频循环 track_id 连续递增
        for alg_id in self.algorithm_ids:
            self._trackers[alg_id] = BYTETracker(args=self._track_args, frame_rate=25)
            self.logger.info(f"视频结束，ByteTrack 追踪器已重置 | algorithm_id={alg_id}")

        # 清空活跃轨迹缓存
        self._active_tracks.clear()
        self._latest_annotated_frames.clear()
        self.logger.info("所有剩余轨迹已上报，追踪器已重置")

    def _send_callback_sync(self, url: str, payload: dict):
        """同步发送 HTTP 回调"""
        try:
            import requests
            resp = requests.post(url, json=payload, timeout=5)
            if resp.status_code != 200:
                self.logger.warning(f"回调失败 | url={url} | status={resp.status_code}")
            else:
                self.logger.debug(f"回调成功 | url={url}")
        except Exception as e:
            self.logger.error(f"回调异常: {e}")

    def _track(self, alg_id: str, detections: list, frame: np.ndarray, raw_results=None) -> list:
        """ByteTrack 追踪，返回带 track_id 的检测结果"""
        tracker = self._trackers.get(alg_id)
        if tracker is None:
            return detections

        try:
            if raw_results and len(raw_results) > 0 and raw_results[0].boxes is not None and len(raw_results[0].boxes) > 0:
                boxes = raw_results[0].boxes.cpu().numpy()
                tracks = tracker.update(boxes, frame.shape[:2])
            else:
                empty_boxes = Boxes(torch.zeros((0, 6), dtype=torch.float32), frame.shape[:2]).numpy()
                tracks = tracker.update(empty_boxes, frame.shape[:2])

        except Exception as e:
            self.logger.warning(f"ByteTrack 更新异常，跳过本帧追踪: {e}")
            return detections

        if len(tracks) == 0:
            return detections

        # tracks 格式：[x1, y1, x2, y2, track_id, score, cls, idx]
        tracked_detections = []
        for track in tracks:
            x1, y1, x2, y2 = float(track[0]), float(track[1]), float(track[2]), float(track[3])
            track_id = int(track[4])
            conf = float(track[5])
            cls_id = int(track[6])

            # 多 GPU 推理引擎用 _pools，单 GPU 用 _models
            if hasattr(self.inference_engine, '_pools'):
                pool = self.inference_engine._pools.get(alg_id)
                classes = pool.classes if pool else []
            else:
                classes = self.inference_engine._models.get(alg_id, {}).get("classes", [])
            class_name = classes[cls_id] if cls_id < len(classes) else str(cls_id)

            tracked_detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls_id,
                "class_name": class_name,
                "track_id": track_id,
            })

        return tracked_detections


    @staticmethod
    def _get_color(track_id: int) -> tuple:
        """根据 track_id 生成固定颜色"""
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255),
            (255, 255, 0), (0, 255, 255), (255, 0, 255),
            (128, 255, 0), (255, 128, 0), (0, 128, 255),
            (128, 0, 255), (255, 128, 128), (128, 255, 128),
        ]
        return colors[track_id % len(colors)]


    def _handle_retry(self):
        """处理重试逻辑"""
        self.state = WorkerState.RETRYING
        self._retry_count += 1

        # 指数退避
        wait_time = min(2 ** (self._retry_count - 1), self._max_retry_interval)
        self.logger.info(f"重试等待 | retry={self._retry_count} | wait={wait_time}s")
        self._stop_event.wait(wait_time)

        # 清理资源
        self._cleanup()

        for alg_id in self.algorithm_ids:
            self._trackers[alg_id] = BYTETracker(args=self._track_args, frame_rate=25)
            self.logger.info(f"ByteTrack 追踪器已重置 | algorithm_id={alg_id}")

        self._consecutive_success = 0

    def _cleanup(self):
        """清理资源"""
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        if self.ffmpeg_pusher:
            self.ffmpeg_pusher.stop()  # stop() 内部会确保杀掉进程
            self.ffmpeg_pusher = None

    def stop(self):
        """停止 Worker"""
        self.logger.info("收到停止信号")
        self._stop_event.set()
        self._cleanup()
        self.state = WorkerState.STOPPED

    def get_restart_func(self):
        """获取重启函数（供清理调度器使用）"""
        def restart():
            self._stop_event.clear()
            self._retry_count = 0
            self._frame_count = 0
            self.state = WorkerState.IDLE
            self.start()
        return restart
