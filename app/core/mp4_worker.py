import os
import time
import cv2
import numpy as np
from enum import Enum
from app.core.video_capture import VideoCapture
from app.utils.detection_util import DetectionUtils
from app.utils.logger import get_task_logger, get_performance_logger


class MP4WorkerState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"


class MP4Worker:
    """MP4 文件 Worker — 精简链路：检测+跟踪+绘制+输出（不上报、不存储）"""

    def __init__(self, task_id: str, algorithm_ids: list, file_path: str,
                 output_path: str, config: dict, inference_engine=None):
        self.task_id = task_id
        self.algorithm_ids = algorithm_ids
        self.file_path = file_path
        self.output_path = output_path
        self.config = config
        self.inference_engine = inference_engine

        self.state = MP4WorkerState.IDLE
        self._stop_event = __import__("threading").Event()
        self._frame_count = 0

        self.video_capture = None
        self.video_writer = None
        self.detector_utils = DetectionUtils(task_id)

        self.logger = get_task_logger(task_id)
        self.perf_logger = get_performance_logger(task_id)

    def start(self):
        """启动 MP4 处理"""
        self.state = MP4WorkerState.IDLE
        self.logger.info(f"MP4 Worker 启动 | file={self.file_path} | output={self.output_path}")

        try:
            self._run()
        except Exception as e:
            self.logger.error(f"MP4 Worker 异常: {e}")
        finally:
            self._cleanup()
            self.state = MP4WorkerState.STOPPED
            self.logger.info(f"MP4 Worker 完成 | total_frames={self._frame_count}")

    def _run(self):
        """主处理流程"""
        # 初始化视频捕获
        self.video_capture = VideoCapture(
            url=self.file_path,
            config=self.config,
            task_id=self.task_id,
        )

        if not self.video_capture.is_alive():
            raise RuntimeError("MP4 文件打开失败")

        # 获取视频属性
        cap = cv2.VideoCapture(self.file_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # 初始化视频写入器
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        if not self.video_writer.isOpened():
            raise RuntimeError("视频写入器初始化失败")

        # 加载模型
        for alg_id in self.algorithm_ids:
            model_info = self.config.get("_model_info", {}).get(alg_id)
            if model_info:
                self.inference_engine.load_model(alg_id, model_info["model_path"], model_info.get("classes_path", ""))

        self.state = MP4WorkerState.RUNNING

        # 逐帧处理
        while not self._stop_event.is_set():
            t_start = time.time()

            success, frame = self.video_capture.read()
            if not success:
                break

            self._frame_count += 1

            # 检测（每帧都跑）
            annotated_frame = frame.copy()
            for alg_id in self.algorithm_ids:
                detections, inference_time = self.inference_engine.detect(
                    alg_id, frame,
                    conf=self.config.get("model", {}).get("default_conf", 0.75),
                )

                # 绘制（不做上报过滤，显示所有结果）
                annotated_frame = self._draw_boxes(annotated_frame, detections)

            # 写入输出视频
            self.video_writer.write(annotated_frame)

            # 性能日志
            t_total = (time.time() - t_start) * 1000
            if self._frame_count % 100 == 0:
                self.perf_logger.info(
                    f"MP4 Frame {self._frame_count} | total={t_total:.1f}ms | "
                    f"fps={1000 / t_total if t_total > 0 else 0:.1f}"
                )

        self.state = MP4WorkerState.COMPLETED

    def _draw_boxes(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """绘制检测框"""
        for det in detections:
            box = det["bbox"]
            # 跳过包含 NaN 的无效检测框
            if not DetectionUtils.is_valid_box(box):
                continue
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def _cleanup(self):
        """清理资源"""
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

    def stop(self):
        """停止 Worker"""
        self.logger.info("MP4 Worker 收到停止信号")
        self._stop_event.set()
        self._cleanup()
        self.state = MP4WorkerState.STOPPED
