import time
import cv2
from enum import Enum
from app.utils.ffmpeg_stream_handler import FFmpegGPUStreamHandler
from app.utils.logger import get_task_logger


class SourceType(Enum):
    LIVE = "live"
    FILE = "file"


class VideoCapture:
    """统一视频捕获接口，输出源流原始分辨率"""

    def __init__(self, url: str, config: dict, task_id: str = "system"):
        self.url = url
        self.config = config
        self.task_id = task_id
        self.source_type = self._detect_source_type(url)
        self.logger = get_task_logger(task_id)
        self._capture = None
        self._ffmpeg_handler = None
        self._frame_count = 0
        self._start_time = 0
        self._width = 0
        self._height = 0

        if self.source_type == SourceType.LIVE:
            self._init_ffmpeg_live()
        else:
            self._init_opencv_file()

    def _detect_source_type(self, url: str) -> SourceType:
        """检测输入源类型"""
        live_prefixes = ("rtmp://", "rtsp://", "http://", "https://")
        if any(url.startswith(p) for p in live_prefixes):
            return SourceType.LIVE
        return SourceType.FILE

    def _init_ffmpeg_live(self):
        """初始化 FFmpeg 直播流解码，输出原始分辨率"""
        ffmpeg_config = self.config.get("ffmpeg", {})

        # 探测源流分辨率
        self._width, self._height = FFmpegGPUStreamHandler.probe_resolution(self.url)
        self.logger.info(f"探测到源流分辨率 | {self._width}x{self._height} | url={self.url}")

        self._ffmpeg_handler = FFmpegGPUStreamHandler(
            stream_url=self.url,
            width=self._width,
            height=self._height,
            hwaccel=ffmpeg_config.get("hwaccel", "cuda"),
            task_id=self.task_id,
            probe_size=ffmpeg_config.get("probe_size", 32),
            analyzeduration=ffmpeg_config.get("analyzeduration", 0),
        )
        success = self._ffmpeg_handler.start()
        if success:
            self.logger.info(f"直播流解码初始化成功 | {self._width}x{self._height} | url={self.url}")
        else:
            self.logger.error(f"直播流解码初始化失败 | url={self.url}")
        self._start_time = time.time()

    def _init_opencv_file(self):
        """初始化 OpenCV 文件解码"""
        self._capture = cv2.VideoCapture(self.url)
        if self._capture.isOpened():
            self._width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.logger.info(f"文件解码初始化成功 | {self._width}x{self._height} | url={self.url}")
        else:
            self.logger.error(f"文件解码初始化失败 | url={self.url}")
        self._start_time = time.time()

    def read(self, skip_old: bool = True) -> tuple:
        """读取一帧，返回源流原始分辨率"""
        if self.source_type == SourceType.LIVE:
            success, frame = self._ffmpeg_handler.read_frame(skip_old=skip_old)
        else:
            success, frame = self._capture.read()

        if success:
            self._frame_count += 1

        return success, frame

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return 0.0
        return self._frame_count / elapsed

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_live(self) -> bool:
        return self.source_type == SourceType.LIVE

    def is_alive(self) -> bool:
        if self.source_type == SourceType.LIVE:
            return self._ffmpeg_handler.is_alive()
        return self._capture is not None and self._capture.isOpened()

    def restart(self) -> bool:
        if self.source_type == SourceType.LIVE:
            self.logger.info("重启直播流解码...")
            return self._ffmpeg_handler.restart()
        return False

    def release(self):
        if self._ffmpeg_handler:
            self._ffmpeg_handler.stop()
        if self._capture:
            self._capture.release()
        self.logger.info(f"视频捕获资源已释放 | frames={self._frame_count}")
