"""
RTMP 直播流录制工具 — 将 RTMP 流保存为本地 MP4 文件
文件名格式：{task_id}_{algorithm_id}.mp4

用法（代码调用）:
    from app.utils.stream_recorder import StreamRecorder
    recorder = StreamRecorder(stream_url, task_id, algorithm_id)
    recorder.start_async()  # 后台线程录制
    recorder.stop()         # 停止录制
"""

import os
import time
import subprocess
import threading
from datetime import datetime

from app.utils.logger import get_task_logger


class StreamRecorder:
    """RTMP 直播流录制器 — 将实时流保存为本地 MP4 文件"""

    def __init__(self, stream_url: str, mp4_file_name: str,
                 output_dir: str = "recordings", hwaccel: str = "cuda",
                 segment_seconds: int = 0):
        """
        Args:
            stream_url:      RTMP/RTSP/HTTP 视频流地址
            mp4_file_name:   MP4 文件名
            output_dir:      输出目录（默认 recordings/）
            hwaccel:         硬件加速（cuda / qsv / None）
            segment_seconds: 分段录制秒数（0=不分段，整段录制）
        """
        self.stream_url = stream_url
        self.output_dir = output_dir
        self.hwaccel = hwaccel
        self.segment_seconds = segment_seconds

        self.filename = mp4_file_name
        self.output_path = os.path.join(output_dir, self.filename)
        self.logger = get_task_logger(self.filename.split('_')[0])

        self._process = None
        self._thread = None
        self._stop_event = threading.Event()
        self._recording = False
        self._start_time = None

    def start(self) -> bool:
        """启动录制（阻塞，直到流结束或调用 stop()）"""
        return self._record()

    def start_async(self) -> bool:
        """启动录制（后台线程，非阻塞）"""
        if self._recording:
            self.logger.warning("录制已在进行中")
            return False
        self._thread = threading.Thread(
            target=self._record,
            daemon=True,
            name=f"recorder_{self.filename.split('.')[0]}",
        )
        self._thread.start()
        return True

    def stop(self):
        """停止录制"""
        self.logger.info("收到停止录制信号")
        self._stop_event.set()
        if self._process and self._process.poll() is None:
            try:
                self._process.stdin.write(b"q\n")
                self._process.stdin.flush()
            except Exception:
                pass
            # 等待 FFmpeg 退出
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.logger.warning("FFmpeg 未响应，强制终止")
                self._process.kill()
                self._process.wait()
        self._recording = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def duration_seconds(self) -> float:
        if self._start_time:
            return time.time() - self._start_time
        return 0

    def _record(self) -> bool:
        """执行录制的核心方法"""
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        # 如果文件已存在，加上时间戳避免覆盖
        if os.path.exists(self.output_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.filename = f"{self.filename.split('.')[0]}_{timestamp}.mp4"
            self.output_path = os.path.join(self.output_dir, self.filename)
            self.logger.info(f"文件已存在，使用新文件名: {self.filename}")

        self.logger.info(
            f"开始录制 | stream={self.stream_url} | "
            f"output={self.output_path} | hwaccel={self.hwaccel}"
        )

        command = self._build_command()
        self.logger.info(f"FFmpeg 命令: {' '.join(command)}")

        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._recording = True
            self._start_time = time.time()

            # 监控 FFmpeg stderr 输出（日志）
            stderr_thread = threading.Thread(
                target=self._monitor_stderr,
                daemon=True,
            )
            stderr_thread.start()

            # 等待进程结束或停止信号
            while not self._stop_event.is_set():
                retcode = self._process.poll()
                if retcode is not None:
                    if retcode == 0:
                        self.logger.info(f"录制正常结束 | output={self.output_path}")
                    else:
                        self.logger.error(f"FFmpeg 异常退出 | retcode={retcode}")
                    break
                time.sleep(0.5)

            # 如果是 stop() 触发的，等一下让 FFmpeg 优雅关闭
            if self._stop_event.is_set() and self._process.poll() is None:
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()

            self._recording = False
            elapsed = time.time() - self._start_time if self._start_time else 0

            # 检查输出文件
            if os.path.exists(self.output_path):
                size_mb = os.path.getsize(self.output_path) / (1024 * 1024)
                self.logger.info(
                    f"录制完成 | duration={elapsed:.1f}s | "
                    f"size={size_mb:.1f}MB | file={self.output_path}"
                )
                return True
            else:
                self.logger.error("录制结束但输出文件不存在")
                return False

        except FileNotFoundError:
            self.logger.error("FFmpeg 未安装或不在 PATH 中")
            self._recording = False
            return False
        except Exception as e:
            self.logger.error(f"录制异常: {e}")
            self._recording = False
            return False

    def _build_command(self) -> list:
        """构建 FFmpeg 录制命令"""
        command = ["ffmpeg", "-y"]

        # 硬件加速解码
        if self.hwaccel:
            command.extend(["-hwaccel", self.hwaccel])

        # 输入参数
        command.extend([
            "-fflags", "nobuffer+discardcorrupt+genpts",
            "-flags", "low_delay",
            "-thread_queue_size", "64",
            "-probesize", "5000000",
            "-analyzeduration", "5000000",
        ])

        # RTMP 特殊参数
        if self.stream_url.startswith("rtmp"):
            command.extend(["-rtmp_live", "live"])

        # 输入源
        command.extend(["-i", self.stream_url])

        # 分段录制
        if self.segment_seconds > 0:
            segment_path = os.path.join(
                self.output_dir,
                f"{self.task_id}_{self.algorithm_id}_%04d.mp4",
            )
            command.extend([
                "-c", "copy",
                "-f", "segment",
                "-segment_time", str(self.segment_seconds),
                "-reset_timestamps", "1",
                "-segment_format", "mp4",
                segment_path,
            ])
        else:
            # 单文件录制
            command.extend([
                "-c", "copy",
                "-movflags", "+faststart",
                self.output_path,
            ])

        return command

    def _monitor_stderr(self):
        """监控 FFmpeg stderr，打印关键日志"""
        if not self._process or not self._process.stderr:
            return
        try:
            for line in iter(self._process.stderr.readline, b""):
                if self._stop_event.is_set():
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                # 只记录关键信息，避免刷屏
                if "error" in text.lower() or "warning" in text.lower():
                    self.logger.warning(f"FFmpeg: {text}")
                elif "frame=" in text or "size=" in text:
                    # 进度信息，定期打印
                    self.logger.debug(f"FFmpeg: {text}")
        except Exception:
            pass