import time
import subprocess
import threading
import numpy as np
from app.utils.logger import get_task_logger


class FFmpegGPUStreamHandler:
    """FFmpeg GPU 解码封装，支持 RTMP/RTSP/HTTP 流"""

    def __init__(self, stream_url: str, width: int = 640, height: int = 640,
                 hwaccel: str = "cuda", task_id: str = "system",
                 probe_size: int = 32, analyzeduration: int = 0):
        self.stream_url = stream_url
        self.width = width
        self.height = height
        self.hwaccel = hwaccel
        self.task_id = task_id
        # MP4视频和直播rtmp 配置调整
        if stream_url.startswith("rtmp"):
            self.probe_size = probe_size
            self.analyzeduration = analyzeduration
        else:
            self.probe_size = 32
            self.analyzeduration = 0
        self.process = None
        self.stderr_thread = None
        self.running = False
        self.frame_size = width * height * 3
        self.logger = get_task_logger(task_id)
        self._gpu_failed = False

    def start(self) -> bool:
        """启动解码进程，GPU 失败自动回退 CPU"""
        if not self._gpu_failed:
            success = self._start_with_hwaccel(self.hwaccel)
            if success:
                return True
            self.logger.warning("GPU 解码启动失败，回退到 CPU 软解码")
            self._gpu_failed = True

        return self._start_with_hwaccel(None)

    def _start_with_hwaccel(self, hwaccel: str) -> bool:
        """使用指定硬件加速启动解码"""
        command = ["ffmpeg"]
        if hwaccel:
            command.extend(["-hwaccel", hwaccel])
        command.extend([
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-probesize", str(self.probe_size),
            "-analyzeduration", str(self.analyzeduration),
        ])
        if self.stream_url.startswith("rtmp"):
            command.extend(["-rtmp_live", "live"])
        command.extend([
            "-i", self.stream_url,
            "-vf", f"scale={self.width}:{self.height}",
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "-an",
            "-sn",
            "pipe:1",
        ])

        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10 ** 8,
            )
            self.running = True

            self.stderr_thread = threading.Thread(
                target=self._consume_stderr,
                daemon=True,
                name=f"ffmpeg_decode_stderr_{self.task_id}"
            )
            self.stderr_thread.start()

            self.logger.info(
                f"FFmpeg 解码进程启动成功 | pid={self.process.pid} | "
                f"hwaccel={'cpu' if not hwaccel else hwaccel} | url={self.stream_url}"
            )
            return True

        except Exception as e:
            self.logger.error(f"FFmpeg 解码进程启动失败: {e}")
            return False

    def read_frame(self, skip_old: bool = True) -> tuple:
        """读取一帧，skip_old=True 时跳过积压帧，只返回最新一帧"""
        if not self.running or self.process is None:
            return False, None

        if self.process.poll() is not None:
            self.logger.warning("FFmpeg 解码进程已退出")
            self.running = False
            return False, None

        try:
            if skip_old:
                # 读取所有可用帧，只保留最后一帧
                last_frame = None
                while True:
                    # 检查缓冲区是否有完整一帧
                    if not self._has_data_ready():
                        if last_frame is None:
                            # 缓冲区无数据，阻塞读取一帧
                            raw = self.process.stdout.read(self.frame_size)
                            if len(raw) < self.frame_size:
                                self.logger.warning(f"读取帧数据不完整 | expected={self.frame_size} | got={len(raw)}")
                                return False, None
                            last_frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
                        break

                    # 缓冲区有数据，读取并覆盖
                    raw = self.process.stdout.read(self.frame_size)
                    if len(raw) < self.frame_size:
                        if last_frame is not None:
                            break
                        self.logger.warning(f"读取帧数据不完整 | expected={self.frame_size} | got={len(raw)}")
                        return False, None
                    last_frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))

                if last_frame is not None:
                    return True, last_frame
                return False, None
            else:
                # 原始逻辑：顺序读取
                raw = self.process.stdout.read(self.frame_size)
                if len(raw) < self.frame_size:
                    self.logger.warning(f"读取帧数据不完整 | expected={self.frame_size} | got={len(raw)}")
                    return False, None
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
                return True, frame

        except Exception as e:
            self.logger.error(f"读取帧异常: {e}")
            return False, None

    def _has_data_ready(self) -> bool:
        """检查 stdout 缓冲区是否有数据可读"""
        try:
            import select
            if self.process and self.process.stdout:
                readable, _, _ = select.select([self.process.stdout], [], [], 0)
                return bool(readable)
        except Exception:
            pass
        return False

    def _consume_stderr(self):
        """消费 stderr 输出"""
        try:
            while self.running and self.process and self.process.poll() is None:
                line = self.process.stderr.readline()
                if line:
                    decoded = line.decode("utf-8", errors="replace").strip()
                    if "error" in decoded.lower():
                        self.logger.error(f"FFmpeg 解码: {decoded}")
                    # elif "warning" in decoded.lower():
                    #     self.logger.warning(f"FFmpeg 解码: {decoded}")
        except Exception:
            pass

    def stop(self):
        """停止解码进程"""
        self.running = False
        if self.process:
            try:
                self.process.stdout.close()
                self.process.terminate()
                self.process.wait(timeout=5)
                self.logger.info(f"FFmpeg 解码进程已停止 | pid={self.process.pid}")
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                self.logger.warning(f"FFmpeg 解码进程被强制杀死 | pid={self.process.pid}")
            except Exception as e:
                self.logger.error(f"FFmpeg 解码进程停止异常: {e}")
            finally:
                self.process = None

    def restart(self) -> bool:
        """重启解码进程"""
        self.stop()
        self._gpu_failed = False
        time.sleep(0.5)
        return self.start()

    def is_alive(self) -> bool:
        """检查进程是否存活"""
        return self.process is not None and self.process.poll() is None and self.running

    @staticmethod
    def probe_resolution(stream_url: str) -> tuple:
        """探测源流分辨率，返回 (width, height)"""
        import subprocess
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                stream_url,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(",")
                if len(parts) == 2:
                    return int(parts[0]), int(parts[1])
        except Exception:
            pass
        return 640, 640

