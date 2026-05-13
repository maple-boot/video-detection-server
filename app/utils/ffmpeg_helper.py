import subprocess
import threading
import time
import numpy as np
import os
from app.utils.logger import get_task_logger


class FFmpegHelper:
    """FFmpeg 推流 — 阻塞写入 + 最新帧覆盖"""

    def __init__(self, push_url: str, width: int, height: int, fps: int,
                 task_id: str = "system", algorithm_id: str = "",
                 hwaccel: str = "cuda", preset: str = "ultrafast",
                 tune: str = "zerolatency"):
        self.push_url = push_url
        self.width = width
        self.height = height
        self.fps = fps
        self.task_id = task_id
        self.algorithm_id = algorithm_id
        self.hwaccel = hwaccel
        self.preset = preset
        self.tune = tune
        self.process = None
        self.stderr_thread = None
        self.running = False
        self.logger = get_task_logger(task_id, algorithm_id)

        self._latest_frame = None
        self._current_push_frame = None
        self._frame_lock = threading.Lock()
        self._frame_event = threading.Event()
        self._writer_thread = None
        self._frames_written = 0
        self._frames_dropped = 0

    def start(self) -> bool:
        command = [
            "ffmpeg",
            "-y",
            "-fflags", "nobuffer+discardcorrupt",
            "-flags", "low_delay",
            "-probesize", "32",
            "-analyzeduration", "0",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", self.preset,
            "-tune", self.tune,
            "-pix_fmt", "yuv420p",
            "-g", str(self.fps),
            "-bf", "0",
            "-x264-params", "rc-lookahead=0:sliced-threads=1:sync-lookahead=0",
            "-flush_packets", "1",
            "-max_muxing_queue_size", "2",
            "-max_delay", "0",
            "-f", "flv",
            self.push_url,
        ]

        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self.running = True

            self.stderr_thread = threading.Thread(
                target=self._consume_stderr, daemon=True,
                name=f"ffmpeg_stderr_{self.task_id}"
            )
            self.stderr_thread.start()

            self._writer_thread = threading.Thread(
                target=self._write_loop, daemon=True,
                name=f"ffmpeg_writer_{self.task_id}"
            )
            self._writer_thread.start()

            self.logger.info(
                f"FFmpeg 推流启动 | pid={self.process.pid} | "
                f"push_url={self.push_url} | 分辨率={self.width}x{self.height}"
            )
            return True

        except Exception as e:
            self.logger.error(f"FFmpeg 推流启动失败: {e}")
            return False

    def write_frame(self, frame: np.ndarray) -> bool:
        """主循环调用：更新最新帧，完全非阻塞"""
        if not self.running:
            return False
        with self._frame_lock:
            self._latest_frame = frame
        self._frame_event.set()
        return True

    def _write_loop(self):
        """
        核心逻辑：
        1. 固定间隔(40ms)醒来
        2. 拿最新帧
        3. 阻塞写入（如果 FFmpeg 慢，这里会等）
        4. 写完后回到步骤1，此时 _latest_frame 已被主循环更新为更新的帧
        5. 中间帧自动跳过
        """
        frame_interval = 1.0 / self.fps

        self.logger.info(f"写帧线程启动 | target_fps={self.fps}")

        while self.running and self.process and self.process.poll() is None:
            # 等待新帧或超时
            self._frame_event.wait(timeout=frame_interval)
            self._frame_event.clear()

            if self.process.poll() is not None:
                self.logger.warning("FFmpeg 进程已退出")
                self.running = False
                break

            # 取最新帧（不加锁取引用，Python GIL 保证原子性）
            frame = self._latest_frame
            if frame is None:
                continue

            self._current_push_frame = frame

            # 阻塞写入
            try:
                self.process.stdin.write(frame.tobytes())
                self.process.stdin.flush()
                self._frames_written += 1
            except BrokenPipeError:
                self.logger.error("FFmpeg 管道断裂")
                self.running = False
                break
            except Exception as e:
                self.logger.error(f"FFmpeg 写帧异常: {e}")
                self.running = False
                break

            if self._frames_written % 250 == 0:
                self.logger.info(
                    f"推流统计 | written={self._frames_written} | dropped={self._frames_dropped}"
                )

        self.logger.info(
            f"写帧线程退出 | written={self._frames_written} | dropped={self._frames_dropped} "
        )

    def _consume_stderr(self):
        try:
            while self.running and self.process and self.process.poll() is None:
                line = self.process.stderr.readline()
                if line:
                    decoded = line.decode("utf-8", errors="replace").strip()
                    if "error" in decoded.lower() and "deprecated" not in decoded.lower():
                        self.logger.error(f"FFmpeg: {decoded}")
        except Exception:
            pass

    def stop(self):
        self.running = False
        self._frame_event.set()

        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=3)

        if self.process:
            try:
                self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=5)
                self.logger.info(
                    f"FFmpeg 推流停止 | written={self._frames_written}"
                )
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            except Exception as e:
                self.logger.error(f"FFmpeg 停止异常: {e}")
            finally:
                self.process = None
