import subprocess
import threading
import time
import numpy as np
import signal
from app.utils.logger import get_task_logger


class FFmpegHelper:
    """FFmpeg 推流 — h264_nvenc + 最新帧覆盖"""

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
            "-c:v", "h264_nvenc",
            "-preset", "fast",
            "-tune", "ll",
            "-rc", "cbr",
            "-b:v", "4M",
            "-maxrate", "4M",
            "-bufsize", "2M",
            "-g", str(self.fps),
            "-bf", "0",
            "-pix_fmt", "yuv420p",
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
                bufsize=0,
            )

            # 等 0.5 秒检查 FFmpeg 是否正常运行
            time.sleep(0.5)
            if self.process.poll() is not None:
                stderr_output = ""
                try:
                    stderr_output = self.process.stderr.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                self.logger.error(
                    f"FFmpeg 启动后立即退出 | returncode={self.process.returncode} | stderr={stderr_output}"
                )
                return False

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
                f"push_url={self.push_url} | 分辨率={self.width}x{self.height} | "
                f"编码=h264_nvenc"
            )
            return True

        except Exception as e:
            self.logger.error(f"FFmpeg 推流启动失败: {e}")
            return False

    def write_frame(self, frame: np.ndarray) -> bool:
        if not self.running:
            return False
        if self.process and self.process.poll() is not None:
            self.logger.warning(f"write_frame: FFmpeg 进程已退出")
            self.running = False
            return False
        with self._frame_lock:
            self._latest_frame = frame
        self._frame_event.set()
        return True

    def _write_loop(self):
        self.logger.info(f"写帧线程启动 | target_fps={self.fps}")
        frame_interval = 1.0 / self.fps  # 40ms
        while self.running and self.process and self.process.poll() is None:
            t_start = time.time()
            self._frame_event.wait(timeout=frame_interval)
            self._frame_event.clear()
            if self.process.poll() is not None:
                self.logger.warning(f"FFmpeg 进程已退出 | returncode={self.process.returncode}")
                self.running = False
                break
            with self._frame_lock:
                frame = self._latest_frame
                self._latest_frame = None
            if frame is None:
                continue
            self._current_push_frame = frame
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
            # ★ 帧率控制：确保写帧间隔 >= 40ms
            elapsed = time.time() - t_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            if self._frames_written % 50 == 0:
                self.logger.info(
                    f"推流统计 | written={self._frames_written} | dropped={self._frames_dropped}"
                )

        self.logger.info(
            f"写帧线程退出 | written={self._frames_written} | dropped={self._frames_dropped}"
        )


    def _consume_stderr(self):
        """消费 FFmpeg stderr — 打印所有输出用于调试"""
        try:
            while self.running and self.process and self.process.poll() is None:
                line = self.process.stderr.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").strip()
                if not decoded:
                    continue

                decoded_lower = decoded.lower()

                # 过滤无意义的进度信息
                if "frame=" in decoded_lower or "fps=" in decoded_lower:
                    continue

                # 根据内容分级日志
                if "error" in decoded_lower and "deprecated" not in decoded_lower:
                    self.logger.error(f"FFmpeg: {decoded}")
                elif "warning" in decoded_lower:
                    self.logger.warning(f"FFmpeg: {decoded}")
                else:
                    self.logger.debug(f"FFmpeg: {decoded}")

        except Exception as e:
            self.logger.error(f"FFmpeg stderr 消费异常: {e}")

    def stop(self):
        self.running = False
        self._frame_event.set()
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=3)
        if self.process:
            try:
                #先关 stdin，再 terminate，最后 kill
                try:
                    self.process.stdin.close()
                except Exception:
                    pass
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
                self.logger.info(f"FFmpeg 推流停止 | pid={self.process.pid}")
            except Exception as e:
                self.logger.error(f"FFmpeg 停止异常: {e}")
            finally:
                self.process = None
        # 确保进程被杀掉
        self._kill_ffmpeg_processes()

    def _kill_ffmpeg_processes(self):
        """杀掉属于本任务的残留 FFmpeg 进程"""
        try:
            result = subprocess.run(
                ["pgrep", "-f", f"ffmpeg.*{self.push_url}"],
                capture_output=True, text=True
            )
            for pid in result.stdout.strip().split('\n'):
                if pid:
                    os.kill(int(pid), signal.SIGKILL)
                    self.logger.warning(f"杀掉残留 FFmpeg 进程 | pid={pid}")
        except Exception:
            pass
