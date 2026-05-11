import subprocess
import threading
import time
import numpy as np
from app.utils.logger import get_task_logger


class FFmpegHelper:
    """FFmpeg 推流进程管理"""

    def __init__(self, push_url: str, width: int, height: int, fps: int,
                 task_id: str = "system", algorithm_id: str = "",
                 hwaccel: str = "cuda", preset: str = "ultrafast",
                 tune: str = "zerolatency", write_retry: int = 2,
                 max_buffer_frames: int = 5):
        self.push_url = push_url
        self.width = width
        self.height = height
        self.fps = fps
        self.task_id = task_id
        self.algorithm_id = algorithm_id
        self.hwaccel = hwaccel
        self.preset = preset
        self.tune = tune
        self.write_retry = write_retry
        self.max_buffer_frames = max_buffer_frames  # 最大缓冲帧数
        self.process = None
        self.stderr_thread = None
        self.running = False
        self.logger = get_task_logger(task_id, algorithm_id)
        self._frames_written = 0
        self._frames_dropped = 0
        self._last_write_time = 0

    def start(self) -> bool:
        """启动 FFmpeg 推流进程"""
        command = [
            "ffmpeg",
            "-y",
            # 输入端低延迟 fflags / flags
            "-fflags", "nobuffer+discardcorrupt",
            "-flags", "low_delay",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", self.preset,
            "-tune", self.tune,
            "-pix_fmt", "yuv420p",
            "-bf", "0",
            "-g", str(self.fps),   # 25 FPS ~ 1s
            # 输出端低延迟  -x264-params
            "-x264-params", "rc-lookahead=0:sliced-threads=1:sync-lookahead=0",   # 禁用前瞻 / 切片线程 / 禁用同步前瞻
            # muxer 低延迟配置
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
                bufsize=0,  # 无缓冲，减少积压
            )
            self.running = True

            self.stderr_thread = threading.Thread(
                target=self._consume_stderr,
                daemon=True,
                name=f"ffmpeg_stderr_{self.task_id}"
            )
            self.stderr_thread.start()

            self.logger.info(f"FFmpeg 推流进程启动成功 | pid={self.process.pid} | push_url={self.push_url}")
            return True

        except Exception as e:
            self.logger.error(f"FFmpeg 推流进程启动失败: {e}")
            return False

    def write_frame(self, frame: np.ndarray) -> bool:
        """写入一帧，带丢帧策略"""
        if not self.running or self.process is None:
            return False

        current_time = time.time()

        # 计算预期帧间隔
        frame_interval = 1.0 / self.fps

        # 如果距离上次写帧时间过短，说明积压了
        # 计算应该丢弃的帧数
        if self._last_write_time > 0:
            elapsed = current_time - self._last_write_time
            expected_frames = elapsed / frame_interval
            if expected_frames < 0.5:
                # 距离上次写帧太近，说明处理太快，推流积压
                # 检查积压程度
                backlog = self._estimate_backlog()
                if backlog > self.max_buffer_frames:
                    self._frames_dropped += 1
                    if self._frames_dropped % 50 == 1:
                        self.logger.warning(
                            f"推流积压，丢帧 | backlog={backlog} | "
                            f"dropped={self._frames_dropped} | written={self._frames_written}"
                        )
                    return True  # 返回 True 表示"处理成功"（实际是丢弃）

        for attempt in range(self.write_retry + 1):
            try:
                self.process.stdin.write(frame.tobytes())
                self.process.stdin.flush()
                self._frames_written += 1
                self._last_write_time = current_time
                return True
            except BrokenPipeError:
                self.logger.warning(f"FFmpeg 管道断裂 | attempt={attempt + 1}")
                if attempt < self.write_retry:
                    time.sleep(0.01)
                else:
                    self.logger.error("FFmpeg 写帧失败，已达最大重试次数")
                    self.running = False
                    return False
            except Exception as e:
                self.logger.error(f"FFmpeg 写帧异常: {e}")
                self.running = False
                return False

        return False

    def _estimate_backlog(self) -> int:
        """估算推流积压帧数"""
        if self._last_write_time == 0:
            return 0
        elapsed = time.time() - self._last_write_time
        frame_interval = 1.0 / self.fps
        # 如果超过2个帧间隔没写，说明没有积压
        if elapsed > frame_interval * 2:
            return 0
        # 通过写入速度估算
        return max(0, int(elapsed / frame_interval))

    def _consume_stderr(self):
        """消费 stderr 输出"""
        try:
            while self.running and self.process and self.process.poll() is None:
                line = self.process.stderr.readline()
                if line:
                    decoded = line.decode("utf-8", errors="replace").strip()
                    if "error" in decoded.lower():
                        self.logger.error(f"FFmpeg: {decoded}")
                    elif "warning" in decoded.lower():
                        self.logger.warning(f"FFmpeg: {decoded}")
        except Exception:
            pass

    def stop(self):
        """停止推流进程"""
        self.running = False
        if self.process:
            try:
                self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=5)
                self.logger.info(
                    f"FFmpeg 推流进程已停止 | pid={self.process.pid} | "
                    f"written={self._frames_written} | dropped={self._frames_dropped}"
                )
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                self.logger.warning(f"FFmpeg 推流进程被强制杀死 | pid={self.process.pid}")
            except Exception as e:
                self.logger.error(f"FFmpeg 推流进程停止异常: {e}")
            finally:
                self.process = None
