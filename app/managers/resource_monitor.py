import os
import time
import threading
import psutil
from app.utils.logger import get_system_logger

logger = get_system_logger()


class ResourceMonitor:
    """资源监控守护线程，阈值触发清理"""

    def __init__(self, config: dict, cleanup_callback=None):
        monitor_config = config.get("resource_monitor", {})
        self.enabled = monitor_config.get("enabled", True)
        self.check_interval = monitor_config.get("check_interval", 60)
        thresholds = monitor_config.get("thresholds", {})
        self.max_threads = thresholds.get("max_threads", 200)
        self.max_memory_percent = thresholds.get("max_memory_percent", 85)
        self.max_gpu_memory_percent = thresholds.get("max_gpu_memory_percent", 90)
        self.max_ffmpeg_count = thresholds.get("max_ffmpeg_count", 100)
        self.action = monitor_config.get("action", "cleanup")

        self.cleanup_callback = cleanup_callback
        self._running = False
        self._thread = None
        logger.info(
            f"资源监控初始化 | enabled={self.enabled} | interval={self.check_interval}s | "
            f"max_threads={self.max_threads} | max_memory={self.max_memory_percent}%"
        )

    def start(self):
        """启动监控线程"""
        if not self.enabled:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="resource_monitor"
        )
        self._thread.start()
        logger.info("资源监控线程已启动")

    def stop(self):
        """停止监控线程"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("资源监控线程已停止")

    def _monitor_loop(self):
        """监控主循环"""
        while self._running:
            try:
                status = self.get_status()
                triggered = self._check_thresholds(status)

                if triggered:
                    logger.warning(f"资源阈值告警 | {status}")
                    if self.cleanup_callback:
                        self.cleanup_callback(reason="threshold_exceeded", status=status)

            except Exception as e:
                logger.error(f"资源监控异常: {e}")

            time.sleep(self.check_interval)

    def _check_thresholds(self, status: dict) -> bool:
        """检查是否超过阈值"""
        triggered = False

        if status["threads"]["total"] > self.max_threads:
            logger.warning(f"线程数超阈值 | current={status['threads']['total']} | max={self.max_threads}")
            triggered = True

        if status["memory"]["percent"] > self.max_memory_percent:
            logger.warning(f"内存使用率超阈值 | current={status['memory']['percent']}% | max={self.max_memory_percent}%")
            triggered = True

        if status["gpu"]["available"] and status["gpu"]["percent"] > self.max_gpu_memory_percent:
            logger.warning(f"GPU显存使用率超阈值 | current={status['gpu']['percent']}% | max={self.max_gpu_memory_percent}%")
            triggered = True

        if status["ffmpeg"]["count"] > self.max_ffmpeg_count:
            logger.warning(f"FFmpeg进程数超阈值 | current={status['ffmpeg']['count']} | max={self.max_ffmpeg_count}")
            triggered = True

        return triggered

    @staticmethod
    def get_status() -> dict:
        """获取当前系统资源状态"""
        process = psutil.Process(os.getpid())
        memory = psutil.virtual_memory()

        # 统计线程
        thread_count = threading.active_count()

        # 统计 FFmpeg 进程
        ffmpeg_count = 0
        for proc in psutil.process_iter(["pid", "name"]):
            if proc.info["name"] == "ffmpeg":
                ffmpeg_count += 1

        # GPU 信息
        gpu_info = {"available": False, "used_mb": 0, "total_mb": 0, "percent": 0}
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info["available"] = True
                gpu_info["used_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_info["total_mb"] = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                gpu_info["percent"] = gpu_info["used_mb"] / gpu_info["total_mb"] * 100
        except ImportError:
            pass

        return {
            "threads": {
                "total": thread_count,
            },
            "memory": {
                "used_mb": memory.used / 1024 / 1024,
                "total_mb": memory.total / 1024 / 1024,
                "percent": memory.percent,
            },
            "gpu": gpu_info,
            "ffmpeg": {
                "count": ffmpeg_count,
            },
            "process": {
                "pid": os.getpid(),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
            },
        }
