import time
import threading
import psutil
from datetime import datetime
from app.utils.logger import get_cleanup_logger

logger = get_cleanup_logger()


class CleanupScheduler:
    """定时清理调度器"""

    def __init__(self, config: dict, thread_pool=None, inference_engine=None,
                 resource_monitor=None, task_registry=None):
        cleanup_config = config.get("cleanup", {})
        self.enabled = cleanup_config.get("enabled", True)
        self.schedule_time = cleanup_config.get("schedule", "03:00")
        self.graceful_timeout = cleanup_config.get("graceful_timeout", 30)
        self.force_kill_timeout = cleanup_config.get("force_kill_timeout", 5)
        self.restart_after_cleanup = cleanup_config.get("restart_after_cleanup", True)

        self.thread_pool = thread_pool
        self.inference_engine = inference_engine
        self.resource_monitor = resource_monitor
        self.task_registry = task_registry or {}

        self._running = False
        self._thread = None
        self._cleanup_lock = threading.Lock()
        logger.info(f"定时清理调度器初始化 | schedule={self.schedule_time} | enabled={self.enabled}")

    def start(self):
        """启动调度线程"""
        if not self.enabled:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._schedule_loop,
            daemon=True,
            name="cleanup_scheduler"
        )
        self._thread.start()
        logger.info("定时清理调度线程已启动")

    def stop(self):
        """停止调度线程"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("定时清理调度线程已停止")

    def _schedule_loop(self):
        """调度主循环"""
        while self._running:
            now = datetime.now().strftime("%H:%M")
            if now == self.schedule_time:
                logger.info(f"定时清理触发 | time={now}")
                self.execute_cleanup(reason="scheduled")
                # 等待 61 秒避免同一分钟重复触发
                time.sleep(61)
            time.sleep(30)

    def execute_cleanup(self, reason: str = "manual", status: dict = None) -> dict:
        """执行清理流程"""
        with self._cleanup_lock:
            cleanup_report = {
                "reason": reason,
                "start_time": time.time(),
                "tasks_stopped": 0,
                "threads_before": 0,
                "threads_after": 0,
                "ffmpeg_killed": 0,
                "memory_freed_mb": 0,
            }

            logger.info(f"开始清理 | reason={reason}")

            # 记录清理前线程数
            cleanup_report["threads_before"] = threading.active_count()
            memory_before = psutil.virtual_memory().used / 1024 / 1024

            # Step 1: 停止空闲/异常/卡死 Worker
            tasks_to_restart = []
            if self.task_registry:
                for task_id, worker_info in list(self.task_registry.items()):
                    worker = worker_info.get("worker")
                    if worker and hasattr(worker, "state"):
                        # 跳过正常运行的任务
                        if worker.state.value == "running":
                            logger.info(f"跳过活跃任务 | task_id={task_id}")
                            continue
                        # 只停止异常/重试中的任务
                        if worker.state.value in ("retrying", "init"):
                            tasks_to_restart.append(worker_info)
                            worker.stop()
                            cleanup_report["tasks_stopped"] += 1
                logger.info(f"停止异常任务完成 | count={cleanup_report['tasks_stopped']}")

            # Step 2: 等待线程退出
            time.sleep(2)

            # Step 3: 杀死残留 FFmpeg 进程
            ffmpeg_killed = self._kill_ffmpeg_processes()
            cleanup_report["ffmpeg_killed"] = ffmpeg_killed

            # Step 4: 清空推理引擎缓存
            if self.inference_engine:
                self.inference_engine.clear_cache()
                logger.info("推理引擎缓存已清空")

            # Step 5: 清空线程池
            if self.thread_pool:
                self.thread_pool.force_cleanup()

            # Step 6: GC
            import gc
            gc.collect()

            # 记录清理后状态
            cleanup_report["threads_after"] = threading.active_count()
            memory_after = psutil.virtual_memory().used / 1024 / 1024
            cleanup_report["memory_freed_mb"] = memory_before - memory_after
            cleanup_report["end_time"] = time.time()
            cleanup_report["duration"] = cleanup_report["end_time"] - cleanup_report["start_time"]

            logger.info(
                f"清理完成 | 耗时={cleanup_report['duration']:.1f}s | "
                f"停止任务={cleanup_report['tasks_stopped']} | "
                f"杀死FFmpeg={cleanup_report['ffmpeg_killed']} | "
                f"释放内存={cleanup_report['memory_freed_mb']:.0f}MB | "
                f"线程 {cleanup_report['threads_before']} → {cleanup_report['threads_after']}"
            )

            # Step 7: 重新拉起任务
            if self.restart_after_cleanup and tasks_to_restart:
                logger.info(f"准备重新拉起 {len(tasks_to_restart)} 个任务...")
                time.sleep(2)
                for task_info in tasks_to_restart:
                    restart_func = task_info.get("restart_func")
                    if restart_func and callable(restart_func):
                        try:
                            restart_func()
                            logger.info(f"任务重新拉起成功 | task_id={task_info.get('task_id')}")
                        except Exception as e:
                            logger.error(f"任务重新拉起失败 | task_id={task_info.get('task_id')} | error={e}")

            return cleanup_report

    def _kill_ffmpeg_processes(self) -> int:
        """杀死系统中由本服务启动的 FFmpeg 进程"""
        killed = 0
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if proc.info["name"] == "ffmpeg":
                    cmdline = " ".join(proc.info.get("cmdline") or [])
                    if "pipe:" in cmdline or "rtmp://" in cmdline or "nobuffer" in cmdline:
                        proc.kill()
                        proc.wait(timeout=self.force_kill_timeout)
                        killed += 1
                        logger.info(f"杀死残留 FFmpeg 进程 | pid={proc.info['pid']}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                pass

        if killed > 0:
            logger.info(f"FFmpeg 进程清理完成 | killed={killed}")
        return killed
