import threading
from concurrent.futures import ThreadPoolExecutor
from app.utils.logger import get_system_logger

logger = get_system_logger()


class VideoThreadPool:
    """视频任务线程池管理"""

    def __init__(self, max_workers: int = 50):
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="video_worker"
        )
        self._futures = {}
        self._lock = threading.Lock()
        self._active_tasks = {}
        logger.info(f"线程池初始化完成 | max_workers={max_workers}")

    def submit_task(self, task_id: str, func, *args, **kwargs):
        """提交任务到线程池"""
        with self._lock:
            if task_id in self._futures:
                logger.warning(f"任务已存在，跳过提交 | task_id={task_id}")
                return

            future = self._executor.submit(func, *args, **kwargs)
            self._futures[task_id] = future
            logger.info(f"任务已提交 | task_id={task_id}")

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self._lock:
            future = self._futures.get(task_id)
            if future:
                cancelled = future.cancel()
                if cancelled:
                    del self._futures[task_id]
                    logger.info(f"任务已取消 | task_id={task_id}")
                return cancelled
            return False

    def remove_task(self, task_id: str):
        """移除已完成的任务记录"""
        with self._lock:
            self._futures.pop(task_id, None)
            self._active_tasks.pop(task_id, None)

    def get_active_tasks(self) -> list:
        """获取活跃任务列表"""
        with self._lock:
            return list(self._futures.keys())

    def get_thread_count(self) -> int:
        """获取当前线程数"""
        return threading.active_count()

    def shutdown(self, wait: bool = True):
        """关闭线程池"""
        logger.info("线程池关闭中...")
        self._executor.shutdown(wait=wait)
        with self._lock:
            self._futures.clear()
            self._active_tasks.clear()
        logger.info("线程池已关闭")

    def force_cleanup(self):
        """强制清理所有任务"""
        with self._lock:
            count = len(self._futures)
            self._futures.clear()
            self._active_tasks.clear()
        logger.info(f"线程池强制清理完成 | 清理任务数={count}")
        return count
