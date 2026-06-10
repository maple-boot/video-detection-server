import threading
from app.core.stream_worker import StreamWorker
from app.core.mp4_worker import MP4Worker
from app.utils.logger import get_system_logger

logger = get_system_logger()


class StreamSupervisor:
    """多路流 Supervisor — 监控所有 Worker，自动拉起崩溃的流"""

    def __init__(self, config: dict, orm_helper=None, minio_helper=None, inference_engine=None):
        self.config = config
        self.orm_helper = orm_helper
        self.minio_helper = minio_helper
        self.inference_engine = inference_engine

        self._workers = {}  # worker_task_id -> worker
        self._worker_threads = {}  # worker_task_id -> thread
        self._task_algorithm_map = {}  # original_task_id -> [worker_task_id, ...]
        self._lock = threading.Lock()
        logger.info("StreamSupervisor 初始化完成")

    def start_stream_worker(self, task_id: str, algorithm_ids: list,
                            stream_url: str, push_url: str,
                            platform_id: str = "", minio_name: str = "default",
                            original_task_id: str = "") -> bool:
        """启动 RTMP 直播流 Worker"""
        with self._lock:
            if task_id in self._workers:
                logger.warning(f"任务已存在 | task_id={task_id} | original_task_id={original_task_id}")
                return False

            worker = StreamWorker(
                task_id=task_id,
                algorithm_ids=algorithm_ids,
                stream_url=stream_url,
                push_url=push_url,
                config=self.config,
                orm_helper=self.orm_helper,
                minio_helper=self.minio_helper,
                inference_engine=self.inference_engine,
                platform_id=platform_id,
                minio_name=minio_name,
                original_task_id=original_task_id,
            )

            thread = threading.Thread(
                target=worker.start,
                daemon=True,
                name=f"stream_worker_{task_id}"
            )

            self._workers[task_id] = worker
            self._worker_threads[task_id] = thread

            # 维护原始 task_id 到 worker_task_id 的映射
            orig_id = original_task_id or task_id
            if orig_id not in self._task_algorithm_map:
                self._task_algorithm_map[orig_id] = []
            if task_id not in self._task_algorithm_map[orig_id]:
                self._task_algorithm_map[orig_id].append(task_id)

            thread.start()

            logger.info(f"StreamWorker 已启动 | task_id={task_id}")
            return True

    def start_mp4_worker(self, task_id: str, algorithm_ids: list,
                         file_path: str, output_path: str) -> bool:
        """启动 MP4 文件 Worker"""
        with self._lock:
            if task_id in self._workers:
                logger.warning(f"任务已存在 | task_id={task_id}")
                return False

            worker = MP4Worker(
                task_id=task_id,
                algorithm_ids=algorithm_ids,
                file_path=file_path,
                output_path=output_path,
                config=self.config,
                inference_engine=self.inference_engine,
            )

            thread = threading.Thread(
                target=worker.start,
                daemon=True,
                name=f"mp4_worker_{task_id}"
            )

            self._workers[task_id] = worker
            self._worker_threads[task_id] = thread
            thread.start()

            logger.info(f"MP4Worker 已启动 | task_id={task_id}")
            return True

    def stop_worker(self, task_id: str) -> bool:
        """停止指定 Worker（支持 worker_task_id 或原始 task_id）"""
        with self._lock:
            # 先尝试直接匹配
            worker = self._workers.get(task_id)
            if worker:
                worker.stop()
                del self._workers[task_id]
                thread = self._worker_threads.pop(task_id, None)
                if thread:
                    thread.join(timeout=10)
                # 清理 _task_algorithm_map 中的引用
                for orig_id, worker_ids in list(self._task_algorithm_map.items()):
                    if task_id in worker_ids:
                        worker_ids.remove(task_id)
                        if not worker_ids:
                            del self._task_algorithm_map[orig_id]
                        break
                logger.info(f"Worker 已停止 | task_id={task_id}")
                return True

            # 尝试按原始 task_id 停止所有相关 Worker
            worker_ids = self._task_algorithm_map.get(task_id, [])
            stopped = False
            for worker_id in list(worker_ids):
                worker = self._workers.get(worker_id)
                if worker:
                    worker.stop()
                    del self._workers[worker_id]
                    thread = self._worker_threads.pop(worker_id, None)
                    if thread:
                        thread.join(timeout=10)
                    logger.info(f"Worker 已停止 | task_id={worker_id}")
                    stopped = True

            if stopped:
                self._task_algorithm_map.pop(task_id, None)

            return stopped

    def stop_all(self):
        """停止所有 Worker"""
        with self._lock:
            for task_id, worker in list(self._workers.items()):
                try:
                    worker.stop()
                    logger.info(f"Worker 已停止 | task_id={task_id}")
                except Exception as e:
                    logger.error(f"Worker 停止异常 | task_id={task_id} | error={e}")

            self._workers.clear()
            self._worker_threads.clear()
            self._task_algorithm_map.clear()
            logger.info("所有 Worker 已停止")

    def get_worker(self, task_id: str):
        """获取 Worker 实例（支持 worker_task_id 或原始 task_id）"""
        # 先尝试直接匹配
        worker = self._workers.get(task_id)
        if worker:
            return worker

        # 尝试按原始 task_id 返回第一个匹配的 Worker
        worker_ids = self._task_algorithm_map.get(task_id, [])
        for worker_id in worker_ids:
            worker = self._workers.get(worker_id)
            if worker:
                return worker

        return None

    def get_worker_by_algorithm(self, task_id: str, algorithm_id: str):
        """按原始 task_id + algorithm_id 精确获取 Worker"""
        worker_id = f"{task_id}_{algorithm_id}"
        return self._workers.get(worker_id)

    def get_all_workers(self) -> dict:
        """获取所有 Worker"""
        with self._lock:
            return {
                task_id: {
                    "state": worker.state.value if hasattr(worker, "state") else "unknown",
                    "type": "stream" if isinstance(worker, StreamWorker) else "mp4",
                }
                for task_id, worker in self._workers.items()
            }

    def get_task_registry(self) -> dict:
        """获取任务注册表（供清理调度器使用）"""
        with self._lock:
            registry = {}
            for task_id, worker in self._workers.items():
                registry[task_id] = {
                    "task_id": task_id,
                    "worker": worker,
                    "restart_func": worker.get_restart_func() if hasattr(worker, "get_restart_func") else None,
                }
            return registry

    def get_active_task_count(self) -> int:
        """获取活跃任务数"""
        with self._lock:
            return len(self._workers)
