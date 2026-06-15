import os
import time
import json
import pickle
import threading
import cv2
import torch
import numpy as np
from queue import Queue
from ultralytics import YOLO
from app.utils.gpu_allocator import resolve_gpu_roles, select_parallel_gpu_count
from app.utils.logger import get_task_logger, get_system_logger, get_memory_logger
from app.utils.detection_util import DetectionUtils

logger = get_system_logger()


class ModelPool:
    """算法模型池 — 多 GPU 并行推理，串行加载（避免 TRT Runtime 冲突）"""

    @staticmethod
    def _read_engine_metadata(engine_path: str) -> dict:
        """从 TensorRT engine 文件中读取 Ultralytics 元数据（兼容 pickle 和 JSON 格式）"""
        try:
            with open(engine_path, "rb") as f:
                data = f.read()
            magic = b"UlTralYtiCsEnGiNe"
            idx = data.rfind(magic)
            if idx >= 0:
                return pickle.loads(data[idx + len(magic):])
            meta_len = int.from_bytes(data[:4], byteorder="little")
            return json.loads(data[4:4 + meta_len].decode("utf-8"))
        except Exception:
            return {}

    def __init__(self, algorithm_id: str, model_path: str,
                 classes: list, gpu_ids: list, batch_size: int = 8):
        self.algorithm_id = algorithm_id
        self.model_path = model_path
        self.classes = classes
        self.gpu_ids = gpu_ids
        self.is_tensorrt = model_path.endswith(".engine")
        self.model_type = "TensorRT" if self.is_tensorrt else "PyTorch"
        self.batch_size = batch_size

        self._models = {}      # gpu_id -> YOLO model
        self._queues = {}      # gpu_id -> Queue
        self._workers = []     # worker threads

        # 从 engine 文件元数据中读取 batch
        if self.is_tensorrt:
            try:
                _meta = self._read_engine_metadata(self.model_path)
                if "batch" in _meta:
                    _engine_batch = int(_meta["batch"])
                    if _engine_batch != self.batch_size:
                        logger.info(
                            f"引擎 batch 覆盖 | algorithm_id={self.algorithm_id} | "
                            f"engine_batch={_engine_batch} | config_batch={self.batch_size}"
                        )
                        self.batch_size = _engine_batch
            except Exception as e:
                logger.warning(
                    f"无法从 engine 元数据读取 batch | algorithm_id={self.algorithm_id} | "
                    f"error={e}（使用配置 batch={self.batch_size}）"
                )

        self._load_models()
        self._start_workers()

        logger.info(
            f"模型池创建 | algorithm_id={algorithm_id} | "
            f"type={self.model_type} | gpus={self.gpu_ids} | batch_size={self.batch_size} | "
            f"path={os.path.basename(model_path)}"
        )

    def _load_models(self):
        """串行加载模型到所有 GPU（逐个加载，不并行，避免 TRT Runtime 冲突）"""
        import torch

        for gpu_id in self.gpu_ids:
            try:
                # 显存预检
                free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
                if self.is_tensorrt:
                    engine_size_mb = os.path.getsize(self.model_path) / (1 << 20)
                    estimated_mib = max(engine_size_mb * 300, 4096)
                    min_free_bytes = int(estimated_mib * (1 << 20))
                else:
                    min_free_bytes = 4 * (1 << 30)

                if free_bytes < min_free_bytes:
                    logger.warning(
                        f"GPU {gpu_id} 显存不足，跳过加载 | "
                        f"free={free_bytes/(1<<30):.1f}GiB, need={min_free_bytes/(1<<30):.1f}GiB"
                    )
                    continue

                # 加载（不传 device，让 YOLO 自动绑定到当前默认设备）
                model = YOLO(self.model_path)

                # Warmup：单帧，不传 device/batch
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                if self.is_tensorrt:
                    for _ in range(min(self.batch_size, 4)):
                        model.predict(dummy, verbose=False)
                else:
                    model.predict(dummy, verbose=False)

                self._models[gpu_id] = model
                logger.info(f"模型加载成功 | algorithm_id={self.algorithm_id} | gpu={gpu_id}")

                # 记录显存占用
                mem_logger = get_memory_logger()
                free_a, total_a = torch.cuda.mem_get_info(gpu_id)
                used_a = total_a - free_a
                mem_logger.info(
                    f"模型加载后显存 | algorithm_id={self.algorithm_id} | "
                    f"gpu={gpu_id} | type={self.model_type} | "
                    f"used={used_a/(1<<30):.2f}GiB / {total_a/(1<<30):.2f}GiB "
                    f"({used_a/total_a*100:.1f}%) | "
                    f"model={os.path.basename(self.model_path)}"
                )
            except Exception as e:
                logger.error(f"模型加载失败 | algorithm_id={self.algorithm_id} | gpu={gpu_id} | error={e}")
                continue

        if not self._models:
            raise RuntimeError(f"所有 GPU 加载均失败: {self.gpu_ids}")

        self.gpu_ids = list(self._models.keys())

    def _start_workers(self):
        """每个 GPU 启动一个推理工作线程"""
        for gpu_id in self.gpu_ids:
            queue = Queue()
            self._queues[gpu_id] = queue

            worker = threading.Thread(
                target=self._inference_loop,
                args=(gpu_id, queue),
                daemon=True,
                name=f"pool_{self.algorithm_id}_gpu{gpu_id}",
            )
            worker.start()
            self._workers.append(worker)

    def _inference_loop(self, gpu_id: int, queue: Queue):
        """GPU 推理工作线程 — 从队列取任务，串行执行"""
        torch.cuda.set_device(gpu_id)
        model = self._models[gpu_id]
        device = str(gpu_id)
        task_count = 0
        total_infer_time = 0.0

        while True:
            item = queue.get()
            if item is None:
                break

            try:
                slices = item["slices"]
                conf = item["conf"]
                imgsz = item["imgsz"]
                result_holder = item["holder"]
                result_event = item["event"]

                batch_n = len(slices)
                t0 = time.time()

                with torch.cuda.device(gpu_id):
                    result_holder["results"] = self._predict_slices(
                        model, slices, conf, imgsz, device,
                    )
                    torch.cuda.synchronize(gpu_id)

                infer_time_ms = (time.time() - t0) * 1000
                result_holder["time"] = infer_time_ms
                result_event.set()

                task_count += 1
                total_infer_time += infer_time_ms

                if task_count % 50 == 0:
                    avg_infer = total_infer_time / task_count
                    logger.info(
                        f"GPU 推理性能 | gpu={gpu_id} | algorithm_id={self.algorithm_id} | "
                        f"总任务={task_count} | "
                        f"本批切片数={batch_n} | 本批耗时={infer_time_ms:.1f}ms | "
                        f"平均耗时={avg_infer:.1f}ms"
                    )

                if batch_n > 0 and infer_time_ms > 200:
                    logger.warning(
                        f"推理耗时过高 | gpu={gpu_id} | algorithm_id={self.algorithm_id} | "
                        f"batch_n={batch_n} | time={infer_time_ms:.1f}ms | "
                        f"平均每帧={infer_time_ms/batch_n:.1f}ms"
                    )

            except Exception as e:
                logger.error(f"推理工作线程异常 | gpu={gpu_id} | error={e}")
                item["holder"]["results"] = []
                item["holder"]["time"] = 0
                item["event"].set()

    def _predict_slices(self, model, slices: list, conf: float, imgsz: int, device: str) -> list:
        """推理切片 — TensorRT 用实际数量提交（引擎 profile 支持动态 batch），PyTorch 按实际数量提交"""
        if not slices:
            return []
        if not self.is_tensorrt:
            return model.predict(
                slices, conf=conf, imgsz=imgsz, verbose=False, device=device,
            )

        all_results = []
        # TRT 引擎 profile: min=(1,3,640,640), max=(batch_size,3,640,640)
        # 支持动态 batch，无需填充虚帧
        for i in range(0, len(slices), self.batch_size):
            chunk = list(slices[i:i + self.batch_size])
            chunk_batch = len(chunk)
            results = model.predict(
                chunk, conf=conf, imgsz=imgsz,
                verbose=False, batch=chunk_batch,
            )
            all_results.extend(results)
        return all_results

    def infer(self, slices: list, conf: float = 0.75, imgsz: int = 640) -> tuple:
        """
        多 GPU 并行推理 — 根据切片数量动态分发
        返回 (all_results, max_infer_time_ms, active_gpu_count)
        """
        gpu_count = select_parallel_gpu_count(len(slices), self.gpu_ids, self.batch_size)
        t_start = time.time()

        if gpu_count == 1:
            # 单卡：提交到第一个 GPU
            event = threading.Event()
            holder = {"results": [], "time": 0}
            self._queues[self.gpu_ids[0]].put({
                "slices": slices,
                "conf": conf,
                "imgsz": imgsz,
                "event": event,
                "holder": holder,
            })
            event.wait(timeout=60.0)
            total_time = (time.time() - t_start) * 1000
            logger.debug(
                f"infer 单卡 | slices={len(slices)} | "
                f"gpu={self.gpu_ids[0]} | infer={holder['time']:.1f}ms | total={total_time:.1f}ms"
            )
            return holder["results"], holder["time"], 1

        # 多卡：切片分发
        active_gpus = self.gpu_ids[:gpu_count]
        chunk_size = (len(slices) + gpu_count - 1) // gpu_count
        chunks = [slices[i:i + chunk_size] for i in range(0, len(slices), chunk_size)]
        chunks = chunks[:gpu_count]

        events = []
        holders = []
        for i, chunk in enumerate(chunks):
            gpu_id = active_gpus[i]
            event = threading.Event()
            holder = {"results": [], "time": 0}
            self._queues[gpu_id].put({
                "slices": chunk,
                "conf": conf,
                "imgsz": imgsz,
                "event": event,
                "holder": holder,
            })
            events.append(event)
            holders.append(holder)

        for event in events:
            event.wait(timeout=60.0)

        all_results = []
        gpu_times = []
        for holder in holders:
            all_results.extend(holder["results"])
            gpu_times.append(holder["time"])

        infer_time = max(gpu_times) if gpu_times else 0
        total_time = (time.time() - t_start) * 1000
        logger.debug(
            f"infer 多卡 | slices={len(slices)} | "
            f"gpus={active_gpus} | chunks={[len(c) for c in chunks]} | "
            f"gpu_times={[f'{t:.1f}' for t in gpu_times]}ms | "
            f"max={infer_time:.1f}ms | total={total_time:.1f}ms"
        )
        return all_results, infer_time, gpu_count

    def unload(self):
        """卸载所有 GPU 上的模型"""
        for gpu_id in list(self.gpu_ids):
            if gpu_id in self._queues:
                self._queues[gpu_id].put(None)
            if gpu_id in self._models:
                del self._models[gpu_id]
        self._models.clear()
        self._queues.clear()
        logger.info(f"模型池卸载 | algorithm_id={self.algorithm_id} | gpus={self.gpu_ids}")


class InferenceEngine:
    """推理引擎 — 池化管理，每个模型独占一张推理卡"""

    def __init__(self, config: dict):
        self.config = config
        self._pools = {}
        self._pool_refcount = {}
        self._lock = threading.Lock()

        roles = resolve_gpu_roles(config)
        self._video_gpu_id = roles["video_gpu_id"]
        self._inference_gpu_ids = roles["inference_gpu_ids"]
        self._batch_size = roles["batch_size"]
        self._gpu_count = roles["total_gpus"]

        logger.info(
            f"推理引擎初始化 | 可用GPU={self._gpu_count} | "
            f"编解码GPU={self._video_gpu_id} | 推理GPU={self._inference_gpu_ids} | "
            f"batch_size={self._batch_size}"
        )

    def _estimate_model_mib(self, model_path: str) -> int:
        """估算模型加载后占用的显存（MiB）"""
        if not os.path.exists(model_path):
            return 4096
        size_mb = os.path.getsize(model_path) / (1 << 20)
        if model_path.endswith(".engine"):
            return int(max(size_mb * 300, 4096))
        return 4096  # PyTorch 模型保守估计 4GiB

    def _evict_oldest(self, keep_alg_id: str) -> bool:
        """驱逐最早加载的模型池释放显存（保留当前要加载的）"""
        for alg_id in list(self._pools.keys()):
            if alg_id == keep_alg_id:
                continue
            logger.warning(f"驱逐模型池释放显存 | algorithm_id={alg_id}")
            pool = self._pools.pop(alg_id, None)
            if pool:
                pool.unload()
                self._pool_refcount.pop(alg_id, None)
                # 清理所有涉及的 GPU 缓存
                for gid in pool.gpu_ids:
                    try:
                        with torch.cuda.device(gid):
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
            return True
        return False

    def load_model(self, algorithm_id: str, model_path: str, classes_path: str = "") -> bool:
        with self._lock:
            if algorithm_id in self._pools:
                self._pool_refcount[algorithm_id] += 1
                logger.info(
                    f"模型池引用 +1 | algorithm_id={algorithm_id} | "
                    f"refcount={self._pool_refcount[algorithm_id]}"
                )
                return True

            try:
                engine_path = model_path.rsplit(".pt", 1)[0] + ".engine"
                if os.path.exists(engine_path):
                    actual_path = engine_path
                    is_tensorrt = True
                    logger.info(f"找到 TensorRT engine: {os.path.basename(engine_path)}")
                else:
                    actual_path = model_path
                    is_tensorrt = False
                    logger.warning(
                        f"未找到 TensorRT engine ({os.path.basename(engine_path)})，"
                        f"回退到 PyTorch 模型: {os.path.basename(model_path)}"
                    )

                if not os.path.exists(actual_path):
                    logger.error(f"模型文件不存在: {actual_path}")
                    return False

                classes = []
                if classes_path and os.path.exists(classes_path):
                    with open(classes_path, "r", encoding="utf-8") as f:
                        classes = [line.strip() for line in f if line.strip()]

                # 估算显存需求，不足则驱逐旧模型
                need_mib = self._estimate_model_mib(actual_path)
                free_bytes, _ = torch.cuda.mem_get_info(self._inference_gpu_ids[0])
                evict_retry = 0
                while free_bytes < need_mib * (1 << 20) and evict_retry < len(self._pools) + 1:
                    if not self._evict_oldest(algorithm_id):
                        break
                    free_bytes, _ = torch.cuda.mem_get_info(self._inference_gpu_ids[0])
                    evict_retry += 1

                # 串行加载到所有推理 GPU（ModelPool 内部逐个加载）
                pool = ModelPool(
                    algorithm_id=algorithm_id,
                    model_path=actual_path,
                    classes=classes,
                    gpu_ids=list(self._inference_gpu_ids),
                    batch_size=self._batch_size,
                )

                self._pools[algorithm_id] = pool
                self._pool_refcount[algorithm_id] = 1
                pool_type = "TensorRT" if is_tensorrt else "PyTorch"
                logger.info(
                    f"模型池创建成功 | algorithm_id={algorithm_id} | "
                    f"type={pool_type} | gpus={pool.gpu_ids}"
                )

                # 记录全局显存快照
                mem_logger = get_memory_logger()
                for gid in self._inference_gpu_ids:
                    try:
                        free_b, total_b = torch.cuda.mem_get_info(gid)
                        used_b = total_b - free_b
                        mem_logger.info(
                            f"推理引擎全局显存 | algorithm_id={algorithm_id} | "
                            f"gpu={gid} | used={used_b/(1<<30):.2f}GiB / "
                            f"{total_b/(1<<30):.2f}GiB ({used_b/total_b*100:.1f}%)"
                        )
                    except Exception as e:
                        mem_logger.warning(
                            f"获取推理 GPU 显存失败 | algorithm_id={algorithm_id} | "
                            f"gpu={gid} | error={e}"
                        )
                return True

            except Exception as e:
                logger.error(f"模型池创建失败 | algorithm_id={algorithm_id} | error={e}")
                return False

    def detect(self, algorithm_id: str, frame: np.ndarray,
               conf: float = 0.75, imgsz: int = 640) -> tuple:
        """单帧检测，返回 (detections, inference_time_ms)"""
        pool = self._pools.get(algorithm_id)
        if pool is None:
            return [], 0
        results, infer_time, gpu_id = pool.infer([frame], conf=conf, imgsz=imgsz)
        detections = self._parse_results(results, pool.classes)
        return detections, infer_time

    def detect_sahi(self, algorithm_id: str, frame: np.ndarray,
                    conf: float = 0.75, imgsz: int = 640,
                    slice_size: int = 640, overlap_ratio: float = 0.1,
                    iou_threshold: float = 0.5) -> tuple:
        """SAHI 切片检测 — 单 GPU 推理"""
        pool = self._pools.get(algorithm_id)
        if pool is None:
            return [], 0, []

        if not pool.is_tensorrt:
            logger.debug(
                f"detect_sahi 回退全图检测 | algorithm_id={algorithm_id} | "
                f"model_type=PyTorch"
            )
            return self.detect(algorithm_id, frame, conf, imgsz)

        h, w = frame.shape[:2]
        if h <= slice_size and w <= slice_size:
            return self.detect(algorithm_id, frame, conf, imgsz)

        t0 = time.time()
        slices, slice_coords = self._create_slices(frame, h, w, slice_size, overlap_ratio)

        all_results, infer_time, gpu_id = pool.infer(
            slices, conf=conf, imgsz=imgsz,
        )

        all_detections = []
        for idx, result in enumerate(all_results):
            x_off, y_off = slice_coords[idx]
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf_val = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                x1 = max(0, min(float(box[0]) + x_off, w))
                y1 = max(0, min(float(box[1]) + y_off, h))
                x2 = max(0, min(float(box[2]) + x_off, w))
                y2 = max(0, min(float(box[3]) + y_off, h))
                class_name = pool.classes[cls_id] if cls_id < len(pool.classes) else str(cls_id)
                all_detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf_val,
                    "class_id": cls_id,
                    "class_name": class_name,
                })

        merged = DetectionUtils.nms_merge(all_detections, iou_threshold)
        inference_time = (time.time() - t0) * 1000
        return merged, inference_time, []

    def warmup_detect_with_targets(self, algorithm_id: str,
                                    imgsz: int = 640,
                                    slice_size: int = 640,
                                    overlap_ratio: float = 0.1,
                                    iou_threshold: float = 0.5):
        """用含模拟目标的图片做一次预热推理"""
        pool = self._pools.get(algorithm_id)
        if pool is None:
            return
        try:
            warmup_frame = np.zeros((720, 960, 3), dtype=np.uint8)
            cv2.rectangle(warmup_frame, (100, 100), (200, 200), (255, 255, 255), -1)
            cv2.rectangle(warmup_frame, (400, 300), (500, 400), (255, 255, 255), -1)
            for i in range(5):
                self.detect_sahi(
                    algorithm_id, warmup_frame,
                    conf=0.75, imgsz=imgsz,
                    slice_size=slice_size,
                    overlap_ratio=overlap_ratio,
                    iou_threshold=iou_threshold,
                )
            logger.info(f"模型检测预热完成 | algorithm_id={algorithm_id}（5 轮）")
        except Exception as e:
            logger.warning(f"模型检测预热跳过: {e}")

    def _create_slices(self, frame, h, w, slice_size, overlap_ratio):
        """SAHI 切片"""
        step = int(slice_size * (1 - overlap_ratio))
        y_positions = list(range(0, max(1, h - slice_size + 1), step))
        x_positions = list(range(0, max(1, w - slice_size + 1), step))
        if y_positions[-1] + slice_size < h:
            y_positions.append(h - slice_size)
        if x_positions[-1] + slice_size < w:
            x_positions.append(w - slice_size)
        slices = []
        slice_coords = []
        for y in y_positions:
            for x in x_positions:
                y_end = min(y + slice_size, h)
                x_end = min(x + slice_size, w)
                slice_img = frame[y:y_end, x:x_end]
                sh, sw = slice_img.shape[:2]
                if sh < slice_size or sw < slice_size:
                    padded = np.full((slice_size, slice_size, 3), 114, dtype=np.uint8)
                    padded[:sh, :sw] = slice_img
                    slice_img = padded
                slices.append(slice_img)
                slice_coords.append((x, y))
        return slices, slice_coords

    def _parse_results(self, results, classes):
        """解析检测结果"""
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf_val = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                class_name = classes[cls_id] if cls_id < len(classes) else str(cls_id)
                detections.append({
                    "bbox": box.tolist(),
                    "confidence": conf_val,
                    "class_id": cls_id,
                    "class_name": class_name,
                })
        return detections

    def unload_model(self, algorithm_id: str):
        with self._lock:
            if algorithm_id not in self._pools:
                return
            self._pool_refcount[algorithm_id] -= 1
            if self._pool_refcount[algorithm_id] > 0:
                logger.info(
                    f"模型池引用 -1，保留 | algorithm_id={algorithm_id} | "
                    f"refcount={self._pool_refcount[algorithm_id]}"
                )
                return
            pool = self._pools.pop(algorithm_id, None)
            if pool:
                pool_gpus = list(pool.gpu_ids)
                pool.unload()
                self._pool_refcount.pop(algorithm_id, None)
                for gid in pool_gpus:
                    try:
                        with torch.cuda.device(gid):
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
            logger.info(f"模型池已卸载 | algorithm_id={algorithm_id}")

    def get_loaded_models(self) -> list:
        return list(self._pools.keys())

    def clear_cache(self):
        with self._lock:
            for alg_id, pool in list(self._pools.items()):
                pool.unload()
            self._pools.clear()
            self._pool_refcount.clear()
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        with torch.cuda.device(i):
                            torch.cuda.empty_cache()
                    logger.info("所有 GPU 显存缓存已清空")
            except ImportError:
                pass
            logger.info("所有模型池已卸载")
