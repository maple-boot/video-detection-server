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
    """算法模型池 — 同一算法共享模型，多 GPU 并行推理"""

    @staticmethod
    def _read_engine_metadata(engine_path: str) -> dict:
        """从 TensorRT engine 文件中读取 Ultralytics 元数据（兼容 pickle 和 JSON 格式）"""
        try:
            with open(engine_path, "rb") as f:
                data = f.read()
            # 优先尝试 pickle 格式
            magic = b"UlTralYtiCsEnGiNe"
            idx = data.rfind(magic)
            if idx >= 0:
                return pickle.loads(data[idx + len(magic):])
            # 回退: ultralytics 8.3+ JSON 格式
            meta_len = int.from_bytes(data[:4], byteorder="little")
            return json.loads(data[4:4 + meta_len].decode("utf-8"))
        except Exception:
            return {}

    def __init__(self, algorithm_id: str, model_path: str,
                 classes: list, gpu_ids: list, batch_size: int = 8,
                 eviction_callback=None):
        self.algorithm_id = algorithm_id
        self.model_path = model_path
        self.classes = classes
        self.gpu_ids = gpu_ids
        self.is_tensorrt = model_path.endswith(".engine")
        self.model_type = "TensorRT" if self.is_tensorrt else "PyTorch"

        # batch_size 先用配置值，TensorRT 引擎加载后从 TRT profile 提取真实值覆盖
        self.batch_size = batch_size

        self._models = {}
        self._queues = {}
        self._workers = []
        self._eviction_callback = eviction_callback
        self._load_models()
        self._start_workers()

        logger.info(
            f"模型池创建 | algorithm_id={algorithm_id} | "
            f"type={self.model_type} | gpus={self.gpu_ids} | batch_size={batch_size} | "
            f"path={os.path.basename(model_path)}"
        )

    def _load_models(self):
        """在每个 GPU 上并行加载模型（独立线程 + CUDA 设备绑定 + 显存预检 + OOM 驱逐回退）"""
        import torch

        # 根据 engine 文件大小动态估算显存需求
        if self.is_tensorrt:
            engine_size_mb = os.path.getsize(self.model_path) / (1 << 20)
            estimated_mib = max(engine_size_mb * 300, 4096)
            min_free_bytes = int(estimated_mib * (1 << 20))
            logger.info(
                f"动态显存阈值 | engine_size={engine_size_mb:.0f}MiB | "
                f"threshold={estimated_mib/1024:.1f}GiB"
            )
        else:
            min_free_bytes = 4 * (1 << 30)

        def _check_gpu(gpu_id):
            free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
            free_gb = free_bytes / (1 << 30)
            total_gb = total_bytes / (1 << 30)
            if free_bytes < min_free_bytes:
                logger.warning(
                    f"GPU {gpu_id} 显存不足 | free={free_gb:.1f}GiB / total={total_gb:.1f}GiB"
                )
                return False
            logger.info(f"GPU {gpu_id} 显存充足 | free={free_gb:.1f}GiB / total={total_gb:.1f}GiB")
            return True

        # 显存预检，跳过不足的 GPU
        candidates = [g for g in self.gpu_ids if _check_gpu(g)]

        # 如果无候选 GPU，尝试驱逐重复模型腾出空间
        if not candidates and self._eviction_callback:
            for gpu_id in self.gpu_ids:
                if self._eviction_callback(gpu_id):
                    torch.cuda.empty_cache()
                    # 重新检查显存
                    if _check_gpu(gpu_id):
                        candidates.append(gpu_id)
                        logger.info(f"GPU {gpu_id} 通过驱逐释放显存，加入候选")
                        break

        if not candidates:
            raise RuntimeError(
                f"所有推理 GPU 显存均不足 {min_free_bytes >> 30}GiB，且无法通过驱逐释放"
            )

        # 并行加载模型到候选 GPU
        loaded = {}
        errors = []
        lock = threading.Lock()
        loaded_event = threading.Event()
        _batch_extracted = threading.Event()

        def _load_on_gpu(gpu_id: int):
            nonlocal _batch_extracted
            try:
                torch.cuda.set_device(gpu_id)
                model = YOLO(self.model_path)

                # 从 TRT engine profile 提取 batch，替代预读元数据
                if self.is_tensorrt and not _batch_extracted.is_set():
                    try:
                        trt_engine = model.engine.engine  # ICudaEngine
                        input_name = trt_engine.get_tensor_name(0)
                        # profile 0 返回 (min_shape, opt_shape, max_shape)
                        opt_shape = trt_engine.get_tensor_profile_shape(input_name, 0)[1]
                        engine_batch = opt_shape[0]
                        if engine_batch != self.batch_size:
                            logger.info(
                                f"引擎 batch 覆盖 | algorithm_id={self.algorithm_id} | "
                                f"engine_batch={engine_batch} | config_batch={self.batch_size}"
                            )
                            self.batch_size = engine_batch
                    except Exception as e:
                        logger.warning(
                            f"无法从 engine 提取 batch | algorithm_id={self.algorithm_id} | "
                            f"error={e}"
                        )
                    _batch_extracted.set()

                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                if self.is_tensorrt:
                    model.predict(
                        [dummy] * self.batch_size,
                        verbose=False, batch=self.batch_size,
                    )
                else:
                    model.predict(dummy, verbose=False)

                # Warmup 后清理推理临时 buffer，保留模型权重常驻
                if self.is_tensorrt:
                    torch.cuda.empty_cache()

                with lock:
                    loaded[gpu_id] = model
                loaded_event.set()
                logger.info(f"模型加载成功 | algorithm_id={self.algorithm_id} | gpu={gpu_id}")
            except Exception as e:
                with lock:
                    errors.append((gpu_id, e))
                logger.error(f"模型加载失败 | algorithm_id={self.algorithm_id} | gpu={gpu_id} | error={e}")

        threads = []
        for gpu_id in candidates:
            t = threading.Thread(target=_load_on_gpu, args=(gpu_id,), daemon=True)
            t.start()
            threads.append(t)

        for t in threads:
            t.join(timeout=180)

        if not loaded:
            raise RuntimeError(f"所有候选 GPU 加载均失败: {errors}")

        if errors:
            logger.warning(
                f"部分 GPU 加载失败，降级运行 | "
                f"成功={list(loaded.keys())} | 失败={[e[0] for e in errors]}"
            )

        self.gpu_ids = list(loaded.keys())
        self._models = loaded

        # ── 记录每个 GPU 加载完成后的显存占用 ──
        mem_logger = get_memory_logger()
        for gpu_id in self.gpu_ids:
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
                used_bytes = total_bytes - free_bytes
                used_gib = used_bytes / (1 << 30)
                total_gib = total_bytes / (1 << 30)
                pct = used_bytes / total_bytes * 100
                mem_logger.info(
                    f"模型加载后显存 | algorithm_id={self.algorithm_id} | "
                    f"gpu={gpu_id} | type={self.model_type} | "
                    f"used={used_gib:.2f}GiB / {total_gib:.2f}GiB ({pct:.1f}%) | "
                    f"model={os.path.basename(self.model_path)}"
                )
            except Exception as e:
                mem_logger.warning(
                    f"获取显存信息失败 | algorithm_id={self.algorithm_id} | "
                    f"gpu={gpu_id} | error={e}"
                )

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
        # 工作线程固定到对应 GPU，确保 TRT 上下文和推理在同一设备
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

                # 推理时也确保设备上下文正确
                with torch.cuda.device(gpu_id):
                    result_holder["results"] = self._predict_slices(
                        model, slices, conf, imgsz, device,
                    )

                infer_time_ms = (time.time() - t0) * 1000
                result_holder["time"] = infer_time_ms
                result_event.set()

                task_count += 1
                total_infer_time += infer_time_ms

                # 每 50 次推理输出一次 GPU 性能
                if task_count % 50 == 0:
                    avg_infer = total_infer_time / task_count
                    logger.info(
                        f"GPU 推理性能 | gpu={gpu_id} | algorithm_id={self.algorithm_id} | "
                        f"总任务={task_count} | "
                        f"本批切片数={batch_n} | 本批耗时={infer_time_ms:.1f}ms | "
                        f"平均耗时={avg_infer:.1f}ms | "
                        f"device={device}"
                    )

                # 单批推理耗时过高时告警
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
        """推理切片 — TensorRT 用 engine 真实 batch 填充，PyTorch 按实际数量提交"""
        if not slices:
            return []
        if not self.is_tensorrt:
            return model.predict(
                slices, conf=conf, imgsz=imgsz, verbose=False, device=device,
            )

        all_results = []
        for i in range(0, len(slices), self.batch_size):
            chunk = list(slices[i:i + self.batch_size])
            chunk_batch = len(chunk)
            # TensorRT engine 为静态 batch，不足时用最后一帧填充
            if chunk_batch < self.batch_size:
                chunk += [chunk[-1]] * (self.batch_size - chunk_batch)
            # 不传 device= 参数，设备由调用方（_inference_loop）的 torch.cuda.device() 上下文管理
            results = model.predict(
                chunk, conf=conf, imgsz=imgsz,
                verbose=False, batch=self.batch_size,
            )
            all_results.extend(results[:chunk_batch])
        return all_results

    def infer_parallel(self, slices: list, conf: float = 0.75, imgsz: int = 640) -> tuple:
        """动态多 GPU 并行推理 — 切片不足时自动回退单卡全量 batch"""
        gpu_count = select_parallel_gpu_count(len(slices), self.gpu_ids, self.batch_size)
        t_start = time.time()

        if gpu_count == 1:
            results, infer_time = self._submit_and_wait(
                slices, conf, imgsz, self.gpu_ids[0]
            )
            total_time = (time.time() - t_start) * 1000
            logger.debug(
                f"infer_parallel 单卡 | slices={len(slices)} | "
                f"gpu={self.gpu_ids[0]} | infer={infer_time:.1f}ms | total={total_time:.1f}ms"
            )
            return results, infer_time, 1

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
        for i, holder in enumerate(holders):
            all_results.extend(holder["results"])
            gpu_times.append(holder["time"])

        infer_time = max(gpu_times) if gpu_times else 0
        total_time = (time.time() - t_start) * 1000
        logger.debug(
            f"infer_parallel 多卡 | slices={len(slices)} | "
            f"gpus={active_gpus} | chunks={[len(c) for c in chunks]} | "
            f"gpu_times={[f'{t:.1f}' for t in gpu_times]}ms | "
            f"max={infer_time:.1f}ms | total={total_time:.1f}ms"
        )
        return all_results, infer_time, gpu_count

    def _submit_and_wait(self, slices, conf, imgsz, gpu_id) -> tuple:
        """提交到指定 GPU 并等待结果"""
        event = threading.Event()
        holder = {"results": [], "time": 0}

        self._queues[gpu_id].put({
            "slices": slices,
            "conf": conf,
            "imgsz": imgsz,
            "event": event,
            "holder": holder,
        })

        event.wait(timeout=60.0)
        return holder["results"], holder["time"]

    def unload(self):
        """卸载模型池（所有 GPU）"""
        for gpu_id in list(self.gpu_ids):
            self._queues[gpu_id].put(None)
            if gpu_id in self._models:
                del self._models[gpu_id]
        self._models.clear()
        self.gpu_ids.clear()
        logger.info(f"模型池卸载 | algorithm_id={self.algorithm_id}")

    def unload_gpu(self, gpu_id: int):
        """从指定 GPU 卸载模型，保留其他 GPU 的副本供继续推理"""
        if gpu_id not in self.gpu_ids:
            return
        self._queues[gpu_id].put(None)
        if gpu_id in self._models:
            del self._models[gpu_id]
        self._queues.pop(gpu_id, None)
        self.gpu_ids.remove(gpu_id)
        logger.info(f"GPU {gpu_id} 模型卸载 | algorithm_id={self.algorithm_id} | 剩余 GPU={self.gpu_ids}")


class InferenceEngine:
    """推理引擎 — 池化管理，支持多 GPU 并行"""

    def __init__(self, config: dict):
        self.config = config
        self._pools = {}
        self._pool_refcount = {}  # algorithm_id -> 引用计数，跨任务共享保护
        self._lock = threading.Lock()

        roles = resolve_gpu_roles(config)
        self._video_gpu_id = roles["video_gpu_id"]
        self._inference_gpu_ids = roles["inference_gpu_ids"]
        self._batch_size = roles["batch_size"]
        self._gpu_count = roles["total_gpus"]

        logger.info(
            f"推理引擎初始化 | 可用GPU={self._gpu_count} | "
            f"编解码GPU={self._video_gpu_id} | 推理GPU={self._inference_gpu_ids} | "
            f"batch_size={self._batch_size} | 多卡阈值={self._batch_size}×卡数"
        )

    def _evict_gpu(self, gpu_id: int) -> bool:
        """
        在指定 GPU 上寻找可驱逐的模型池并卸载（仅卸载该 GPU 的副本）。
        条件：该模型池在其他 GPU 上仍有副本，卸载后不影响推理。
        """
        with self._lock:
            for alg_id, pool in list(self._pools.items()):
                if gpu_id not in pool.gpu_ids:
                    continue
                if len(pool.gpu_ids) <= 1:
                    # 该池只有这张 GPU 有模型，不能驱逐（否则完全不可用）
                    continue
                pool.unload_gpu(gpu_id)
                logger.warning(
                    f"驱逐 GPU {gpu_id} 上的模型 | algorithm_id={alg_id} | "
                    f"该池剩余 GPU={pool.gpu_ids}"
                )
                return True
            logger.warning(f"GPU {gpu_id} 上无符合驱逐条件的模型池")
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
                # [优化4] 优先加载 TensorRT engine（用 rsplit 避免路径中含 .pt 时的错误替换）
                engine_path = model_path.rsplit(".pt", 1)[0] + ".engine"
                if os.path.exists(engine_path):
                    actual_path = engine_path
                    is_tensorrt = True
                    logger.info(f"找到 TensorRT engine: {os.path.basename(engine_path)}")
                else:
                    # 未找到 .engine，回退到 .pt 模型
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

                gpu_ids = list(self._inference_gpu_ids)

                # 传入驱逐回调，当 GPU 显存不足时自动释放其他池的重复副本
                pool = ModelPool(
                    algorithm_id=algorithm_id,
                    model_path=actual_path,
                    classes=classes,
                    gpu_ids=gpu_ids,
                    batch_size=self._batch_size,
                    eviction_callback=self._evict_gpu,
                )

                self._pools[algorithm_id] = pool
                self._pool_refcount[algorithm_id] = 1
                pool_type = "TensorRT" if is_tensorrt else "PyTorch"
                logger.info(
                    f"模型池创建成功 | algorithm_id={algorithm_id} | "
                    f"type={pool_type} | gpus={gpu_ids}"
                )

                # ── 记录加载后所有推理 GPU 的全局显存快照 ──
                mem_logger = get_memory_logger()
                for gid in self._inference_gpu_ids:
                    try:
                        free_b, total_b = torch.cuda.mem_get_info(gid)
                        used_b = total_b - free_b
                        used_gi = used_b / (1 << 30)
                        total_gi = total_b / (1 << 30)
                        pct = used_b / total_b * 100
                        mem_logger.info(
                            f"推理引擎全局显存 | algorithm_id={algorithm_id} | "
                            f"gpu={gid} | used={used_gi:.2f}GiB / {total_gi:.2f}GiB ({pct:.1f}%)"
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
        """单帧检测"""
        pool = self._pools.get(algorithm_id)
        if pool is None:
            return [], 0, []

        t0 = time.time()
        results, infer_time = pool._submit_and_wait([frame], conf, imgsz, pool.gpu_ids[0])
        total_time = (time.time() - t0) * 1000

        detections = self._parse_results(results, pool.classes)
        if total_time > 100:
            logger.debug(
                f"detect | algorithm_id={algorithm_id} | "
                f"infer={infer_time:.1f}ms | total={total_time:.1f}ms | "
                f"detections={len(detections)} | gpu={pool.gpu_ids[0]}"
            )
        return detections, total_time, results

    def detect_sahi(self, algorithm_id: str, frame: np.ndarray,
                    conf: float = 0.75, imgsz: int = 640,
                    slice_size: int = 640, overlap_ratio: float = 0.1,
                    iou_threshold: float = 0.5) -> tuple:
        """SAHI 切片检测 — 多 GPU 并行推理（非 TensorRT 模型直接回退全图检测）"""
        pool = self._pools.get(algorithm_id)
        if pool is None:
            return [], 0, []

        # PyTorch 模型不做切片推理，直接全图检测
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
        actual_count = len(slices)
        t_slice_done = time.time()
        slice_create_ms = (t_slice_done - t0) * 1000

        all_results, infer_time, active_gpus = pool.infer_parallel(
            slices, conf=conf, imgsz=imgsz
        )

        inference_time = (time.time() - t0) * 1000
        t_infer_done = time.time()
        nms_merge_ms = (t_infer_done - t_slice_done - infer_time/1000) * 1000

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

        # logger.info(
        #     f"SAHI 切片检测 | algorithm_id={algorithm_id} | "
        #     f"原图={w}x{h} | slices={actual_count} | "
        #     f"active_gpus={active_gpus}/{len(pool.gpu_ids)} | "
        #     f"切片创建={slice_create_ms:.1f}ms | "
        #     f"推理={infer_time:.1f}ms | "
        #     f"NMS合并={nms_merge_ms:.1f}ms | "
        #     f"总耗时={inference_time:.1f}ms | "
        #     f"raw={len(all_detections)} | merged={len(merged)} | "
        #     f"slice_size={slice_size} | overlap={overlap_ratio}"
        # )

        return merged, inference_time, []

    def warmup_detect_with_targets(self, algorithm_id: str,
                                    imgsz: int = 640,
                                    slice_size: int = 640,
                                    overlap_ratio: float = 0.1,
                                    iou_threshold: float = 0.5):
        """
        用含模拟目标的图片做一次预热推理，
        触发 TensorRT 的检测输出路径初始化，避免首次真实检测时的 180ms+ 尖峰。
        """
        pool = self._pools.get(algorithm_id)
        if pool is None:
            return
        try:
            # 构造一张包含模拟白色框的图片（模拟检测目标）
            warmup_frame = np.zeros((720, 960, 3), dtype=np.uint8)
            cv2.rectangle(warmup_frame, (100, 100), (200, 200), (255, 255, 255), -1)
            cv2.rectangle(warmup_frame, (400, 300), (500, 400), (255, 255, 255), -1)

            self.detect_sahi(
                algorithm_id, warmup_frame,
                conf=0.75, imgsz=imgsz,
                slice_size=slice_size,
                overlap_ratio=overlap_ratio,
                iou_threshold=iou_threshold,
            )
            logger.info(f"模型检测预热完成 | algorithm_id={algorithm_id}")
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
            self._pools[algorithm_id].unload()
            del self._pools[algorithm_id]
            del self._pool_refcount[algorithm_id]
            logger.info(f"模型池已卸载 | algorithm_id={algorithm_id}")

    def get_loaded_models(self) -> list:
        return list(self._pools.keys())

    def clear_cache(self):
        with self._lock:
            for pool in self._pools.values():
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
