import os
import time
import threading
import cv2
import numpy as np
from queue import Queue
from ultralytics import YOLO
from app.utils.gpu_allocator import resolve_gpu_roles, select_parallel_gpu_count
from app.utils.logger import get_task_logger, get_system_logger

logger = get_system_logger()


class ModelPool:
    """算法模型池 — 同一算法共享模型，多 GPU 并行推理"""

    def __init__(self, algorithm_id: str, model_path: str,
                 classes: list, gpu_ids: list, batch_size: int = 8):
        self.algorithm_id = algorithm_id
        self.model_path = model_path
        self.classes = classes
        self.gpu_ids = gpu_ids
        self.batch_size = batch_size
        self.is_tensorrt = model_path.endswith(".engine")
        self.model_type = "TensorRT" if self.is_tensorrt else "PyTorch"

        self._models = {}
        self._queues = {}
        self._workers = []
        self._load_models()
        self._start_workers()

        logger.info(
            f"模型池创建 | algorithm_id={algorithm_id} | "
            f"type={self.model_type} | gpus={gpu_ids} | batch_size={batch_size} | "
            f"path={os.path.basename(model_path)}"
        )

    def _load_models(self):
        """在每个 GPU 上加载模型"""
        for gpu_id in self.gpu_ids:
            device = str(gpu_id)
            model = YOLO(self.model_path)

            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            if self.is_tensorrt:
                for warmup_batch in (1, self.batch_size):
                    model.predict(
                        [dummy] * warmup_batch,
                        device=device, verbose=False, batch=warmup_batch,
                    )
            else:
                model.predict(dummy, device=device, verbose=False)

            self._models[gpu_id] = model
            logger.info(f"模型加载 | algorithm_id={self.algorithm_id} | gpu={gpu_id} | path={self.model_path}")

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
        """推理切片 — TensorRT 动态 batch，按实际数量提交，不补 dummy"""
        if not slices:
            return []
        if not self.is_tensorrt:
            return model.predict(
                slices, conf=conf, imgsz=imgsz, verbose=False, device=device,
            )

        all_results = []
        for i in range(0, len(slices), self.batch_size):
            chunk = slices[i:i + self.batch_size]
            chunk_batch = len(chunk)
            results = model.predict(
                chunk, conf=conf, imgsz=imgsz,
                verbose=False, device=device, batch=chunk_batch,
            )
            all_results.extend(results)
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
        """卸载模型池"""
        for gpu_id in self.gpu_ids:
            self._queues[gpu_id].put(None)
            if gpu_id in self._models:
                del self._models[gpu_id]
        logger.info(f"模型池卸载 | algorithm_id={self.algorithm_id}")


class InferenceEngine:
    """推理引擎 — 池化管理，支持多 GPU 并行"""

    def __init__(self, config: dict):
        self.config = config
        self._pools = {}
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

    def load_model(self, algorithm_id: str, model_path: str, classes_path: str = "") -> bool:
        with self._lock:
            if algorithm_id in self._pools:
                logger.info(f"模型池已存在 | algorithm_id={algorithm_id}")
                return True

            try:
                # 优先加载 TensorRT engine
                engine_path = model_path.replace(".pt", ".engine")
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

                pool = ModelPool(
                    algorithm_id=algorithm_id,
                    model_path=actual_path,
                    classes=classes,
                    gpu_ids=gpu_ids,
                    batch_size=self._batch_size,
                )

                self._pools[algorithm_id] = pool
                pool_type = "TensorRT" if is_tensorrt else "PyTorch"
                logger.info(
                    f"模型池创建成功 | algorithm_id={algorithm_id} | "
                    f"type={pool_type} | gpus={gpu_ids}"
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

        merged = self._nms_merge(all_detections, iou_threshold)

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

    @staticmethod
    def _nms_merge(detections: list, iou_threshold: float = 0.5) -> list:
        if not detections:
            return []

        class_groups = {}
        for det in detections:
            cls_id = det["class_id"]
            if cls_id not in class_groups:
                class_groups[cls_id] = []
            class_groups[cls_id].append(det)

        merged = []
        for cls_id, dets in class_groups.items():
            dets.sort(key=lambda x: x["confidence"], reverse=True)
            keep = []
            while dets:
                best = dets.pop(0)
                keep.append(best)
                remaining = []
                for det in dets:
                    iou = InferenceEngine._calculate_iou(best["bbox"], det["bbox"])
                    if iou < iou_threshold:
                        remaining.append(det)
                dets = remaining
            merged.extend(keep)

        return merged

    @staticmethod
    def _calculate_iou(box1: list, box2: list) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        if inter_area == 0:
            return 0.0

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter_area / (box1_area + box2_area - inter_area)

    def unload_model(self, algorithm_id: str):
        with self._lock:
            if algorithm_id in self._pools:
                self._pools[algorithm_id].unload()
                del self._pools[algorithm_id]
                logger.info(f"模型池已卸载 | algorithm_id={algorithm_id}")

    def get_loaded_models(self) -> list:
        return list(self._pools.keys())

    def clear_cache(self):
        with self._lock:
            for pool in self._pools.values():
                pool.unload()
            self._pools.clear()
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
