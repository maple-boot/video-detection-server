"""GPU 角色分配 — 编解码与推理隔离，推理侧动态多卡"""

import torch
from app.utils.logger import get_system_logger

logger = get_system_logger()


def get_gpu_count() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def resolve_gpu_roles(config: dict) -> dict:
    """
    解析 GPU 职责：
    - video_gpu_id: FFmpeg 解码/编码专用
    - inference_gpu_ids: 推理专用（多卡时排除 video_gpu_id）
    """
    ffmpeg_config = config.get("ffmpeg", {})
    inference_config = config.get("inference", {})
    video_gpu_id = ffmpeg_config.get("video_gpu_id", 0)
    batch_size = inference_config.get("batch_size", 8)

    total = get_gpu_count()
    if total == 0:
        logger.warning("未检测到 CUDA GPU，推理将回退 CPU")
        return {
            "video_gpu_id": 0,
            "inference_gpu_ids": [0],
            "batch_size": batch_size,
            "total_gpus": 0,
        }

    if total == 1:
        inference_gpu_ids = [0]
    else:
        inference_gpu_ids = [i for i in range(total) if i != video_gpu_id]
        if not inference_gpu_ids:
            inference_gpu_ids = [0]

    logger.info(
        f"GPU 角色分配 | 总数={total} | 编解码GPU={video_gpu_id} | "
        f"推理GPU={inference_gpu_ids} | batch_size={batch_size}"
    )

    return {
        "video_gpu_id": video_gpu_id,
        "inference_gpu_ids": inference_gpu_ids,
        "batch_size": batch_size,
        "total_gpus": total,
    }


def select_parallel_gpu_count(slice_count: int, gpu_ids: list, batch_size: int) -> int:
    """
    根据切片数量动态决定并行 GPU 数。
    每张卡至少承担 batch_size 个切片，且总切片数须达到 batch_size * 并行卡数。
    """
    if slice_count <= 1 or len(gpu_ids) <= 1:
        return 1

    max_gpus = min(len(gpu_ids), slice_count // batch_size)
    if max_gpus <= 1:
        return 1

    if slice_count < batch_size * max_gpus:
        return 1

    return max_gpus
