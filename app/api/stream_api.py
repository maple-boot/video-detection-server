import datetime
import json
import os
import re
import time
import cv2
from pydantic import BaseModel
from typing import List, Optional
from fastapi import APIRouter
from app.models.schemas import StreamRequest, StreamUrlRequest, LocationRequest, StopTaskRequest
from app.utils.response_helper import ResponseHelper
from app.utils.logger import get_system_logger

logger = get_system_logger()

router = APIRouter()

# 全局引用（在 main.py 中注入）
_supervisor = None
_orm_helper = None
_config = None
_minio_helper = None


class TransferDetectionBoxRequest(BaseModel):
    """检测框转移请求"""
    recognized_image_path: str  # 识别后图片在MinIO中的object key
    unrecognized_image_path: str  # 未识别图片在MinIO中的object key
    bounding_boxes: str  # 边界框坐标字符串，格式为 "[[x1, y1, x2, y2], ...]" 或 "[x1, y1, x2, y2]"


def _get_push_url_prefix():
    """从配置中获取推流地址前缀"""
    if _config:
        return _config.get("srs", {}).get("push_url_prefix", "rtmp://localhost/live/stream/")
    return "rtmp://localhost/live/stream/"


def init_stream_api(supervisor, orm_helper, config=None, minio_helper=None):
    global _supervisor, _orm_helper, _config, _minio_helper
    _supervisor = supervisor
    _orm_helper = orm_helper
    _config = config
    _minio_helper = minio_helper


@router.post("/stream")
async def start_stream(request: StreamRequest):
    """启动视频流检测任务"""
    try:
        task_id = request.taskId
        algorithm_ids = [request.algorithmId]
        stream_url = request.streamUrl
        platform_id = request.platformId
        minio_name = request.minioName

        if not task_id or not algorithm_ids or not stream_url:
            return ResponseHelper.bad_request("缺少必要参数")

        # 拦截：检查任务是否已存在
        if _orm_helper:
            for alg_id in algorithm_ids:
                existing = _orm_helper.get_task_record(int(task_id), alg_id)
                if existing:
                    # 跳过打印
                    # logger.info(f"任务已存在，跳过 | task_id={task_id} | algorithm_id={alg_id}")
                    push_url_prefix = _get_push_url_prefix()
                    push_url = f"{push_url_prefix}{task_id}_{alg_id}"
                    return ResponseHelper.success(
                        data={"output_url": push_url.replace('rtmp://', 'webrtc://'), "taskId": task_id}
                    )

        # 协议替换
        if "webrtc" in stream_url:
            stream_url = stream_url.replace("webrtc", "rtmp")

        # 判断流类型
        is_file = ".mp4" in stream_url or ".avi" in stream_url or ".mkv" in stream_url

        if is_file:
            output_path = f"output/{task_id}_output.mp4"
            success = _supervisor.start_mp4_worker(
                task_id=task_id,
                algorithm_ids=algorithm_ids,
                file_path=stream_url,
                output_path=output_path,
            )
            if success:
                worker_task_id = f"{task_id}_{algorithm_ids}"
                push_url_prefix = _get_push_url_prefix()
                push_url = f"{push_url_prefix}{worker_task_id}"
                return ResponseHelper.success(data={"output_url": push_url.replace('rtmp://', 'webrtc://'), "taskId": task_id})
            else:
                return ResponseHelper.error("任务启动失败")
        else:
            # 每个算法独立 Worker + 独立推流地址
            push_url_prefix = _get_push_url_prefix()
            results = []
            for alg_id in algorithm_ids:
                worker_task_id = f"{task_id}_{alg_id}"
                push_url = f"{push_url_prefix}{worker_task_id}"

                success = _supervisor.start_stream_worker(
                    task_id=worker_task_id,
                    algorithm_ids=[alg_id],
                    stream_url=stream_url,
                    push_url=push_url,
                    platform_id=platform_id,
                    minio_name=minio_name,
                    original_task_id=task_id,  # 传递原始task_id 用来操作task_id下所有任务的进程状态
                )
                results.append({
                    "algorithmId": alg_id,
                    "taskId": worker_task_id,
                    "pushUrl": push_url.replace('rtmp://', 'webrtc://'),
                    "success": success,
                })

                # 保存任务记录
                if success and _orm_helper:
                    _orm_helper.add_task_record(
                        task_id=int(task_id),
                        algorithm_id=alg_id,
                        input_url=stream_url,
                        output_url=push_url,
                    )

            # 检查是否全部成功
            failed = [r for r in results if not r["success"]]
            if failed:
                return ResponseHelper.error(f"部分任务启动失败: {failed}")

            return ResponseHelper.success(data={
                "taskId": task_id,
                "streams": results,
                "output_url":results[0]["pushUrl"],
            })

    except Exception as e:
        logger.error(f"启动任务异常: {e}")
        return ResponseHelper.error(f"启动任务异常: {str(e)}")


@router.post("/stream/url")
async def get_stream_url(request: StreamUrlRequest):
    time.sleep(5)
    logger.info("========================请求开始==============================")
    """查询任务推流地址"""
    try:
        task_id = request.taskId
        alg_id = request.algorithmId

        for attempt in range(3):
            if _orm_helper:
                existing = _orm_helper.get_task_record(int(task_id), alg_id)
                if existing:
                    worker_task_id = f"{task_id}_{alg_id}"
                    push_url_prefix = _get_push_url_prefix()
                    push_url = f"{push_url_prefix}{worker_task_id}"
                    response = ResponseHelper.success(data={"output_url": push_url.replace('rtmp://', 'webrtc://'), "taskId": task_id})
                    logger.info(f"查询到结果 | task_id={task_id} | algorithm_id={alg_id}, 返回：{response}")
                    return ResponseHelper.success(
                        data={"output_url": push_url.replace('rtmp://', 'webrtc://'), "taskId": task_id}
                    )
            time.sleep(2)
        return ResponseHelper.not_found(f"任务不存在或已停止: {task_id}")

    except Exception as e:
        logger.error(f"查询推流地址异常: {e}")
        return ResponseHelper.error(f"查询异常: {str(e)}")

@router.post("/stream/stop")
async def stop_stream(request: StopTaskRequest):
    """停止指定任务"""
    try:
        task_id = request.taskId
        algorithm_id = request.algorithmId

        if not task_id or not algorithm_id:
            return ResponseHelper.bad_request("缺少必要参数 taskId 或 algorithmId")

        # 构造 worker_task_id
        worker_task_id = f"{task_id}_{algorithm_id}"
        logger.info(f"需要关闭的任务ID | worker_task_id={worker_task_id} ")

        # 停止 Worker
        success = _supervisor.stop_worker(worker_task_id)
        if not success:
            return ResponseHelper.not_found(f"任务不存在或已停止: {worker_task_id}")

        # 停止成功后删除对应的数据库记录
        if _orm_helper:
            try:
                _orm_helper.delete_task_record(int(task_id), algorithm_id)
            except Exception as e:
                logger.error(f"删除任务记录异常 | task_id={task_id} | algorithm_id={algorithm_id} | error={e}")

        logger.info(f"任务已停止 | task_id={task_id} | algorithm_id={algorithm_id}")
        return ResponseHelper.success(message="任务已停止")

    except Exception as e:
        logger.error(f"停止任务异常: {e}")
        return ResponseHelper.error(f"停止任务异常: {str(e)}")


@router.post("/task/location")
async def save_task_location(request: LocationRequest):
    """保存任务经纬度定位信息（修正：使用 ORM 方法替代原始 SQL）"""
    try:
        if not _orm_helper:
            return ResponseHelper.error("数据库未初始化")

        _orm_helper.save_task_location(
            taskId=int(request.taskId),
            lng=request.longitude,
            lat=request.latitude,
            height=request.height,
            elevation=request.elevation,
            gimbalPitch=request.gimbalPitch,
            gimbalYaw=request.gimbalYaw,
            gimbalRoll=request.gimbalRoll,
            timestamp=datetime.datetime.utcnow(),
        )

        return ResponseHelper.success(message="定位信息保存成功")

    except Exception as e:
        logger.error(f"保存定位信息异常: {e}")
        return ResponseHelper.error(f"保存异常: {str(e)}")

@router.post("/transfer_detection_box")
async def transfer_detection_box_api(request: TransferDetectionBoxRequest):
    """
    在图片上绘制检测框的接口
    将识别框坐标从未识别图片转移到标注后的图片上
    """
    try:
        if not _minio_helper:
            return ResponseHelper.error("MinIO 未初始化")

        # 验证参数
        if not request.recognized_image_path or not isinstance(request.recognized_image_path, str):
            return ResponseHelper.bad_request("recognized_image_path 参数必须是有效的字符串")

        if not request.unrecognized_image_path or not isinstance(request.unrecognized_image_path, str):
            return ResponseHelper.bad_request("unrecognized_image_path 参数必须是有效的字符串")

        if not request.bounding_boxes or not isinstance(request.bounding_boxes, str):
            return ResponseHelper.bad_request("bounding_boxes 参数必须是有效的字符串")

        logger.info(f"transfer_detection_box | recognized_image_path={request.recognized_image_path} | bounding_boxes={request.bounding_boxes}")

        # 解析字符串格式的bounding_boxes参数
        try:
            parsed_bounding_boxes = json.loads(request.bounding_boxes)

            if isinstance(parsed_bounding_boxes, list) and len(parsed_bounding_boxes) > 0:
                # 单个边界框格式 [x1, y1, x2, y2]
                if len(parsed_bounding_boxes) == 4 and all(isinstance(coord, (int, float)) for coord in parsed_bounding_boxes):
                    bounding_boxes = [parsed_bounding_boxes]
                # 多个边界框格式 [[x1, y1, x2, y2], ...]
                elif all(isinstance(box, list) and len(box) == 4 and all(isinstance(coord, (int, float)) for coord in box) for box in parsed_bounding_boxes):
                    bounding_boxes = parsed_bounding_boxes
                else:
                    return ResponseHelper.bad_request("bounding_boxes 格式不正确，应为 [x1, y1, x2, y2] 或 [[x1, y1, x2, y2], ...] 的JSON字符串")
            else:
                return ResponseHelper.bad_request("bounding_boxes 参数解析后应为包含坐标列表的数组")
        except json.JSONDecodeError:
            return ResponseHelper.bad_request("bounding_boxes 参数必须是有效的JSON字符串")

        # 验证并清理 object_key，防止包含空字符
        if '\x00' in request.recognized_image_path:
            return ResponseHelper.bad_request("recognized_image_path 包含无效字符")
        if '\x00' in request.unrecognized_image_path:
            return ResponseHelper.bad_request("unrecognized_image_path 包含无效字符")

        # 移除控制字符
        clean_recognized_key = re.sub(r'[\x00-\x1f\x7f]', '', request.recognized_image_path)
        clean_unrecognized_key = re.sub(r'[\x00-\x1f\x7f]', '', request.unrecognized_image_path)

        # 从MinIO下载图片到临时文件
        recognized_local_path = _minio_helper.download_file(clean_recognized_key)
        unrecognized_local_path = _minio_helper.download_file(clean_unrecognized_key)

        try:
            # 读取图片
            recognized_img = cv2.imread(recognized_local_path)
            unrecognized_img = cv2.imread(unrecognized_local_path)

            if recognized_img is None:
                return ResponseHelper.bad_request(f"无法读取识别后图片: {clean_recognized_key}")
            if unrecognized_img is None:
                return ResponseHelper.bad_request(f"无法读取未识别图片: {clean_unrecognized_key}")

            # 检查两张图片尺寸是否相同
            if recognized_img.shape != unrecognized_img.shape:
                return ResponseHelper.bad_request("识别后图片和未识别图片尺寸不一致")

            # 在未识别图片上绘制检测框
            annotated_img = unrecognized_img.copy()

            if bounding_boxes:
                x1, y1, x2, y2 = bounding_boxes[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                height, width = annotated_img.shape[:2]
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))

                if x2 <= x1 or y2 <= y1:
                    return ResponseHelper.bad_request(f"无效的坐标范围: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

                # 绘制红色框
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # 生成输出文件名
            safe_key = clean_recognized_key.replace('/', '_').replace('\\', '_')
            base_name = os.path.splitext(os.path.basename(safe_key))[0] if '.' in safe_key else safe_key
            output_filename = f"{base_name}_transfer_annotated.jpg"

            # 编码图片并上传到MinIO
            _, buffer = cv2.imencode('.jpg', annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            image_bytes = buffer.tobytes()

            result_object_key = _minio_helper.upload_bytes(image_bytes, output_filename)

            logger.info(f"transfer_detection_box 完成 | result={result_object_key}")

            return ResponseHelper.success(
                data={"result_path": result_object_key},
                message="检测框转移完成"
            )

        finally:
            # 清理临时文件
            for local_path in [recognized_local_path, unrecognized_local_path]:
                if local_path and os.path.exists(local_path):
                    os.remove(local_path)

    except Exception as e:
        logger.error(f"transfer_detection_box 异常: {e}")
        return ResponseHelper.error(f"处理图片时发生错误: {str(e)}")
