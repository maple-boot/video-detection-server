import datetime
import time
from fastapi import APIRouter
from app.models.schemas import StreamRequest, StreamUrlRequest, LocationRequest, StopTaskRequest
from app.utils.response_helper import ResponseHelper
from app.utils.logger import get_system_logger

logger = get_system_logger()

router = APIRouter()

# 全局引用（在 main.py 中注入）
_supervisor = None
_orm_helper = None


def init_stream_api(supervisor, orm_helper):
    global _supervisor, _orm_helper
    _supervisor = supervisor
    _orm_helper = orm_helper


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
                    push_url = f"rtmp://112.14.53.185/live/stream/{task_id}_{alg_id}"
                    return ResponseHelper.success(
                        data={"output_url": push_url.replace('rtmp://', 'webrtc://'), "taskId": task_id}
                    )

        # 协议替换
        if "webrtc" in stream_url:
            stream_url = stream_url.replace("webrtc", "rtmp")

        # 判断流类型
        is_file = stream_url.endswith(".mp4") or stream_url.endswith(".avi") or stream_url.endswith(".mkv")

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
                push_url = f"rtmp://112.14.53.185/live/stream/{worker_task_id}"
                return ResponseHelper.success(data={"output_url": push_url.replace('rtmp://', 'webrtc://'), "taskId": task_id})
            else:
                return ResponseHelper.error("任务启动失败")
        else:
            # 每个算法独立 Worker + 独立推流地址
            results = []
            for alg_id in algorithm_ids:
                worker_task_id = f"{task_id}_{alg_id}"
                push_url = f"rtmp://112.14.53.185/live/stream/{worker_task_id}"

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
                    push_url = f"rtmp://112.14.53.185/live/stream/{worker_task_id}"
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
