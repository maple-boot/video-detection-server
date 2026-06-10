import os
import time
import yaml
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.utils.logger import setup_logger, get_system_logger
from app.utils.orm_helper import ORMHelper
from app.utils.minio_helper import MinioHelper
from app.core.inference_engine import InferenceEngine
from app.core.stream_supervisor import StreamSupervisor
from app.managers.thread_pool import VideoThreadPool
from app.managers.cleanup_scheduler import CleanupScheduler
from app.managers.resource_monitor import ResourceMonitor
from app.api import stream_api, admin_api


def load_config(config_path: str = "config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# 全局变量
config = None
orm_helper = None
minio_helper = None
inference_engine = None
supervisor = None
thread_pool = None
cleanup_scheduler = None
resource_monitor = None
start_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global config, orm_helper, minio_helper, inference_engine
    global supervisor, thread_pool, cleanup_scheduler, resource_monitor, start_time

    # ===== 启动阶段 =====
    start_time = time.time()

    # 加载配置
    config = load_config()

    # 初始化日志
    setup_logger(config)
    logger = get_system_logger()
    logger.info("=" * 60)
    logger.info("YOLO 视频流检测服务启动中...")
    logger.info("=" * 60)

    # 初始化数据库
    try:
        orm_helper = ORMHelper(config)
        logger.info("数据库初始化完成")
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        orm_helper = None

    # 初始化 MinIO
    try:
        minio_helper = MinioHelper(config)
        logger.info("MinIO 初始化完成")
    except Exception as e:
        logger.error(f"MinIO 初始化失败: {e}")
        minio_helper = None

    # 初始化推理引擎
    inference_engine = InferenceEngine(config)
    logger.info("推理引擎初始化完成")

    # 初始化 Supervisor
    supervisor = StreamSupervisor(
        config=config,
        orm_helper=orm_helper,
        minio_helper=minio_helper,
        inference_engine=inference_engine,
    )
    logger.info("StreamSupervisor 初始化完成")

    # 初始化线程池
    thread_pool = VideoThreadPool(max_workers=50)
    logger.info("线程池初始化完成")

    # 初始化清理调度器
    cleanup_scheduler = CleanupScheduler(
        config=config,
        thread_pool=thread_pool,
        inference_engine=inference_engine,
        task_registry=supervisor.get_task_registry(),
    )
    cleanup_scheduler.start()
    logger.info("清理调度器初始化完成")

    # 初始化资源监控
    resource_monitor = ResourceMonitor(
        config=config,
        cleanup_callback=lambda reason="", status=None: cleanup_scheduler.execute_cleanup(reason=reason, status=status),
    )
    resource_monitor.start()
    logger.info("资源监控初始化完成")

    # 初始化 API 模块
    stream_api.init_stream_api(supervisor, orm_helper, config)
    admin_api.init_admin_api(supervisor, cleanup_scheduler, thread_pool, resource_monitor, start_time)

    logger.info("=" * 60)
    logger.info("YOLO 视频流检测服务启动完成")
    logger.info(f"PID: {os.getpid()}")
    logger.info("=" * 60)

    yield  # 服务运行中

    # ===== 关闭阶段 =====
    logger.info("=" * 60)
    logger.info("YOLO 视频流检测服务关闭中...")
    logger.info("=" * 60)

    # 停止资源监控
    if resource_monitor:
        resource_monitor.stop()

    # 停止清理调度器
    if cleanup_scheduler:
        cleanup_scheduler.stop()

    # 停止所有 Worker
    if supervisor:
        supervisor.stop_all()

    # 关闭线程池
    if thread_pool:
        thread_pool.shutdown(wait=True)

    # 清空推理引擎
    if inference_engine:
        inference_engine.clear_cache()

    # 关闭数据库
    if orm_helper:
        orm_helper.close()

    logger.info("YOLO 视频流检测服务已关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title="YOLO 视频流检测服务",
    description="基于 YOLO 的实时视频流目标检测系统",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(stream_api.router, tags=["业务接口"])
app.include_router(admin_api.router, tags=["运维管理"])


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "pid": os.getpid(), "uptime": int(time.time() - start_time) if start_time else 0}


if __name__ == "__main__":
    config = load_config()
    server_config = config.get("server", {})
    uvicorn.run(
        "main:app",
        host=server_config.get("host", "0.0.0.0"),
        port=server_config.get("port", 6950),
        reload=False,
        workers=1,
    )
