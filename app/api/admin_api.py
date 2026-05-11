from fastapi import APIRouter
from app.utils.response_helper import ResponseHelper
from app.managers.resource_monitor import ResourceMonitor
from app.utils.logger import get_system_logger

logger = get_system_logger()

router = APIRouter()

# 全局引用
_supervisor = None
_cleanup_scheduler = None
_thread_pool = None
_resource_monitor = None
_start_time = None


def init_admin_api(supervisor, cleanup_scheduler, thread_pool, resource_monitor, start_time):
    global _supervisor, _cleanup_scheduler, _thread_pool, _resource_monitor, _start_time
    _supervisor = supervisor
    _cleanup_scheduler = cleanup_scheduler
    _thread_pool = thread_pool
    _resource_monitor = resource_monitor
    _start_time = start_time


@router.get("/admin/status")
async def get_status():
    """查看当前资源状态"""
    try:
        import time
        import psutil

        status = ResourceMonitor.get_status()

        # 补充任务信息
        workers_info = _supervisor.get_all_workers() if _supervisor else {}
        running_count = sum(1 for w in workers_info.values() if w.get("state") == "running")
        retrying_count = sum(1 for w in workers_info.values() if w.get("state") == "retrying")
        stopped_count = sum(1 for w in workers_info.values() if w.get("state") in ("stopped", "completed"))

        status["tasks"] = {
            "running": running_count,
            "retrying": retrying_count,
            "stopped": stopped_count,
            "total": len(workers_info),
        }
        status["workers"] = workers_info
        status["uptime_seconds"] = int(time.time() - _start_time) if _start_time else 0

        return ResponseHelper.success(data=status)

    except Exception as e:
        logger.error(f"获取状态异常: {e}")
        return ResponseHelper.error(f"获取状态异常: {str(e)}")


@router.post("/admin/cleanup")
async def execute_cleanup():
    """执行全部线程池清理"""
    try:
        if not _cleanup_scheduler:
            return ResponseHelper.error("清理调度器未初始化")

        report = _cleanup_scheduler.execute_cleanup(reason="manual_api")
        return ResponseHelper.success(data=report, message="清理完成")

    except Exception as e:
        logger.error(f"清理异常: {e}")
        return ResponseHelper.error(f"清理异常: {str(e)}")


@router.post("/admin/restart")
async def execute_restart():
    """执行优雅服务重启"""
    try:
        logger.info("收到服务重启请求")

        # 先执行清理
        if _cleanup_scheduler:
            _cleanup_scheduler.execute_cleanup(reason="restart_api")

        # 停止所有 Worker
        if _supervisor:
            _supervisor.stop_all()

        # 关闭线程池
        if _thread_pool:
            _thread_pool.shutdown(wait=True)

        logger.info("服务资源已清理，准备重启...")

        # 在实际部署中，这里应该触发进程重启
        # 例如通过 systemd 或 supervisorctl
        # 这里返回提示让运维手动重启
        return ResponseHelper.success(
            message="资源已清理，请重启服务进程以完成重启",
            data={"pid": __import__("os").getpid()}
        )

    except Exception as e:
        logger.error(f"重启异常: {e}")
        return ResponseHelper.error(f"重启异常: {str(e)}")
