import sys
import os
from loguru import logger
from pathlib import Path


def setup_logger(config: dict):
    """初始化全局日志配置"""
    log_config = config.get("logging", {})
    log_dir = log_config.get("log_dir", "logs")
    log_level = log_config.get("log_level", "INFO")
    rotation = log_config.get("rotation", "100 MB")
    retention = log_config.get("retention", "7 days")

    # 创建日志目录
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # 移除默认处理器
    logger.remove()

    # 控制台输出
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{extra[task_id]}</cyan> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        filter=lambda record: record["extra"].get("task_id", "system") == "system"
                              or True,
    )

    # 全局日志文件
    logger.add(
        os.path.join(log_dir, "app.log"),
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[task_id]} | {name}:{function}:{line} | {message}",
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
    )

    # 错误日志文件
    logger.add(
        os.path.join(log_dir, "error.log"),
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[task_id]} | {name}:{function}:{line} | {message}",
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
    )

    # 性能日志文件
    logger.add(
        os.path.join(log_dir, "performance.log"),
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {extra[task_id]} | {message}",
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        filter=lambda record: record["extra"].get("log_type") == "performance",
    )

    # 清理日志文件
    logger.add(
        os.path.join(log_dir, "cleanup.log"),
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        filter=lambda record: record["extra"].get("log_type") == "cleanup",
    )

    # 显存监控日志文件
    logger.add(
        os.path.join(log_dir, "memory.log"),
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {extra[task_id]} | {message}",
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        filter=lambda record: record["extra"].get("log_type") == "memory",
    )


def get_task_logger(task_id: str, algorithm_id: str = ""):
    """获取任务专属日志器，绑定 task_id 上下文"""
    tag = f"task_{task_id}" if not algorithm_id else f"task_{task_id}_alg_{algorithm_id}"
    return logger.bind(task_id=tag)


def get_system_logger():
    """获取系统级日志器"""
    return logger.bind(task_id="system")


def get_cleanup_logger():
    """获取清理专用日志器"""
    return logger.bind(task_id="system", log_type="cleanup")


def get_memory_logger():
    """获取显存监控专用日志器"""
    return logger.bind(task_id="system", log_type="memory")


def get_performance_logger(task_id: str):
    """获取性能日志器"""
    tag = f"task_{task_id}"
    return logger.bind(task_id=tag, log_type="performance")


def add_task_log_file(log_dir: str, task_id: str, algorithm_id: str = ""):
    """为特定任务添加独立日志文件"""
    tag = f"task_{task_id}" if not algorithm_id else f"task_{task_id}_alg_{algorithm_id}"
    log_file = os.path.join(log_dir, f"stream_{tag}.log")

    logger.add(
        log_file,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="50 MB",
        retention="3 days",
        encoding="utf-8",
        filter=lambda record, tid=tag: record["extra"].get("task_id") == tid,
    )
