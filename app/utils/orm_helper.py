"""
基于SQLAlchemy的ORM工具，提供数据库连接、模型基类和常用操作方法。
"""

import datetime

from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Float,
    UniqueConstraint, Index, text, BigInteger
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError

from app.utils.logger import get_system_logger

logger = get_system_logger()


# ============================================================
# URL 编码工具
# ============================================================

def quote_plus(s):
    """
    自定义 URL 编码函数，类似 urllib.parse.quote_plus，
    将特殊字符和空格编码为适合URL的形式。
    空格会被编码为 + 号，其他特殊字符被编码为 %XX 的形式。
    """
    s = str(s)
    unsafe_chars = {
        ' ': '+',
        '!': '%21', '"': '%22', '#': '%23', '$': '%24', '&': '%26',
        "'": '%27', '(': '%28', ')': '%29', '*': '%2A',
        ',': '%2C', '/': '%2F', ':': '%3A', ';': '%3B', '<': '%3C',
        '=': '%3D', '>': '%3E', '?': '%3F', '@': '%40', '[': '%5B',
        ']': '%5D', '{': '%7B', '}': '%7D'
    }

    result = ""
    for char in s:
        if char in unsafe_chars:
            result += unsafe_chars[char]
        elif ord(char) < 32 or ord(char) == 127:
            result += f'%{ord(char):02X}'
        else:
            result += char
    return result


# ============================================================
# ORM 模型定义
# ============================================================

Base = declarative_base()


class ModelInfo(Base):
    """模型信息表"""
    __tablename__ = 'ai_model_info'
    id = Column(Integer, primary_key=True, autoincrement=True)
    algorithm_id = Column(String(32), nullable=False, unique=True, index=True)
    model_name = Column(String(255), nullable=False)
    model_path = Column(String(255), nullable=False)
    cls_path = Column(String(255), nullable=False)
    inference_size = Column(Integer, nullable=False)


class TaskRecord(Base):
    """任务记录表"""
    __tablename__ = 'ai_task_record'
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, nullable=False, index=True)
    algorithm_id = Column(String(255), nullable=False)
    input_url = Column(String(512), nullable=False)
    output_url = Column(String(512), nullable=True)
    status = Column(String(32), nullable=False, default='processing')
    create_time = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    update_time = Column(DateTime, nullable=False, default=datetime.datetime.utcnow,
                         onupdate=datetime.datetime.utcnow)
    remark = Column(String(256), nullable=True)
    is_deleted = Column(Integer, nullable=False, default=0)

    __table_args__ = (
        UniqueConstraint('task_id', 'algorithm_id', name='uq_ai_task_record_task_algo'),
    )


class TaskLocation(Base):
    """任务经纬度信息表"""
    __tablename__ = 'ai_task_location'
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, nullable=True, index=True)
    lng = Column(Float, nullable=False)
    lat = Column(Float, nullable=False)
    height = Column(Float, nullable=False)
    elevation = Column(Float, nullable=False)
    gimbal_pitch = Column(Float, nullable=False)
    gimbal_yaw = Column(Float, nullable=False)
    gimbal_roll = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    create_time = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    update_time = Column(DateTime, nullable=False, default=datetime.datetime.utcnow,
                         onupdate=datetime.datetime.utcnow)
    is_deleted = Column(Integer, nullable=False, default=0)


class AlgorithmRecognitionRecord(Base):
    """算法识别结果表"""
    __tablename__ = 'algorithm_recognition_record'
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_out_bid = Column(BigInteger, index=True)
    thumb_url = Column(String(255))
    image_url = Column(String(255))


class PlatformCallback(Base):
    """平台回调地址关系表"""
    __tablename__ = 'platform_callback_url_map'
    id = Column(Integer, primary_key=True, autoincrement=True)
    platform_id = Column(String(255), nullable=False, index=True)
    callback_url = Column(String(255), nullable=False)
    minio_config = Column(String(255), nullable=False)


# ============================================================
# ORMHelper — 数据库操作封装
# ============================================================

class ORMHelper:
    """数据库操作封装，适配系统架构的统一入口"""

    def __init__(self, config: dict):
        """
        初始化数据库连接。
        config: 系统 config.yaml 解析后的字典。
        """
        orm_cfg = config.get('orm', {})
        db_type = orm_cfg.get('db_type', 'sqlite')

        if db_type == 'mysql':
            user = orm_cfg.get('username', orm_cfg.get('user', 'root'))
            pwd = orm_cfg.get('password', '')
            pwd = quote_plus(str(pwd))
            host = orm_cfg.get('host', '127.0.0.1')
            port = orm_cfg.get('port', 3306)
            db = orm_cfg.get('database', '')
            database_url = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}?charset=utf8mb4"
        elif db_type == 'sqlite':
            database_url = orm_cfg.get('database', 'sqlite:///./test.db')
        else:
            raise ValueError(f'不支持的数据库类型: {db_type}')

        self.engine = create_engine(database_url, echo=False, pool_pre_ping=True)
        self.SessionLocal = scoped_session(
            sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        )

        # 确保表和索引就绪
        self.ensure_tables()
        self.ensure_indexes()

        logger.info(f"数据库连接初始化完成 | db_type={db_type} | host={orm_cfg.get('host', 'N/A')}")

    # ----------------------------------------------------------
    # 表与索引管理
    # ----------------------------------------------------------

    def ensure_tables(self):
        """确保所有ORM模型对应的数据表已创建。"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("已确保ORM相关数据表存在")
        except Exception as e:
            logger.error(f"创建数据表失败: {e}")

    def ensure_indexes(self):
        """在应用启动时校验并修复 ai_task_record 的唯一索引结构。
        目标：删除仅 task_id 的唯一索引，添加 (task_id, algorithm_id) 复合唯一索引。
        仅在 MySQL 上执行，其他数据库跳过。
        """
        try:
            backend = self.engine.url.get_backend_name()
            if backend != 'mysql':
                logger.info(f"跳过索引检查：当前数据库后端为 {backend}")
                return

            with self.engine.begin() as conn:
                # 查找仅包含 task_id 的唯一索引
                find_single_unique_sql = text(
                    """
                    SELECT s.INDEX_NAME
                    FROM information_schema.statistics s
                    WHERE s.TABLE_SCHEMA = DATABASE()
                      AND s.TABLE_NAME = 'ai_task_record'
                      AND s.NON_UNIQUE = 0
                    GROUP BY s.INDEX_NAME
                    HAVING SUM(CASE WHEN s.COLUMN_NAME='task_id' THEN 1 ELSE 0 END) = COUNT(*)
                       AND COUNT(*) = 1
                    """
                )
                idx_rows = conn.execute(find_single_unique_sql).fetchall()
                for (idx_name,) in idx_rows:
                    try:
                        conn.execute(text(f"ALTER TABLE ai_task_record DROP INDEX `{idx_name}`"))
                        logger.info(f"已删除仅 task_id 的唯一索引: {idx_name}")
                    except Exception as drop_e:
                        logger.warning(f"删除索引 {idx_name} 失败: {drop_e}")

                # 确认复合唯一索引存在
                exists_sql = text(
                    """
                    SELECT COUNT(*) FROM information_schema.statistics
                    WHERE TABLE_SCHEMA = DATABASE()
                      AND TABLE_NAME = 'ai_task_record'
                      AND INDEX_NAME = 'uq_ai_task_record_task_algo'
                    """
                )
                exists = conn.execute(exists_sql).scalar() or 0
                if int(exists) == 0:
                    conn.execute(text(
                        "ALTER TABLE ai_task_record ADD UNIQUE INDEX "
                        "`uq_ai_task_record_task_algo` (`task_id`,`algorithm_id`)"
                    ))
                    logger.info("已创建 (task_id, algorithm_id) 复合唯一索引")
                else:
                    logger.info("已存在复合唯一索引: uq_ai_task_record_task_algo")

        except Exception as e:
            logger.error(f"校验/修复索引结构失败: {e}")

    # ----------------------------------------------------------
    # 会话管理
    # ----------------------------------------------------------

    def get_session(self):
        """获取线程安全的数据库会话"""
        return self.SessionLocal()

    def close(self):
        """关闭数据库连接池"""
        try:
            self.SessionLocal.remove()
            self.engine.dispose()
            logger.info("数据库连接池已关闭")
        except Exception as e:
            logger.error(f"关闭数据库连接池异常: {e}")

    # ----------------------------------------------------------
    # 模型信息
    # ----------------------------------------------------------

    def get_model_info(self, algorithm_id: str):
        """根据 algorithm_id 查询模型信息。
        返回 ModelInfo 实体或 None。
        """
        session = self.get_session()
        try:
            return session.query(ModelInfo).filter_by(algorithm_id=algorithm_id).first()
        except SQLAlchemyError as e:
            logger.error(f"查询模型信息失败 | algorithm_id={algorithm_id} | error={e}")
            raise e
        finally:
            session.close()

    # ----------------------------------------------------------
    # 任务记录
    # ----------------------------------------------------------

    def get_task_record(self, task_id: int, algorithm_id: str):
        """根据 task_id 与 algorithm_id 查询任务记录，按创建时间倒序取最新一条。
        返回 TaskRecord 实体或 None。
        """
        session = self.get_session()
        try:
            return (
                session.query(TaskRecord)
                .filter_by(task_id=task_id, algorithm_id=algorithm_id)
                .order_by(TaskRecord.create_time.desc())
                .first()
            )
        except SQLAlchemyError as e:
            logger.error(f"查询任务记录失败 | task_id={task_id} | algorithm_id={algorithm_id} | error={e}")
            raise e
        finally:
            session.close()

    def add_task_record(self, task_id, algorithm_id, input_url,
                        output_url=None, status='processing', remark=None):
        """添加任务记录。若已存在相同 task_id 和 algorithm_id 的记录则直接返回现有记录。"""
        session = self.get_session()
        try:
            existing_record = session.query(TaskRecord).filter_by(
                task_id=task_id, algorithm_id=algorithm_id
            ).first()
            if existing_record:
                return existing_record

            record = TaskRecord(
                task_id=task_id,
                algorithm_id=algorithm_id,
                input_url=input_url,
                output_url=output_url,
                status=status,
                remark=remark,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"添加任务记录失败 | task_id={task_id} | error={e}")
            raise e
        finally:
            session.close()

    def update_task_status(self, task_id, algorithm_id, status, output_url=None, remark=None):
        """更新任务状态。按创建时间倒序取最新一条记录进行更新。"""
        session = self.get_session()
        try:
            record = (
                session.query(TaskRecord)
                .filter_by(task_id=task_id, algorithm_id=algorithm_id)
                .order_by(TaskRecord.create_time.desc())
                .first()
            )
            if record:
                record.status = status
                if output_url:
                    record.output_url = output_url
                if remark:
                    record.remark = remark
                session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"更新任务状态失败 | task_id={task_id} | error={e}")
            raise e
        finally:
            session.close()

    # ----------------------------------------------------------
    # 任务经纬度
    # ----------------------------------------------------------

    def save_task_location(self, taskId, lng, lat, height, elevation,
                           gimbalPitch, gimbalYaw, gimbalRoll, timestamp):
        """保存任务经纬度定位信息"""
        session = self.get_session()
        try:
            location = TaskLocation(
                task_id=taskId,
                lng=lng,
                lat=lat,
                height=height,
                elevation=elevation,
                gimbal_pitch=gimbalPitch,
                gimbal_yaw=gimbalYaw,
                gimbal_roll=gimbalRoll,
                timestamp=timestamp,
            )
            session.add(location)
            session.commit()
            session.refresh(location)
            return location
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"保存任务经纬度失败 | task_id={taskId} | error={e}")
            raise e
        finally:
            session.close()

    def get_latest_task_location(self, task_id: int):
        """获取最新的无人机飞行坐标信息"""
        session = self.get_session()
        try:
            return (
                session.query(TaskLocation)
                .filter_by(task_id=task_id)
                .order_by(TaskLocation.timestamp.desc())
                .first()
            )
        except SQLAlchemyError as e:
            logger.error(f"获取最新坐标失败 | task_id={task_id} | error={e}")
            raise e
        finally:
            session.close()

    # ----------------------------------------------------------
    # 算法识别结果记录
    # ----------------------------------------------------------

    def update_image_record(self, task_out_bid: int, thumb_url: str, image_url: str):
        """更新算法识别记录表中的缩略图URL和图片URL；若记录不存在则跳过。"""
        session = self.get_session()
        try:
            record = (
                session.query(AlgorithmRecognitionRecord)
                .filter_by(task_out_bid=task_out_bid)
                .filter(AlgorithmRecognitionRecord.image_url.like(f"%{thumb_url}"))
                .first()
            )
            if record:
                record.thumb_url = image_url
                record.image_url = image_url
                session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"更新图片记录失败 | task_out_bid={task_out_bid} | error={e}")
            raise e
        finally:
            session.close()

    # ----------------------------------------------------------
    # 平台回调地址
    # ----------------------------------------------------------

    def get_platform_callback_url(self, platform_id: str):
        """获取平台回调地址。
        返回 PlatformCallback 实体或 None。
        """
        session = self.get_session()
        try:
            return (
                session.query(PlatformCallback)
                .filter_by(platform_id=platform_id)
                .first()
            )
        except SQLAlchemyError as e:
            logger.error(f"获取平台回调地址失败 | platform_id={platform_id} | error={e}")
            raise e
        finally:
            session.close()
