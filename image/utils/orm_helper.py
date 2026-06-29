"""
基于SQLAlchemy的ORM工具，提供数据库连接、模型基类和常用操作方法。
"""
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, UniqueConstraint, Index, text, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError
import os
import datetime
import logging
import yaml

logger = logging.getLogger(__name__)

# 读取配置文件
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')

def load_db_config(db_config_name='orm'):
    """根据配置名称加载数据库配置"""
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    db_cfg = config.get(db_config_name, {})
    db_type = db_cfg.get('db_type', 'mysql')
    if db_type == 'mysql':
        user = db_cfg.get('username', 'root')
        pwd = db_cfg.get('password', '')
        # URL编码密码中的特殊字符
        pwd = quote_plus(str(pwd))
        host = db_cfg.get('host', '127.0.0.1')
        port = db_cfg.get('port', 3306)
        db = db_cfg.get('database', '')
        return f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}?charset=utf8mb4"
    elif db_type == 'sqlite':
        return db_cfg.get('database', 'sqlite:///./test.db')
    else:
        raise ValueError('不支持的数据库类型')

def quote_plus(s):
    """
    自定义 URL 编码函数，类似 urllib.parse.quote_plus，将特殊字符和空格编码为适合URL的形式
    空格会被编码为 + 号，其他特殊字符被编码为 %XX 的形式
    """
    s = str(s)
    # 特殊字符映射，注意 + 号需要特殊处理
    unsafe_chars = {
        ' ': '+',      # 空格转换为 +
        '!': '%21',    '"': '%22',  '#': '%23',  '$': '%24',  '&': '%26',
        "'": '%27',    '(': '%28',  ')': '%29',  '*': '%2A',
        ',': '%2C',    '/': '%2F',  ':': '%3A',  ';': '%3B',  '<': '%3C',
        '=': '%3D',    '>': '%3E',  '?': '%3F',  '@': '%40',  '[': '%5B',
        ']': '%5D',    '{': '%7B',  '}': '%7D'
    }
    
    result = ""
    for char in s:
        if char in unsafe_chars:
            result += unsafe_chars[char]
        elif ord(char) < 32 or ord(char) == 127:  # 控制字符
            result += f'%{ord(char):02X}'
        else:
            result += char
    return result

# 主数据库配置
DATABASE_URL = load_db_config('orm')
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))



Base = declarative_base()

# 线程安全的Session获取
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



# AI算法识别结果
class AlRecognitionRecord(Base):
    __tablename__ = 'algorithm_recognition_record'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_url = Column(String(255), nullable=False)
    device_lng = Column(Float, nullable=False)
    device_lat = Column(Float, nullable=False)
    device_altitude = Column(Float, nullable=False)
    device_height = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.datetime.now)
    task_out_id = Column(Integer, nullable=False)
    task_out_bid = Column(Integer, nullable=False)
    is_del = Column(Integer, nullable=False)

# AI算法识别结果分类
class AlRecordClass(Base):
    __tablename__ = 'algorithm_record_class'
    id = Column(Integer, primary_key=True, autoincrement=True)
    algorithm_record_id = Column(Integer, nullable=False)
    class_name = Column(String(255), nullable=False)
    objects = Column(String(255), nullable=False)
    is_del = Column(Integer, nullable=False)


# AI图片截取记录表
class AlRecordImageSplit(Base):
    __tablename__ = 'algorithm_record_image_split'
    id = Column(Integer, primary_key=True, autoincrement=True)
    algorithm_record_id = Column(Integer, nullable=True, comment='记录的ID')
    image_url = Column(String(255), nullable=True, comment='截取后的图片地址')
    create_time = Column(DateTime, nullable=True)
    update_time = Column(DateTime, nullable=True)
    is_del = Column(Integer, default=1, comment='软删除标识')


class ORMHelper:
    @staticmethod
    def get_algorithm_image_list(task_out_bid: int):
        """根据任务输出ID获取算法识别结果列表"""
        session = SessionLocal()
        try:
            return (
                session.query(AlRecognitionRecord)
                .filter_by(task_out_bid=task_out_bid)
                .all()
            )
        except SQLAlchemyError as e:
            raise e
        finally:
            session.close()

    @staticmethod
    def get_algorithm_image_class_list(algorithm_record_id: int):
        """根据算法识别结果ID获取算法识别结果分类列表"""
        session = SessionLocal()
        try:
            return (
                session.query(AlRecordClass)
                .filter_by(algorithm_record_id=algorithm_record_id)
                .filter(AlRecordClass.objects.isnot(None))  # 添加objects不为null的条件
                .filter(AlRecordClass.objects != '')  # 同时排除空字符串
                .all()
            )
        except SQLAlchemyError as e:
            raise e
        finally:
            session.close()

    @staticmethod
    def insert_algorithm_image_split(algorithm_record_id: int, image_url: str):
        """插入AI图片截取记录"""
        session = SessionLocal()
        try:
            # 创建新记录
            record = AlRecordImageSplit(
                algorithm_record_id=algorithm_record_id,
                image_url=image_url,
                create_time=datetime.datetime.now(),
                update_time=datetime.datetime.now(),
                is_del=1  # 默认值
            )
            session.add(record)
            session.commit()
            session.refresh(record)  # 刷新以获取自动生成的ID
            return record
        except SQLAlchemyError as e:
            session.rollback()
            raise e
        finally:
            session.close()