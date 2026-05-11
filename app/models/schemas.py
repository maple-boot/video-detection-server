from pydantic import BaseModel
from typing import List, Optional

class StreamRequest(BaseModel):
    """启动视频流检测任务请求"""
    taskId: str
    algorithmId: str
    streamUrl: str
    platformId: Optional[str] = "nbuav"
    minioName: Optional[str] = "nbuav"


class StreamUrlRequest(BaseModel):
    """查询推流地址请求"""
    taskId: str
    algorithmId: str

class StopTaskRequest(BaseModel):
    taskId: str
    algorithmId: str


class LocationRequest(BaseModel):
    """保存任务经纬度定位信息请求"""
    taskId: str
    longitude: float
    latitude: float
    height: float
    elevation: Optional[float] = 0.0
    gimbalPitch: Optional[float] = 0.0
    gimbalYaw: Optional[float] = 0.0
    gimbalRoll: Optional[float] = 0.0


class DetectionResult(BaseModel):
    """检测结果"""
    bbox: List[float]
    confidence: float
    classId: int
    className: str
    longitude: Optional[float] = None
    latitude: Optional[float] = None


class ReportPayload(BaseModel):
    """HTTP 回调上报数据"""
    taskId: str
    algorithmId: str
    frameId: int
    detections: List[DetectionResult]
    imageUrl: Optional[str] = ""
    annotatedUrl: Optional[str] = ""
    minioObject: Optional[str] = ""
    timestamp: float
