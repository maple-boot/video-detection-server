import math
import time
from app.utils.logger import get_task_logger


class GeoUtils:
    """像素坐标转地理坐标"""

    def __init__(self, task_id: str = "system", cache_ttl: int = 2):
        self.logger = get_task_logger(task_id)
        self._cache = {}
        self._cache_ttl = cache_ttl

    def get_location(self, task_id, orm_helper) -> dict:
        """获取任务定位信息（带缓存），返回 dict

        修正说明：
        - 调用 ORMHelper.get_latest_task_location(int(task_id))
        - 将 ORM 实体字段名（lng/lat）映射为 API 使用的字段名（longitude/latitude）
        """
        cache_key = f"location_{task_id}"
        now = time.time()

        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if now - cached_time < self._cache_ttl:
                return cached_data

        location_entity = orm_helper.get_latest_task_location(int(task_id))
        if location_entity:
            location = {
                "longitude": location_entity.lng,
                "latitude": location_entity.lat,
                "height": location_entity.height,
                "elevation": location_entity.elevation,
                "gimbalPitch": location_entity.gimbal_pitch,
                "gimbalYaw": location_entity.gimbal_yaw,
                "gimbalRoll": location_entity.gimbal_roll,
            }
            self._cache[cache_key] = (now, location)
            return location
        return None

    @staticmethod
    def pixel_to_geo(pixel_x: int, pixel_y: int,
                     img_width: int, img_height: int,
                     longitude: float, latitude: float,
                     height: float, gimbal_pitch: float,
                     gimbal_yaw: float, gimbal_roll: float,
                     hfov: float = 72.0) -> tuple:
        """像素坐标转换为地理坐标（经纬度）"""
        dx = (pixel_x - img_width / 2) / img_width
        dy = (pixel_y - img_height / 2) / img_height

        vfov = hfov * img_height / img_width

        yaw_rad = math.radians(gimbal_yaw)
        pitch_rad = math.radians(gimbal_pitch)

        angle_h = dx * hfov
        angle_v = dy * vfov

        ground_distance = height * math.tan(math.radians(abs(gimbal_pitch) + angle_v))

        lat_offset = ground_distance * math.cos(yaw_rad + math.radians(angle_h)) / 111320
        lon_offset = ground_distance * math.sin(yaw_rad + math.radians(angle_h)) / (
                111320 * math.cos(math.radians(latitude))
        )

        target_longitude = longitude + lon_offset
        target_latitude = latitude + lat_offset

        return target_longitude, target_latitude
