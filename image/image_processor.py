import cv2
import os
import numpy as np
from utils.minio_helper import MinioHelper


class ImageProcessor:
    """图片处理工具类，用于裁剪和放大图片中的指定区域"""
    
    def __init__(self):
        """
        初始化图片处理器
        """
        self.minio_helper = MinioHelper()
    
    def crop_and_zoom(self, image_path, coordinates, zoom_factor=1.0, output_filename=None):
        """
        根据坐标裁剪并放大图片区域
        :param image_path: 原始图片路径
        :param coordinates: 坐标列表 [x1, y1, x2, y2]
        :param zoom_factor: 放大倍数，默认为1.0（不放大）
        :param output_filename: 输出文件名，默认为 None，会自动生成
        :return: 保存在MinIO中的文件路径
        """
        # 读取原始图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        # 解析坐标
        x1, y1, x2, y2 = coordinates
        
        # 确保坐标值是整数
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 获取图片尺寸
        height, width = image.shape[:2]
        
        # 确保坐标在图片范围内
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        # 确保 x2 > x1 且 y2 > y1
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"无效的坐标范围: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        
        # 裁剪图片
        cropped_image = image[y1:y2, x1:x2]
        
        # 放大裁剪后的图片
        if zoom_factor != 1.0:
            height, width = cropped_image.shape[:2]
            new_width = int(width * zoom_factor)
            new_height = int(height * zoom_factor)
            
            # 如果是放大，使用更适合的插值方法来保持清晰度
            if zoom_factor > 1.0:
                # 对于放大，使用LANCZOS4插值算法，能更好地保持清晰度
                cropped_image = cv2.resize(cropped_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            else:
                # 对于缩小，使用AREA插值
                cropped_image = cv2.resize(cropped_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # 生成输出文件名
        if output_filename is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_filename = f"{base_name}_cropped_zoomed_{zoom_factor}x.jpg"
        
        # 将图片转换为字节流
        # 使用更高的JPEG质量来保持清晰度
        _, buffer = cv2.imencode('.jpg', cropped_image, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
        image_bytes = buffer.tobytes()

        # 上传到MinIO
        return self.minio_helper.upload_bytes(image_bytes, output_filename)


    def transfer_detection_box(self, recognized_image_object_key, unrecognized_image_object_key,
                               bounding_boxes):
        """
        将识别框从识别后图片转移到未识别的同帧图片上（直接使用边界框坐标）

        :param recognized_image_object_key: 识别后图片在MinIO中的object key
        :param unrecognized_image_object_key: 未识别的同帧图片在MinIO中的object key
        :param bounding_boxes: 边界框坐标列表 [[x1, y1, x2, y2], ...]，每次调用通常包含一个边界框
        :return: 保存在MinIO中的标注后图片的object key
        """
        # 验证并清理 object_key，防止包含空字符
        if '\x00' in recognized_image_object_key:
            raise ValueError("recognized_image_object_key 包含无效的空字符")
        if '\x00' in unrecognized_image_object_key:
            raise ValueError("unrecognized_image_object_key 包含无效的空字符")

        # 验证和清理 object_key 以确保安全
        import re
        # 移除控制字符
        clean_recognized_key = re.sub(r'[\x00-\x1f\x7f]', '', recognized_image_object_key)
        clean_unrecognized_key = re.sub(r'[\x00-\x1f\x7f]', '', unrecognized_image_object_key)

        # 从MinIO下载图片
        recognized_local_path = self.minio_helper.download_file(clean_recognized_key)
        unrecognized_local_path = self.minio_helper.download_file(clean_unrecognized_key)

        # 读取图片
        recognized_img = cv2.imread(recognized_local_path)
        unrecognized_img = cv2.imread(unrecognized_local_path)

        if recognized_img is None:
            raise ValueError(f"无法读取识别后图片: {recognized_local_path}")
        if unrecognized_img is None:
            raise ValueError(f"无法读取未识别图片: {unrecognized_local_path}")

        # 检查两张图片尺寸是否相同
        if recognized_img.shape != unrecognized_img.shape:
            raise ValueError("识别后图片和未识别图片尺寸不一致")

        # 直接使用传入的边界框坐标进行标注
        annotated_img = unrecognized_img.copy()

        # 由于每次调用仅包含一个边界框，直接使用第一个边界框
        if bounding_boxes:
            x1, y1, x2, y2 = bounding_boxes[0]

            # 确保坐标值是整数
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 获取图片尺寸
            height, width = annotated_img.shape[:2]

            # 确保坐标在图片范围内
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            # 确保 x2 > x1 且 y2 > y1
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"无效的坐标范围: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

            # 绘制红色框
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 生成输出文件名
        safe_recognized_key = clean_recognized_key.replace('/', '_').replace('\\', '_')
        base_name = os.path.splitext(os.path.basename(safe_recognized_key))[0] if '.' in safe_recognized_key else safe_recognized_key
        output_filename = f"{base_name}_transfer_annotated.jpg"

        # 将图片转换为字节流
        _, buffer = cv2.imencode('.jpg', annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        image_bytes = buffer.tobytes()

        # 上传到MinIO
        result_object_key = self.minio_helper.upload_bytes(image_bytes, output_filename)

        # 清理临时下载的文件
        for local_path in [recognized_local_path, unrecognized_local_path]:
            if os.path.exists(local_path):
                os.remove(local_path)

        return result_object_key

    def _find_detection_boxes_by_center_points(self, image, center_points, tolerance=5):
        """
        通过中心点在图片中找到对应的识别框
        内部方法，用于检测框转移功能
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用Canny边缘检测找到所有边缘
        edges = cv2.Canny(gray, 50, 150)

        # 寻找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_boxes = []

        for contour in contours:
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)

            # 过滤掉太小的轮廓（可能是噪声）
            if w < 10 or h < 10:
                continue

            # 计算中心点
            center_x = x + w // 2
            center_y = y + h // 2

            # 检查是否与给定中心点匹配
            for target_cx, target_cy in center_points:
                distance = np.sqrt((center_x - target_cx)**2 + (center_y - target_cy)**2)

                if distance <= tolerance:
                    detected_boxes.append([x, y, x+w, y+h])
                    break

        # 如果边缘检测没有找到任何框，尝试颜色检测
        if not detected_boxes:
            detected_boxes = self._find_boxes_by_color(image, center_points, tolerance)

        return detected_boxes

    def _find_boxes_by_color(self, image, center_points, tolerance=5):
        """
        根据颜色查找检测框（用于处理彩色检测框）
        内部方法，用于检测框转移功能
        """
        detected_boxes = []

        # 定义常见检测框颜色的HSV范围
        colors = [
            ([0, 50, 50], [10, 255, 255]),      # 红色
            ([170, 50, 50], [180, 255, 255]),  # 红色（HSV环的另一侧）
            ([100, 50, 50], [130, 255, 255]),  # 蓝色
            ([40, 50, 50], [80, 255, 255]),    # 绿色
            ([20, 50, 50], [40, 255, 255]),    # 黄色
            ([0, 0, 200], [180, 30, 255])      # 白色/浅色
        ]

        for lower, upper in colors:
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

            # 创建颜色掩码
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)

            # 应用形态学操作清理掩码
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # 寻找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # 获取边界框
                x, y, w, h = cv2.boundingRect(contour)

                # 过滤掉太小的轮廓
                if w < 10 or h < 10:
                    continue

                # 计算中心点
                center_x = x + w // 2
                center_y = y + h // 2

                # 检查是否与给定中心点匹配
                for target_cx, target_cy in center_points:
                    distance = np.sqrt((center_x - target_cx)**2 + (center_y - target_cy)**2)

                    if distance <= tolerance:
                        detected_boxes.append([x, y, x+w, y+h])
                        break

        return detected_boxes