from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import json
from image_processor import ImageProcessor
from utils.orm_helper import ORMHelper
import uvicorn
import logging

logger = logging.getLogger(__name__)

class CropZoomRequest(BaseModel):
    image_path: str
    coordinates: List[float]  # [x1, y1, x2, y2]
    zoom_factor: float = 1.0
    output_filename: Optional[str] = None


class ProcessByTaskRequest(BaseModel):
    task_record_id: int
    zoom_factor: float = 1.0


class SaveImageSplitRequest(BaseModel):
    algorithm_record_id: int
    image_url: str


class TransferDetectionBoxRequest(BaseModel):
    recognized_image_path: str  # 识别后图片在MinIO中的object key
    unrecognized_image_path: str  # 未识别图片在MinIO中的object key
    bounding_boxes: str  # 边界框坐标字符串，格式为 "[[x1, y1, x2, y2], ...]" 或 "[x1, y1, x2, y2]"



app = FastAPI(title="Image Processing API", description="图片裁剪和放大API")

# 创建图片处理器实例
processor = ImageProcessor()
# 创建ORM Helper实例
orm_helper = ORMHelper()


@app.post("/crop_and_zoom")
async def crop_and_zoom_api(request: CropZoomRequest):
    """
    根据坐标裁剪并放大图片区域的接口
    """
    try:
        # 验证坐标参数
        if len(request.coordinates) != 4:
            raise HTTPException(status_code=400, detail="coordinates参数必须是包含4个数值的数组 [x1, y1, x2, y2]")
        
        # 处理图片
        result = processor.crop_and_zoom(
            request.image_path,
            request.coordinates,
            request.zoom_factor,
            request.output_filename
        )
        
        return {
            "success": True,
            "message": "图片处理成功",
            "result_path": result
        }
    
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理图片时发生错误: {str(e)}")


@app.post("/process_by_task")
async def process_by_task_api(request: ProcessByTaskRequest):
    """
    根据任务记录ID查询数据并处理图片
    """
    try:
        # 根据task_record_id获取算法识别结果
        recognition_records = ORMHelper.get_algorithm_image_list(request.task_record_id)
        
        if not recognition_records:
            raise HTTPException(status_code=404, detail="未找到对应的任务记录")
        
        results = []
        
        for record in recognition_records:
            # 获取识别结果分类信息
            class_records = ORMHelper.get_algorithm_image_class_list(record.id)
            
            # 从minio下载图片
            image_object_name = record.image_url  # 这是minio的object name
            minio_helper = processor.minio_helper  # 使用已有的minio_helper实例
            
            # 下载图片到临时文件
            temp_path = minio_helper.download_file(image_object_name)
            
            try:
                # 从分类记录中提取坐标信息
                # objects字段格式为 [x1, y1, x2, y2]
                for class_record in class_records:
                    try:
                        # 直接解析objects字段为坐标列表
                        coordinates = json.loads(class_record.objects)
                        if not isinstance(coordinates, list) or len(coordinates) != 4:
                            continue  # 跳过坐标格式不正确的记录
                        # 确保所有坐标都是数字
                        coordinates = [float(coord) for coord in coordinates]
                    except (json.JSONDecodeError, ValueError, TypeError):
                        continue  # 跳过无法解析的记录
                    
                    # 处理图片
                    result = processor.crop_and_zoom(
                        temp_path,
                        coordinates,
                        request.zoom_factor
                    )
                    
                    # 保存处理结果到AI图片截取记录表
                    split_record = orm_helper.insert_algorithm_image_split(
                        algorithm_record_id=record.id,
                        image_url=result
                    )
                    
                    results.append({
                        "record_id": record.id,
                        "class_name": class_record.class_name,
                        "original_image_url": record.image_url,
                        "result_path": result,
                        "split_record_id": split_record.id,
                        "coordinates": coordinates
                    })
            finally:
                # 清理临时文件
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        if not results:
            raise HTTPException(status_code=404, detail="未找到有效的坐标数据进行处理")
        
        return {
            "success": True,
            "message": "任务图片处理完成",
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理任务记录时发生错误: {str(e)}")


@app.post("/transfer_detection_box")
async def transfer_detection_box_api(request: TransferDetectionBoxRequest):
    """
    在图片上绘制检测框的接口
    """
    try:
        # 验证参数
        if not request.recognized_image_path or not isinstance(request.recognized_image_path, str):
            raise HTTPException(status_code=400, detail="recognized_image_path 参数必须是有效的字符串")

        if not request.unrecognized_image_path or not isinstance(request.unrecognized_image_path, str):
            raise HTTPException(status_code=400, detail="unrecognized_image_path 参数必须是有效的字符串")

        if not request.bounding_boxes or not isinstance(request.bounding_boxes, str):
            raise HTTPException(status_code=400, detail="bounding_boxes 参数必须是有效的字符串")
        
        logger.info(f"request.recognized_image_path：{request.recognized_image_path}， request.bounding_boxes： {request.bounding_boxes}")

        # 解析字符串格式的bounding_boxes参数
        import json
        try:
            # 尝试解析JSON字符串
            parsed_bounding_boxes = json.loads(request.bounding_boxes)

            # 验证解析结果格式
            if isinstance(parsed_bounding_boxes, list) and len(parsed_bounding_boxes) > 0:
                # 如果是单个边界框的格式 [x1, y1, x2, y2]
                if len(parsed_bounding_boxes) == 4 and all(isinstance(coord, (int, float)) for coord in parsed_bounding_boxes):
                    bounding_boxes = [parsed_bounding_boxes]  # 包装成单元素列表
                # 如果是多个边界框的格式 [[x1, y1, x2, y2], ...]
                elif all(isinstance(box, list) and len(box) == 4 and all(isinstance(coord, (int, float)) for coord in box) for box in parsed_bounding_boxes):
                    bounding_boxes = parsed_bounding_boxes
                else:
                    raise HTTPException(status_code=400, detail="bounding_boxes 格式不正确，应为 [x1, y1, x2, y2] 或 [[x1, y1, x2, y2], ...] 的JSON字符串")
            else:
                raise HTTPException(status_code=400, detail="bounding_boxes 参数解析后应为包含坐标列表的数组")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="bounding_boxes 参数必须是有效的JSON字符串")
        
        # 调用图片处理器的transfer_detection_box方法
        result = processor.transfer_detection_box(
            recognized_image_object_key=request.recognized_image_path,
            unrecognized_image_object_key=request.unrecognized_image_path,
            bounding_boxes=bounding_boxes
        )

        return {
            "success": True,
            "message": "检测框转移完成",
            "result_path": result  # 返回处理后图片在MinIO中的object key
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理图片时发生错误: {str(e)}")


@app.get("/health")
async def health_check():
    """
    健康检查接口
    """
    return {"status": "healthy", "message": "服务运行正常"}


if __name__ == '__main__':
    # 从环境变量获取端口，默认为5000
    port = int(os.environ.get('PORT', 6951))
    uvicorn.run("api:app", host='0.0.0.0', port=port, reload=True)