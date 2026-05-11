from fastapi.responses import JSONResponse
from typing import Any


class ResponseHelper:
    """API 响应格式统一"""

    @staticmethod
    def success(data: Any = None, message: str = "success") -> JSONResponse:
        return JSONResponse(
            status_code=200,
            content={
                "code": 200,
                "message": message,
                "data": data,
            }
        )

    @staticmethod
    def error(message: str = "error", code: int = 500, data: Any = None) -> JSONResponse:
        return JSONResponse(
            status_code=code,
            content={
                "code": code,
                "message": message,
                "data": data,
            }
        )

    @staticmethod
    def bad_request(message: str = "参数错误") -> JSONResponse:
        return ResponseHelper.error(message=message, code=400)

    @staticmethod
    def not_found(message: str = "资源不存在") -> JSONResponse:
        return ResponseHelper.error(message=message, code=404)
