import os
from datetime import timedelta
from minio import Minio
from minio.error import S3Error
from app.utils.logger import get_system_logger

logger = get_system_logger()


class MinioHelper:
    """MinIO 对象存储封装，支持多配置"""

    def __init__(self, config: dict):
        self.clients = {}
        minio_configs = config.get("minio", {})

        for name, cfg in minio_configs.items():
            client = Minio(
                endpoint=cfg["endpoint"],
                access_key=cfg["access_key"],
                secret_key=cfg["secret_key"],
                secure=cfg.get("secure", False),
            )
            bucket = cfg.get("bucket", "nbuav")
            self._ensure_bucket(client, bucket)
            self.clients[name] = {"client": client, "bucket": bucket}
            logger.info(f"MinIO 客户端初始化完成 | name={name} | endpoint={cfg['endpoint']} | bucket={bucket}")

    def _ensure_bucket(self, client: Minio, bucket: str):
        """确保 Bucket 存在"""
        try:
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket)
                logger.info(f"MinIO Bucket 创建成功: {bucket}")
        except S3Error as e:
            logger.error(f"MinIO Bucket 检查失败: {e}")

    def get_client(self, name: str = "default") -> tuple:
        """获取指定名称的 MinIO 客户端和 Bucket"""
        if name not in self.clients:
            logger.warning(f"MinIO 配置 '{name}' 不存在，使用 default")
            name = "default"
        cfg = self.clients[name]
        return cfg["client"], cfg["bucket"]

    def upload_file(self, file_path: str, object_name: str, minio_name: str = "default") -> str:
        """上传文件到 MinIO，返回对象名"""
        client, bucket = self.get_client(minio_name)
        try:
            client.fput_object(bucket, object_name, file_path)
            logger.info(f"文件上传成功 | bucket={bucket} | object={object_name}")
            return object_name
        except S3Error as e:
            logger.error(f"文件上传失败: {e}")
            return ""

    def upload_bytes(self, data: bytes, object_name: str, content_type: str = "image/jpeg",
                     minio_name: str = "default") -> str:
        """上传字节数据到 MinIO"""
        import io
        client, bucket = self.get_client(minio_name)
        try:
            data_stream = io.BytesIO(data)
            client.put_object(bucket, object_name, data_stream, len(data), content_type)
            logger.debug(f"字节数据上传成功 | object={object_name}")
            return object_name
        except S3Error as e:
            logger.error(f"字节数据上传失败: {e}")
            return ""

    def get_presigned_url(self, object_name: str, minio_name: str = "default",
                          expires: int = 3600) -> str:
        """获取预签名 URL"""
        client, bucket = self.get_client(minio_name)
        try:
            url = client.presigned_get_object(
                bucket, object_name,
                expires=timedelta(seconds=expires)
            )
            return url
        except S3Error as e:
            logger.error(f"获取预签名 URL 失败: {e}")
            return ""

    def get_public_url(self, object_name: str, minio_name: str = "default") -> str:
        """拼接公开访问 URL（适用于 MinIO 配置了公开访问的场景）"""
        cfg = self.clients.get(minio_name, self.clients.get("default"))
        endpoint = cfg["client"]._base_url._url.netloc
        bucket = cfg["bucket"]
        protocol = "https" if cfg["client"]._base_url._url.scheme == "https" else "http"
        return f"{protocol}://{endpoint}/{bucket}/{object_name}"

    def download_file(self, object_name: str, file_path: str = None, minio_name: str = "default") -> str:
        """从 MinIO 下载文件到本地临时目录"""
        import tempfile
        import re
        safe_object_name = re.sub(r'[\x00-\x1f\x7f]', '', object_name)

        client, bucket = self.get_client(minio_name)

        if file_path is None:
            _, ext = os.path.splitext(safe_object_name)
            file_path = os.path.join(tempfile.gettempdir(), os.urandom(8).hex() + (ext or '.tmp'))

        client.fget_object(bucket, safe_object_name, file_path)
        logger.info(f"文件下载成功 | bucket={bucket} | object={safe_object_name} -> {file_path}")
        return file_path

    def delete_file(self, object_name: str, minio_name: str = "default") -> bool:
        """删除文件"""
        client, bucket = self.get_client(minio_name)
        try:
            client.remove_object(bucket, object_name)
            logger.info(f"文件删除成功 | object={object_name}")
            return True
        except S3Error as e:
            logger.error(f"文件删除失败: {e}")
            return False
