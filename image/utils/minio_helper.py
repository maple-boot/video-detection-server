import os
import logging
import io
from typing import Optional
from minio import Minio
import yaml

logger = logging.getLogger(__name__)

# 读取配置文件
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')

def load_minio_config():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config.get('minio', {})


def url_encode(s):
    """
    Simple URL encoding function to replace urllib.parse.quote
    Encodes characters that are unsafe in URLs: space, #, %, <, >, [, ], etc.
    """
    # Characters to encode (incomplete list, but covers common cases)
    unsafe_chars = {
        ' ': '%20', '#': '%23', '<': '%3C', '>': '%3E',
        '[': '%5B', ']': '%5D', '{': '%7B', '}': '%7D', '|': '%7C',
        '\\': '%5C', '^': '%5E', '`': '%60', '?': '%3F', '&': '%26',
        '=': '%3D', '+': '%2B', '$': '%24', ';': '%3B', '%': '%25',
        '(': '%28', ')': '%29'
    }
    result = ""
    for char in s:
        if char in unsafe_chars:
            result += unsafe_chars[char]
        elif ord(char) < 32 or ord(char) == 127:  # Control characters
            result += f'%{ord(char):02X}'
        else:
            result += char
    return result


class MinioHelper:
    def __init__(self):
        cfg = load_minio_config()
        self.endpoint = cfg.get('endpoint')
        self.access_key = cfg.get('access_key')
        self.secret_key = cfg.get('secret_key')
        self.secure = bool(cfg.get('secure', False))
        self.bucket_name = cfg.get('bucket_name')
        self._client = Minio(
            endpoint=self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure
        )

    def init_client(self):
        # 确保桶存在
        try:
            if not self._client.bucket_exists(self.bucket_name):
                self._client.make_bucket(self.bucket_name)
                logger.info(f"已创建 MinIO 桶: {self.bucket_name}")
        except Exception as e:
            logger.error(f"检查/创建桶失败: {e}")
            raise
        return self._client, self.bucket_name

    def _guess_content_type(self, object_name: str) -> str:
        ext = os.path.splitext(object_name)[1].lower()
        if ext in ('.jpg', '.jpeg'):
            return 'image/jpeg'
        if ext == '.png':
            return 'image/png'
        if ext == '.bmp':
            return 'image/bmp'
        return 'application/octet-stream'

    def upload_bytes(self, file_bytes: bytes, object_name: str, content_type: Optional[str] = None) -> str:
        """
        上传字节数据到MinIO
        :param file_bytes: 文件字节数据
        :param object_name: MinIO对象名称
        :param content_type: 内容类型
        :return: 保存在MinIO中的对象名称
        """
        # 验证并清理 object_name，防止包含空字符等控制字符
        import re
        safe_object_name = re.sub(r'[\x00-\x1f\x7f]', '', object_name)

        client, bucket = self.init_client()
        if content_type is None:
            content_type = self._guess_content_type(safe_object_name)

        # 使用 put_object 上传字节数据
        client.put_object(
            bucket_name=bucket,
            object_name=safe_object_name,
            data=io.BytesIO(file_bytes),
            length=len(file_bytes),
            content_type=content_type
        )
        logger.info(f"字节数据上传到 MinIO 完成: {bucket}/{safe_object_name}")
        return safe_object_name

    def download_file(self, object_name: str, file_path: str = None) -> str:
        """
        从MinIO下载文件到本地

        Args:
            object_name: MinIO中的对象名
            file_path: 本地保存路径，如果不提供则保存到临时文件

        Returns:
            本地文件路径
        """
        # 验证并清理 object_name，防止包含空字符等控制字符
        import re
        safe_object_name = re.sub(r'[\x00-\x1f\x7f]', '', object_name)

        client, bucket = self.init_client()

        if file_path is None:
            import tempfile
            _, ext = os.path.splitext(safe_object_name)
            file_path = os.path.join(tempfile.gettempdir(), os.urandom(8).hex() + ext)

        client.fget_object(bucket, safe_object_name, file_path)
        logger.info(f"从 MinIO 下载完成: {bucket}/{safe_object_name} -> {file_path}")
        return file_path

# 创建默认实例
minio_helper = MinioHelper()
