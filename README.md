# 视频流 YOLO 检测系统需求清单

## 项目概述

基于 FastAPI 的实时视频流目标检测服务，集成 YOLO 模型、ByteTrack 跟踪算法、FFmpeg GPU 解码/推流、MinIO 存储与 MySQL 数据库。

---
日志格式如下：
logs/
├── app.log              # 全局日志（启动、配置、异常）
├── stream_{task_id}.log # 各任务独立日志
├── performance.log      # 性能指标（FPS、延迟、队列深度）
└── error.log            # 仅错误日志（快速定位问题）

上报流程如下：
目标出现 → 开始追踪 → 累积检测数据 → 目标消失(track_buffer帧后ByteTrack丢弃) → 上报最终结果

``` 服务器显卡驱动安装命令：
1、ubuntu-drivers devices
2、ubuntu-drivers autoinstall
3、reboot
# 如果nvidia-smi显示的cuda版本低于12.4，执行以下命令，检查是否有更高版本支持
1、添加NVIDIA驱动PPA
add-apt-repository ppa:graphics-drivers/ppa
apt update
2、找版本
apt list 2>/dev/null | grep nvidia-driver
3、安装可用版本
apt install nvidia-driver-*version*
4、重启
reboot
```
```conda 环境安装
# 下载miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# 运行安装命令 (1、ENTER 2、yes 3、直接回车 4、yes)
bash Miniconda3-latest-Linux-x86_64.sh
# 手动初始化 如果在安装时没有自动初始化执行此命令
~/miniconda3/bin/conda init bash
# 生效环境变量
source ~/.bashrc
# 开始配置虚拟环境前，需要同意执行条款
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
# 登录时不默认进入conda环境命令
conda config --set auto_activate_base false
```

## 一、环境与依赖

### 1.1 运行环境

操作系统 Ubuntu 20.04 / 22.04 
Python 3.10  
GPU NVIDIA GPU，支持 NVDEC  
CUDA 12.4
cuDNN  9.1 +         
FFmpeg 4.2.7  
MySQL 
MinIO 
SRS  推流目标: `rtmp://112.14.53.185/live/stream/`

### 1.2 FFmpeg 安装与验证

**预计版本：FFmpeg 6.1.2（LTS 长期维护版本） 服务器限制实际使用 4.2.7**

``` FFmpeg 安装
# 更新软件包列表
apt update
# 安装 ffmpeg
apt install ffmpeg
# 验证安装
ffmpeg -version
```

``` 环境安装命令
conda create -n video-detection-server python=3.10 -y
conda activate video-detection-server
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

``` 服务启动命令
cd /usr/local/nbuav/ai/video-detection-server
conda activate video-detection
# 更改后清理缓存
ps aux | grep "python main.py" | grep -v grep
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
nohup python main.py > /usr/local/nbuav/ai/video-detection-server/logs/app.log 2>&1 &
```

# 测试接口
curl -v -X POST http://112.14.53.185:6950/stream \
  -H "Content-Type: application/json" \
  -d '{
    "streamUrl": "http://112.14.53.185:9000/nbuav/upload/file/99a7239c-d0f2-4733-a4be-becede2892a1.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=hxm%2F20260506%2Fnbuav%2Fs3%2Faws4_request&X-Amz-Date=20260506T081112Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=fea105d6076d27b978ca2324eea657fa35ecd69301a3184043b7bac0acf39b15",
    "taskId": "122121461736981",
    "algorithmId": "Sdiej96cw11"
  }'
