#!/bin/bash

APP_NAME="python main.py"
LOG_DIR="/usr/local/nbuav/ai/video-detection-server/logs"
LOG_FILE="${LOG_DIR}/app.log"
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"

# 确保日志目录存在
mkdir -p "$LOG_DIR"

# 杀掉已有的进程
PIDS=$(pgrep -f "$APP_NAME" | head -n 5)
if [ -n "$PIDS" ]; then
    echo ">>> 发现已有进程，正在终止: $PIDS"
    echo "$PIDS" | xargs kill -9
    sleep 1
    echo ">>> 已终止"
else
    echo ">>> 未发现运行中的进程"
fi

# 清理 __pycache__
find "$WORK_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
echo ">>> 已清理 __pycache__"

# 启动新进程
cd "$WORK_DIR" || exit 1
nohup python main.py > "$LOG_FILE" 2>&1 &
NEW_PID=$!
echo ">>> 已启动 (PID: $NEW_PID)"
echo ">>> 日志: $LOG_FILE"
echo ">>> 使用 tail -f $LOG_FILE 查看日志"

