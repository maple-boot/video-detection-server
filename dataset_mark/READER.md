```
# 开启数据集标注服务
nohup python server.py -d /media0/chy/shuju/ai/data/jpg_dir/xiwu/total_jpg -o /media0/chy/shuju/ai/data/jpg_dir/xiwu/completed_jpg -p 6951 > changemark_server.log 2>&1 &
#                ├─ 图片来源        ├─ 标注去处        └─ 端口
# 处理进程
kill $(ps aux | grep 'server.py.*-p 8081' | grep -v grep | awk '{print $2}')
```
