# 1. 上传单个文件
python upload.py --type file \
--repo_id yourname/dataset \
--local ./data.csv \
--remote data.csv

# 2. 上传普通文件夹（SDK 内置并发）
python upload.py --type folder \
--repo_id yourname/dataset \
--local ./my_data \
--remote my_data/ \
--max_workers 16

# 3. 上传大文件夹（多级并发，推荐大文件量场景）
python upload_file.py --type large --repo_id maplesyr/smoke_fire_uav --local "D:\\BaiduNetdiskDownload\\smoke-datasets" --max_workers 16 --ignore "*.log *.tmp __pycache__"

# 4. 设置 token 环境变量后可免输入
export AISTUDIO_ACCESS_TOKEN="your_token_here"
python upload.py --type large --repo_id yourname/dataset --local ./big_dataset --remote big_dataset/

# 5、 运行数据集拆分
# 运行转换（拆分图片 + 生成配置）
python prepare_dataset.py --dataset_dir D:\\BaiduNetdiskDownload\\smoke-datasets

# 不拆分图片（只生成配置文件，适用于已经分好目录的情况）
python prepare_dataset.py --dataset_dir ./dataset --no_split
