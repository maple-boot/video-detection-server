#!/usr/bin/env python3
"""
PaddleDetection 数据集准备脚本
功能：
  1. 将扁平 images/ 按 train/val/test JSON 拆分到子目录
  2. 生成 PaddleDetection 数据集配置 YAML
  3. 校验标注完整性

目录结构转换：
  转换前:                     转换后:
  dataset/                   dataset/
  ├── annotations/           ├── annotations/
  │   ├── train.json         │   ├── train.json    (不动)
  │   ├── val.json           │   ├── val.json
  │   └── test.json          │   └── test.json
  └── images/                ├── images/
      ├── 000001.jpg         │   ├── train/
      ├── 000002.jpg         │   │   ├── 000001.jpg
      └── ...                │   │   └── ...
                             │   ├── val/
                             │   │   ├── 000003.jpg
                             │   │   └── ...
                             │   └── test/
                             │       ├── 000005.jpg
                             │       └── ...
                             └── dataset.yml       (自动生成)
"""

import json
import os
import shutil
import argparse
import sys
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def validate_annotation(anno_path):
    """校验 COCO JSON 结构完整性"""
    data = load_json(anno_path)
    errors = []

    # 检查必要字段
    for key in ["images", "annotations", "categories"]:
        if key not in data:
            errors.append(f"缺少必要字段: {key}")

    if errors:
        return False, errors

    # 检查图片字段
    for img in data["images"]:
        for field in ["id", "file_name"]:
            if field not in img:
                errors.append(f"image 缺少字段 {field}: {img}")
                break

    # 检查标注字段
    for ann in data["annotations"]:
        for field in ["id", "image_id", "category_id", "bbox"]:
            if field not in ann:
                errors.append(f"annotation 缺少字段 {field}: {ann}")
                break

    # 检查类别字段
    for cat in data["categories"]:
        for field in ["id", "name"]:
            if field not in cat:
                errors.append(f"category 缺少字段 {field}: {cat}")
                break

    if errors:
        return False, errors[:10]  # 最多显示 10 个

    return True, []


def split_images(dataset_dir, split_name, anno_data):
    """根据 JSON 将 images/ 中的图片复制或软链接到子目录"""
    src_img_dir = os.path.join(dataset_dir, "images")
    dst_img_dir = os.path.join(dataset_dir, "images", split_name)
    os.makedirs(dst_img_dir, exist_ok=True)

    image_files = {img["file_name"] for img in anno_data["images"]}
    copied, missing = 0, 0

    for fname in image_files:
        # 处理 file_name 可能带子路径的情况
        basename = os.path.basename(fname)
        src = os.path.join(src_img_dir, basename)
        dst = os.path.join(dst_img_dir, basename)

        if os.path.exists(src):
            if not os.path.exists(dst):
                # 优先硬链接节省空间，不支持则用软链接
                try:
                    os.link(src, dst)
                except OSError:
                    try:
                        os.symlink(os.path.abspath(src), dst)
                    except OSError:
                        shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1

    return copied, missing


def update_annotation_filenames(anno_data, split_name):
    """更新 JSON 中的 file_name 为带子目录的路径（PaddleDetection 需要）"""
    for img in anno_data["images"]:
        basename = os.path.basename(img["file_name"])
        img["file_name"] = os.path.join(split_name, basename)
    return anno_data


def get_category_info(anno_path):
    """从 JSON 提取类别信息"""
    data = load_json(anno_path)
    categories = sorted(data["categories"], key=lambda x: x["id"])
    return categories


def generate_paddle_config(dataset_dir, categories, splits):
    """生成 PaddleDetection 数据集配置文件"""
    cat_names = [c["name"] for c in categories]
    num_classes = len(categories)

    # 生成 dataset.yml
    lines = [
        "# 自动生成的 PaddleDetection 数据集配置",
        f"# 生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"metric: COCO",
        f"num_classes: {num_classes}",
        "",
    ]

    # 类别映射
    lines.append("# 类别列表")
    lines.append(f"# 类别数: {num_classes}")
    for i, cat in enumerate(categories):
        lines.append(f"#   {cat['id']}: {cat['name']}")
    lines.append("")

    # TrainDataset
    if "train" in splits:
        lines.extend([
            "TrainDataset:",
            "  !COCODataSet",
            f"    image_dir: images",
            f"    anno_path: annotations/train.json",
            f"    dataset_dir: .",
            "    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']",
            "",
        ])

    # EvalDataset
    if "val" in splits:
        lines.extend([
            "EvalDataset:",
            "  !COCODataSet",
            f"    image_dir: images",
            f"    anno_path: annotations/val.json",
            f"    dataset_dir: .",
            "",
        ])

    # TestDataset
    if "test" in splits:
        lines.extend([
            "TestDataset:",
            "  !ImageFolder",
            f"    anno_path: annotations/test.json",
            "",
        ])

    config_content = "\n".join(lines)
    config_path = os.path.join(dataset_dir, "dataset.yml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)

    return config_path, config_content


def print_summary(dataset_dir, categories, split_stats):
    """打印数据集摘要"""
    print("\n" + "=" * 60)
    print("数据集摘要")
    print("=" * 60)
    print(f"  数据集路径:  {dataset_dir}")
    print(f"  类别数量:    {len(categories)}")
    print(f"  类别列表:")
    for cat in categories:
        print(f"    {cat['id']:>3d}: {cat['name']}")
    print()
    for split, stats in split_stats.items():
        print(f"  [{split}]")
        print(f"    图片数:    {stats['images']}")
        print(f"    标注数:    {stats['annotations']}")
        print(f"    已复制:    {stats['copied']}")
        if stats["missing"] > 0:
            print(f"    缺失图片:  {stats['missing']} ⚠")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="PaddleDetection 数据集准备工具")
    parser.add_argument("--dataset_dir", required=True,
                        help="数据集根目录，包含 images/ 和 annotations/")
    parser.add_argument("--no_split", action="store_true",
                        help="不拆分图片子目录，仅生成配置（适用于已有子目录的场景）")
    parser.add_argument("--update_anno", action="store_true",
                        help="更新 JSON 中的 file_name 为带子目录路径（拆分模式下自动启用）")
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    anno_dir = os.path.join(dataset_dir, "annotations")
    img_dir = os.path.join(dataset_dir, "images")

    # 校验目录
    if not os.path.isdir(anno_dir):
        print(f"错误: 标注目录不存在: {anno_dir}")
        sys.exit(1)
    if not os.path.isdir(img_dir):
        print(f"错误: 图片目录不存在: {img_dir}")
        sys.exit(1)

    # 扫描可用的 split
    split_files = {}
    for split_name in ["train", "val", "test"]:
        anno_file = os.path.join(anno_dir, f"{split_name}.json")
        if os.path.exists(anno_file):
            split_files[split_name] = anno_file

    if not split_files:
        print(f"错误: annotations/ 下未找到 train.json / val.json / test.json")
        sys.exit(1)

    print(f"发现标注文件: {list(split_files.keys())}")

    # 校验每个 JSON
    all_categories = None
    for split_name, anno_file in split_files.items():
        print(f"\n校验 {split_name}.json ...")
        valid, errors = validate_annotation(anno_file)
        if not valid:
            print(f"  ❌ 校验失败:")
            for e in errors:
                print(f"    - {e}")
            sys.exit(1)

        data = load_json(anno_file)
        print(f"  ✅ 图片: {len(data['images'])}, 标注: {len(data['annotations'])}, 类别: {len(data['categories'])}")

        # 取第一个的类别作为统一类别
        if all_categories is None:
            all_categories = sorted(data["categories"], key=lambda x: x["id"])

    # 拆分图片
    split_stats = {}
    for split_name, anno_file in split_files.items():
        anno_data = load_json(anno_file)
        stats = {
            "images": len(anno_data["images"]),
            "annotations": len(anno_data["annotations"]),
            "copied": 0,
            "missing": 0,
        }

        if not args.no_split:
            print(f"\n拆分 {split_name} 图片 ...")
            copied, missing = split_images(dataset_dir, split_name, anno_data)
            stats["copied"] = copied
            stats["missing"] = missing
            print(f"  复制: {copied}, 缺失: {missing}")

        split_stats[split_name] = stats

    # 生成配置文件
    config_path, config_content = generate_paddle_config(
        dataset_dir, all_categories, split_files
    )
    print(f"\n配置文件已生成: {config_path}")
    print(f"\n{config_content}")

    # 打印摘要
    print_summary(dataset_dir, all_categories, split_stats)

    # 提示
    print(f"\n接下来你可以:")
    print(f"  1. 将 {dataset_dir} 上传至 AI Studio 数据集")
    print(f"  2. 在 PaddleDetection 配置中引用:")
    print(f"     _BASE_: ['{config_path}']")


if __name__ == "__main__":
    main()
