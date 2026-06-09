#!/usr/bin/env python3
"""
AI Studio 数据集上传脚本
用法:
  上传文件:   python upload.py --type file --repo_id myname/reponame --local ./README.md --remote README.md
  上传文件夹: python upload.py --type folder --repo_id myname/reponame --local ./data --remote data/
"""

import os
import sys
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_token():
    """从环境变量获取 token，支持交互式输入"""
    token = os.environ.get("AISTUDIO_ACCESS_TOKEN", "7aaf5fce97eb0471e72a31c10046e4e8d2a50c48")
    if not token:
        token = input("请输入 AISTUDIO_ACCESS_TOKEN (控制台-令牌获取): ").strip()
    if not token:
        print("错误: 未提供 Access Token，退出")
        sys.exit(1)
    os.environ["AISTUDIO_ACCESS_TOKEN"] = token
    return token


def upload_single_file(args):
    """上传单个文件"""
    from aistudio_sdk.hub import upload_file

    start = time.time()
    print(f"[文件上传] 本地: {args.local} -> 远程: {args.remote}")

    res = upload_file(
        repo_id=args.repo_id,
        path_or_fileobj=args.local,
        path_in_repo=args.remote,
        commit_message=args.message or f"upload file {os.path.basename(args.local)}",
        repo_type=args.repo_type,
        token=args.token,
    )

    elapsed = time.time() - start
    print(f"[文件上传] 完成，耗时 {elapsed:.1f}s")
    print(f"[结果] {res}")
    return res


def upload_single_folder(args):
    """上传文件夹（利用 SDK 内置并发 + 文件类型过滤）"""
    from aistudio_sdk.hub import upload_folder

    start = time.time()

    # 统计待上传文件数
    file_count = 0
    for root, dirs, files in os.walk(args.local):
        file_count += len(files)
    print(f"[文件夹上传] 本地: {args.local} -> 远程: {args.remote or '(根目录)'}")
    print(f"[文件夹上传] 待上传文件数: {file_count}")
    print(f"[并发线程数] {args.max_workers}")

    res = upload_folder(
        repo_id=args.repo_id,
        folder_path=args.local,
        path_in_repo=args.remote or "",
        commit_message=args.message or f"upload folder {os.path.basename(args.local)}",
        repo_type=args.repo_type,
        token=args.token,
        allow_patterns=args.allow,
        ignore_patterns=args.ignore,
        max_workers=args.max_workers,
        revision=args.branch,
    )

    elapsed = time.time() - start
    print(f"[文件夹上传] 完成，耗时 {elapsed:.1f}s")
    print(f"[结果] {res}")
    return res


def upload_large_folder(args):
    """
    大文件夹分批并发上传：
    将本地文件夹拆分为子目录，每个子目录作为一个任务并发上传，
    绕过 SDK 单次调用的线程限制，实现更高速度。
    """
    from aistudio_sdk.hub import upload_folder

    base_local = os.path.abspath(args.local)
    base_remote = args.remote or ""

    # 收集所有子目录（含根目录自身作为兜底上传非目录文件）
    sub_dirs = []
    for entry in sorted(os.listdir(base_local)):
        full_path = os.path.join(base_local, entry)
        if os.path.isdir(full_path):
            sub_dirs.append((full_path, os.path.join(base_remote, entry)))

    # 根目录下的单独文件用 upload_file 并发上传
    root_files = []
    for entry in sorted(os.listdir(base_local)):
        full_path = os.path.join(base_local, entry)
        if os.path.isfile(full_path):
            remote_path = os.path.join(base_remote, entry) if base_remote else entry
            root_files.append((full_path, remote_path))

    total_tasks = len(sub_dirs) + len(root_files)
    print(f"[大文件夹上传] 子目录: {len(sub_dirs)}, 根目录文件: {len(root_files)}, 总任务: {total_tasks}")
    print(f"[并发线程数] {args.max_workers}")

    start = time.time()
    results = []
    failed = []

    def _upload_dir(local_path, remote_path):
        from aistudio_sdk.hub import upload_folder as uf
        return uf(
            repo_id=args.repo_id,
            folder_path=local_path,
            path_in_repo=remote_path,
            commit_message=f"batch upload {os.path.basename(local_path)}",
            repo_type=args.repo_type,
            token=args.token,
            allow_patterns=args.allow,
            ignore_patterns=args.ignore,
            max_workers=min(8, os.cpu_count() + 4),
            revision=args.branch,
        )

    def _upload_file(local_path, remote_path):
        from aistudio_sdk.hub import upload_file as uf
        return uf(
            repo_id=args.repo_id,
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            commit_message=f"batch upload {os.path.basename(local_path)}",
            repo_type=args.repo_type,
            token=args.token,
        )

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}

        # 提交子目录任务
        for local_path, remote_path in sub_dirs:
            f = executor.submit(_upload_dir, local_path, remote_path)
            futures[f] = f"[目录] {os.path.basename(local_path)}"

        # 提交根目录文件任务
        for local_path, remote_path in root_files:
            f = executor.submit(_upload_file, local_path, remote_path)
            futures[f] = f"[文件] {os.path.basename(local_path)}"

        # 等待完成
        for i, future in enumerate(as_completed(futures), 1):
            task_name = futures[future]
            try:
                res = future.result()
                results.append(res)
                print(f"  [{i}/{total_tasks}] {task_name} 完成")
            except Exception as e:
                failed.append((task_name, str(e)))
                print(f"  [{i}/{total_tasks}] {task_name} 失败: {e}")

    elapsed = time.time() - start
    print(f"\n[上传完成] 成功: {len(results)}, 失败: {len(failed)}, 耗时: {elapsed:.1f}s")

    if failed:
        print("\n失败列表:")
        for name, err in failed:
            print(f"  - {name}: {err}")


def main():
    parser = argparse.ArgumentParser(description="AI Studio 数据集上传工具")
    parser.add_argument("--type", choices=["file", "folder", "large"], default="folder",
                        help="上传类型: file=单文件, folder=文件夹(SDK内置并发), large=大文件夹(多级并发)")
    parser.add_argument("--repo_id", required=True, help="数据集 repo_id，如 myname/reponame")
    parser.add_argument("--local", required=True, help="本地文件/文件夹路径")
    parser.add_argument("--remote", default="", help="远程目标路径，不填则上传到根目录")
    parser.add_argument("--repo_type", default="dataset", help="仓库类型，默认 dataset")
    parser.add_argument("--message", default="", help="commit message")
    parser.add_argument("--branch", default="master", help="目标分支，默认 master")
    parser.add_argument("--allow", default=None, help="允许的文件类型，如 *.json")
    parser.add_argument("--ignore", default=None, help="忽略的文件类型，如 *.log *.tmp")
    parser.add_argument("--max_workers", type=int, default=8, help="并发线程数，默认 8")

    args = parser.parse_args()

    # 获取 token
    args.token = get_token()

    # 检查本地路径
    if not os.path.exists(args.local):
        print(f"错误: 本地路径不存在: {args.local}")
        sys.exit(1)

    # 分发任务
    if args.type == "file":
        upload_single_file(args)
    elif args.type == "folder":
        upload_single_folder(args)
    elif args.type == "large":
        upload_large_folder(args)


if __name__ == "__main__":
    main()
