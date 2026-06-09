#!/usr/bin/env python3
"""
ChangeMark Server — 变化检测标注工具服务端

用法:
  python server.py -d /path/to/dataset
  python server.py -d /path/to/dataset -o /path/to/annotations
  python server.py -d /path/to/dataset -o /path/to/annotations -p 8080

参数:
  -d / --dir     图片数据集目录（必须）
  -o / --output  标注结果保存目录（可选，默认为数据集目录下的 .changemark_ann）
  -p / --port    HTTP 端口（可选，默认 8080）

自动配对命名规则:
  {name}_t1.ext / {name}_t2.ext
  {name}_T1.ext / {name}_T2.ext
  {name}_before.ext / {name}_after.ext
  {name}_pre.ext / {name}_post.ext
  {name}_A.ext / {name}_B.ext
  无匹配时按文件名排序两两配对
"""
import http.server
import socketserver
import json
import os
import re
import sys
import argparse
import io
import zipfile
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

IMG_EXT = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}

PAIR_RULES = [
    ('_t1', '_t2'), ('_T1', '_T2'),
    ('_before', '_after'), ('_Before', '_After'),
    ('_pre', '_post'), ('_Pre', '_Post'),
    ('_A', '_B'), ('_a', '_b'),
]


# ─────────────────────────────────────────────────────────
# Dataset：管理图片扫描、配对、标注读写
# ─────────────────────────────────────────────────────────
class Dataset:
    def __init__(self, img_dir: str, ann_dir: str):
        self.img_root = Path(img_dir).resolve()
        self.ann_root = Path(ann_dir).resolve()

        # 确保标注目录存在
        self.ann_root.mkdir(parents=True, exist_ok=True)

        self.pairs = []
        self._scan()

        print(f'  数据集目录 : {self.img_root}')
        print(f'  标注保存到 : {self.ann_root}')
        print(f'  图片对数量 : {len(self.pairs)}')

    # ── 扫描 & 配对 ──
    def _scan(self):
        files = sorted(
            [f for f in self.img_root.iterdir()
             if f.is_file() and f.suffix.lower() in IMG_EXT],
            key=lambda f: f.name.lower()
        )
        names_map = {f.name: f for f in files}
        used = set()
        pairs = []

        # 1) 按命名规则配对
        for suf_a, suf_b in PAIR_RULES:
            for f in files:
                if f.name in used:
                    continue
                stem, ext = f.stem, f.suffix
                if not stem.endswith(suf_a):
                    continue
                base = stem[:-len(suf_a)]
                mate_name = base + suf_b + ext
                if mate_name in names_map and mate_name not in used:
                    pair_id = base if base else f'pair_{len(pairs):05d}'
                    pairs.append({
                        'id': pair_id,
                        't1': f.name,
                        't2': mate_name,
                    })
                    used.update([f.name, mate_name])
            if pairs:
                break

        # 2) 剩余文件按顺序两两配对
        remaining = [f.name for f in files if f.name not in used]
        for i in range(0, len(remaining) - 1, 2):
            pairs.append({
                'id': Path(remaining[i]).stem,
                't1': remaining[i],
                't2': remaining[i + 1],
            })

        self.pairs = pairs

    # ── 获取单个配对信息 ──
    def get(self, idx: int):
        if 0 <= idx < len(self.pairs):
            d = dict(self.pairs[idx])
            d['index'] = idx
            d['annotated'] = self.mask_path(idx).exists()
            return d
        return None

    # ── 标注文件路径（保存在 ann_root 下） ──
    def mask_path(self, idx: int) -> Path:
        return self.ann_root / f'{idx:05d}.png'

    # ── 图片文件路径（位于 img_root 下） ──
    def image_path(self, name: str) -> Path:
        return self.img_root / name

    # ── 保存标注 ──
    def save_mask(self, idx: int, data: bytes):
        self.mask_path(idx).write_bytes(data)

    # ── 统计信息 ──
    def stats(self) -> dict:
        total = len(self.pairs)
        done = sum(1 for i in range(total) if self.mask_path(i).exists())
        return {
            'total': total,
            'done': done,
            'progress': round(done / max(total, 1) * 100, 1),
            'img_dir': str(self.img_root),
            'ann_dir': str(self.ann_root),
        }


# ─────────────────────────────────────────────────────────
# HTTP Handler
# ─────────────────────────────────────────────────────────
class Handler(http.server.BaseHTTPRequestHandler):
    ds: Dataset = None  # 由 main() 注入

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        qs = parse_qs(urlparse(self.path).query)

        # ── 路由 ──
        if path in ('/', '/index.html'):
            self._serve_file('index.html', 'text/html; charset=utf-8')

        elif path == '/api/stats':
            self._json(self.ds.stats())

        elif path == '/api/pairs':
            self._pairs(qs)

        elif re.match(r'/api/img/\d+/(t1|t2)$', path):
            self._serve_image(path)

        elif re.match(r'/api/mask/\d+$', path):
            self._serve_mask(path)

        elif path == '/api/export':
            self._export_zip()

        else:
            self._serve_static(path)

    def do_PUT(self):
        path = unquote(urlparse(self.path).path)
        m = re.match(r'/api/mask/(\d+)$', path)
        if m:
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            self.ds.save_mask(int(m.group(1)), body)
            self._json({'ok': True, 'saved': str(self.ds.mask_path(int(m.group(1))))})
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    # ── 配对列表（支持搜索 & 筛选） ──
    def _pairs(self, qs):
        search = qs.get('q', [''])[0].lower()
        filt = qs.get('filter', ['all'])[0]
        result = []
        for i, p in enumerate(self.ds.pairs):
            ann = self.ds.mask_path(i).exists()
            if filt == 'done' and not ann:
                continue
            if filt == 'todo' and ann:
                continue
            if search and search not in p['id'].lower() and search not in p['t1'].lower():
                continue
            result.append({**p, 'index': i, 'annotated': ann})
        self._json({
            'total': len(result),
            'items': result,
            'stats': self.ds.stats(),
        })

    # ── 返回图片 ──
    def _serve_image(self, path):
        parts = path.split('/')
        idx, side = int(parts[3]), parts[4]
        pair = self.ds.get(idx)
        if not pair:
            return self.send_error(404)
        fp = self.ds.image_path(pair[side])
        if not fp.exists():
            return self.send_error(404)
        self._serve_file(str(fp), self._mime(fp.suffix))

    # ── 返回已有标注 ──
    def _serve_mask(self, path):
        idx = int(path.split('/')[3])
        fp = self.ds.mask_path(idx)
        if fp.exists():
            self._serve_file(str(fp), 'image/png')
        else:
            self.send_error(404)

    # ── 导出全部标注 ZIP ──
    def _export_zip(self):
        buf = io.BytesIO()
        count = 0
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for i in range(len(self.ds.pairs)):
                mp = self.ds.mask_path(i)
                if mp.exists():
                    # zip 内使用配对 id 作为文件名
                    zf.write(mp, f'mask_{self.ds.pairs[i]["id"]}.png')
                    count += 1
        data = buf.getvalue()
        self.send_response(200)
        self.send_header('Content-Type', 'application/zip')
        self.send_header('Content-Disposition',
                         'attachment; filename="changemark_masks.zip"')
        self.send_header('Content-Length', len(data))
        self._cors()
        self.end_headers()
        self.wfile.write(data)
        print(f'  导出 ZIP: {count} 个标注文件')

    # ── 文件 / 静态资源 ──
    def _serve_file(self, fp, ct):
        try:
            data = Path(fp).read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', ct)
            self.send_header('Content-Length', len(data))
            self._cors()
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self.send_error(404)

    def _serve_static(self, path):
        fp = path.lstrip('/')
        if os.path.isfile(fp):
            self._serve_file(fp, self._mime(Path(fp).suffix))
        else:
            self.send_error(404)

    def _json(self, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', len(data))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, PUT, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    @staticmethod
    def _mime(ext):
        return {
            '.png': 'image/png', '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg', '.tif': 'image/tiff',
            '.tiff': 'image/tiff', '.bmp': 'image/bmp',
            '.webp': 'image/webp',
            '.html': 'text/html; charset=utf-8',
            '.js': 'application/javascript',
            '.css': 'text/css',
        }.get(ext.lower(), 'application/octet-stream')

    def log_message(self, *a):
        pass


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description='ChangeMark 标注服务',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 标注保存在数据集目录下的 .changemark_ann
  python server.py -d ./dataset

  # 标注保存到自定义目录
  python server.py -d ./dataset -o ./my_annotations

  # 指定端口
  python server.py -d ./dataset -o ./my_annotations -p 9090
        """
    )
    ap.add_argument('-d', '--dir', required=True,
                    help='图片数据集目录')
    ap.add_argument('-o', '--output', default=None,
                    help='标注结果保存目录（默认：<数据集目录>/.changemark_ann）')
    ap.add_argument('-p', '--port', type=int, default=8080,
                    help='HTTP 端口（默认 8080）')
    args = ap.parse_args()

    # ── 验证数据集目录 ──
    img_dir = Path(args.dir).resolve()
    if not img_dir.is_dir():
        print(f'  错误: 数据集目录不存在 → {img_dir}')
        sys.exit(1)

    # ── 确定标注保存目录 ──
    if args.output:
        ann_dir = Path(args.output).resolve()
    else:
        ann_dir = img_dir / '.changemark_ann'

    # ── 启动 ──
    print(f'\n  ChangeMark Server')
    print(f'  {"═" * 40}')

    Handler.ds = Dataset(str(img_dir), str(ann_dir))
    stats = Handler.ds.stats()

    print(f'  已标注数量 : {stats["done"]}  ({stats["progress"]}%)')

    if stats['total'] == 0:
        print(f'\n  ⚠  未找到图片对，请检查目录中的文件命名')
        print(f'  支持的命名: name_t1 / name_t2, name_before / name_after ...')

    print(f'\n  → http://localhost:{args.port}')
    print(f'  按 Ctrl+C 停止\n')

    with socketserver.ThreadingTCPServer(('', args.port), Handler) as srv:
        srv.allow_reuse_address = True
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print(f'\n  已停止')


if __name__ == '__main__':
    main()
