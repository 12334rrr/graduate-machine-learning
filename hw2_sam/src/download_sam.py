# -*- coding: utf-8 -*-
"""下载 SAM ViT-B 原版权重（sam_vit_b_01ec64.pth，375 MB）。

本机已存在该文件时直接跳过。换机复现时优先官方直链；若官方不可达，
脚本会给出 hf-mirror / ModelScope 的手动下载指引（不硬编码可能失效的镜像路径）。

用法：
    python src/download_sam.py
    python src/download_sam.py --url https://your-mirror/path/sam_vit_b_01ec64.pth
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import SAM_CKPT  # noqa: E402

OFFICIAL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
MIN_BYTES = 300 * 1024 * 1024          # 完整权重约 375 MB，小于此值视为下载不完整


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 DatasetHub/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)          # 1 MB 分块，避免占内存
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done / total * 100
                sys.stdout.write(f"\r  下载中 {done / 1048576:.1f}/{total / 1048576:.1f} MB ({pct:.1f}%)")
                sys.stdout.flush()
    print()
    if tmp.stat().st_size < MIN_BYTES:
        raise RuntimeError(f"下载文件过小（{tmp.stat().st_size} B），可能被重定向到错误页面")
    tmp.replace(dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=OFFICIAL_URL, help="权重直链（默认 Meta 官方）")
    ap.add_argument("--force", action="store_true", help="已存在时仍重新下载")
    args = ap.parse_args()

    if SAM_CKPT.exists() and not args.force:
        size = SAM_CKPT.stat().st_size
        print(f"[跳过] 权重已存在：{SAM_CKPT}（{size / 1048576:.1f} MB）")
        return

    print(f"[下载] {args.url}")
    try:
        download(args.url, SAM_CKPT)
    except (urllib.error.URLError, RuntimeError, OSError) as exc:
        print(f"[失败] {exc.__class__.__name__}: {exc}")
        print("\n可尝试以下镜像（任选其一，下载后放到：")
        print(f"  {SAM_CKPT}）")
        print("  - hf-mirror：https://hf-mirror.com/ 搜索 facebook/sam-vit-base")
        print("  - ModelScope：搜索 sam-vit-base / sam_vit_b_01ec64.pth")
        print("  - 或使用 --url 指定可达的直链")
        raise SystemExit(1)
    print(f"[完成] {SAM_CKPT}（{SAM_CKPT.stat().st_size / 1048576:.1f} MB）")


if __name__ == "__main__":
    main()
