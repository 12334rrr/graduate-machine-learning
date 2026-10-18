# -*- coding: utf-8 -*-
"""
并行分块下载 ISIC 2018 官方压缩包。

为什么要分块：单连接从 S3 只有约 1.4 MB/s，13.8 GB 要下 2.7 小时；
开 12 路 Range 请求后聚合带宽能到 8~15 MB/s，20 分钟内下完。
每个分块单独落盘 + 断点续传，中断后重跑脚本会自动补齐未完成的分块。

用法：
    python src/download_isic.py                 # 下载缺失的包
    python src/download_isic.py --connections 16
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 项目根目录（本文件位于 <root>/hw1_unet/src/ 下）
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

BASE = "https://isic-challenge-data.s3.amazonaws.com/2018"

# (文件名, 预期字节数) —— 预期大小只用于进度显示，实际以服务器返回为准
TARGETS = [
    "ISIC2018_Task1-2_Training_Input.zip",
    "ISIC2018_Task1_Training_GroundTruth.zip",
    "ISIC2018_Task1-2_Validation_Input.zip",
    "ISIC2018_Task1_Validation_GroundTruth.zip",
    "ISIC2018_Task1-2_Test_Input.zip",
    "ISIC2018_Task1_Test_GroundTruth.zip",
]


def head_length(url: str) -> int:
    """用 HEAD 请求拿文件总长度（S3 支持 Range，必须能拿到长度）。"""
    out = subprocess.run(
        ["curl", "-sIL", "-m", "60", url],
        capture_output=True, text=True, check=True,
    ).stdout
    size = 0
    for line in out.splitlines():
        if line.lower().startswith("content-length:"):
            size = int(line.split(":")[1].strip())
    if size <= 0:
        raise RuntimeError(f"拿不到 Content-Length: {url}")
    return size


def download_part(url: str, part: Path, start: int, end: int, retries: int = 5) -> int:
    """下载 [start, end] 这一段到 part 文件，支持断点续传（-C -）。"""
    for attempt in range(retries):
        done = part.stat().st_size if part.exists() else 0
        if done >= end - start + 1:
            return done
        cmd = [
            "curl", "-sL", "--retry", "3", "--retry-delay", "2",
            "-r", f"{start + done}-{end}", "-C", "-",
            "-o", str(part), url,
        ]
        subprocess.run(cmd, capture_output=True)
        done = part.stat().st_size if part.exists() else 0
        if done >= end - start + 1:
            return done
        print(f"  [重试 {attempt + 1}/{retries}] {part.name} 已下 {done}/{end - start + 1}", flush=True)
    raise RuntimeError(f"分块下载失败: {part}")


def fetch(name: str, connections: int) -> None:
    url = f"{BASE}/{name}"
    out = RAW / name
    if out.exists() and out.stat().st_size > 0:
        print(f"[跳过] {name} 已存在 ({out.stat().st_size / 1e6:.1f} MB)")
        return

    total = head_length(url)
    print(f"[下载] {name}  {total / 1e9:.2f} GB  分 {connections} 块", flush=True)
    chunk = (total + connections - 1) // connections
    parts = []
    for i in range(connections):
        s = i * chunk
        e = min(total - 1, s + chunk - 1)
        if s > e:
            continue
        parts.append((RAW / f"{name}.part{i:02d}", s, e))

    with ThreadPoolExecutor(max_workers=connections) as pool:
        futs = {pool.submit(download_part, url, p, s, e): p for p, s, e in parts}
        for fut in as_completed(futs):
            fut.result()   # 任一分块失败就抛错，不做静默合并

    # 按序拼接成完整 zip
    with open(out, "wb") as fout:
        for p, _, _ in parts:
            with open(p, "rb") as fin:
                while True:
                    buf = fin.read(8 << 20)
                    if not buf:
                        break
                    fout.write(buf)
            p.unlink()
    print(f"[完成] {name}  {out.stat().st_size / 1e9:.2f} GB", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--connections", type=int, default=12)
    ap.add_argument("--only", nargs="*", default=None, help="只下指定的文件名（子串匹配）")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    names = TARGETS
    if args.only:
        names = [n for n in TARGETS if any(k in n for k in args.only)]
    for n in names:
        fetch(n, args.connections)


if __name__ == "__main__":
    sys.exit(main())
