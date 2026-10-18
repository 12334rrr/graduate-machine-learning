# -*- coding: utf-8 -*-
"""预计算并缓存 SAM 的 image embedding（E1 与 E2 共用）。

冻结 image encoder 时，图像嵌入与提示无关 —— 这是本作业能在本机跑完的关键：
- 2594 张训练图 × 256×64×64 × fp16(2B) ≈ 5.2 GB
- E2 训练每 epoch 从"十几分钟"降到"几十秒"

用法：
    python src/precompute_embeddings.py --split train
    python src/precompute_embeddings.py --split all
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common import EMB_DIR, EVAL_SIZE, HW1_PROC, autocast_ctx, get_device, set_seed
from sam_dataset import ISICRaw
from sam_utils import build_sam, preprocess_image


@torch.no_grad()
def run(split: str, device, batch: int = 1, limit: int = None):
    sam = build_sam(device)
    sam.eval()
    out_dir = EMB_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = ISICRaw(split, size=EVAL_SIZE, data_dir=HW1_PROC)
    n = len(ds) if limit is None else min(limit, len(ds))
    print(f"[预计算] split={split} 共 {n} 张 -> {out_dir}")

    t0 = time.time()
    done = 0
    for i in tqdm(range(n), desc=f"embed {split}"):
        sid, img, _ = ds[i]
        f = out_dir / f"{sid}.npy"
        if f.exists():
            done += 1
            continue
        x = preprocess_image(img, sam, device)
        with autocast_ctx(device, enabled=True):
            emb = sam.image_encoder(x)          # (1,256,64,64)
        emb = emb.float().cpu().numpy()[0].astype(np.float16)
        np.save(f, emb)
        done += 1
        if (i + 1) % 200 == 0:
            torch.mps.empty_cache() if device.type == "mps" else None
            print(f"  已处理 {i+1}/{n}，用时 {(time.time()-t0)/60:.1f} min")
    print(f"[完成] {split}: {done} 张，用时 {(time.time()-t0)/60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    set_seed(42)
    device = get_device()
    print(f"[设备] {device}")
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for s in splits:
        run(s, device, args.batch, args.limit)


if __name__ == "__main__":
    main()
