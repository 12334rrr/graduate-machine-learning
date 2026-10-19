# -*- coding: utf-8 -*-
"""作业 2 定性案例图：同一测试样本在 E1 / E2 / E3（GT 框提示）下的预测对比。

作业要求「展示一个测试样本的案例，标出真实标签和预测标签」。本脚本按 GT 前景面积
取 10% / 50% / 90% 分位，覆盖小 / 中 / 大病灶，各跑三个阶段并画四联图。

用法：
    python src/plot_cases.py --split test
产物：
    outputs/figures/case_sam_gt_test.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Heiti TC", "Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (EVAL_SIZE, FIG_DIR, autocast_ctx, get_device, set_seed,  # noqa: E402
                    supports_amp)
from evaluate_sam import get_embedding, load_stage_weights  # noqa: E402
from metrics import dice_score  # noqa: E402
from sam_dataset import ISICRaw  # noqa: E402
from sam_utils import (build_sam, encode_and_decode, gt_box_from_mask,  # noqa: E402
                       resize_logits_to_eval, to_1024_box)

STAGES = [("e1", "e1", "E1 零样本"),
          ("e2", "e2_box", "E2 冻结微调"),
          ("e3", "e3_lora", "E3 LoRA")]


def pick_samples(ds, n=3):
    """按 GT 前景面积分位选样本（小/中/大病灶），返回 [(idx, id, img, mask), …]。"""
    areas = np.array([int((np.asarray(ds[i][2]) > 0).sum()) for i in range(len(ds))])
    quantiles = np.linspace(0.1, 0.9, n) if n > 1 else np.array([0.5])
    idxs = []
    for q in quantiles:
        target = np.quantile(areas, q)
        idx = int(np.argmin(np.abs(areas - target)))
        while idx in idxs:
            idx = (idx + 1) % len(ds)
        idxs.append(idx)
    return [(i, *ds[i]) for i in idxs]


def run_stage(stage, tag, samples, split, device):
    """加载某阶段权重，对给定样本用 GT 框提示跑一遍，返回 {id: (pred, dice)}。"""
    sam = build_sam(device)
    sam = load_stage_weights(sam, stage, tag, device)
    sam.eval()
    use_cache = stage in ("e1", "e2")   # E3 改动了 ViT，只能在线前向
    out = {}
    with torch.no_grad():
        for _, sid, img, mask in samples:
            emb = get_embedding(sam, stage, split, sid, img, device, use_cache)
            box256 = gt_box_from_mask(mask)
            box1024 = to_1024_box(box256)[None]
            with autocast_ctx(device, enabled=supports_amp(device)):
                low_res, _ = encode_and_decode(sam, emb, box_1024=box1024)
            logits = resize_logits_to_eval(low_res, EVAL_SIZE)
            pred = (torch.sigmoid(logits.float()) > 0.5).cpu().numpy()[0, 0].astype(np.uint8)
            out[sid] = (pred, float(dice_score(pred, mask)))
            print(f"  {stage} {sid}: Dice={out[sid][1]:.4f}")
    del sam
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--num", type=int, default=3)
    args = ap.parse_args()

    set_seed(42)
    device = get_device()
    ds = ISICRaw(args.split, size=EVAL_SIZE)
    samples = pick_samples(ds, args.num)
    print(f"[样本] {[s[1] for s in samples]}")

    results = {stage: run_stage(stage, tag, samples, args.split, device)
               for stage, tag, _ in STAGES}

    n = len(samples)
    fig, axes = plt.subplots(n, 2 + len(STAGES), figsize=(4 * (2 + len(STAGES)), 3.4 * n))
    axes = np.atleast_2d(axes)
    for row, (_, sid, img, mask) in enumerate(samples):
        area = 100.0 * float(mask.mean())
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"{sid}\n原图（前景 {area:.1f}%）", fontsize=9)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("金标准掩膜", fontsize=9)
        axes[row, 1].axis("off")
        for col, (stage, _, label) in enumerate(STAGES, start=2):
            pred, score = results[stage][sid]
            axes[row, col].imshow(pred, cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_title(f"{label}\nDice={score:.3f}", fontsize=9)
            axes[row, col].axis("off")
    fig.suptitle(f"SAM 提示分割定性案例（{args.split}，GT 框提示）", fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / f"case_sam_gt_{args.split}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[保存] {out}")


if __name__ == "__main__":
    main()
