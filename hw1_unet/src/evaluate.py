# -*- coding: utf-8 -*-
"""测试集评测 + 案例四联图 + 导出预测框（给作业 2 的级联实验用）。

用法：
    python src/evaluate.py --tag unet_s42 --split test
    python src/evaluate.py --tag unet_s42 --split val

产出：
    outputs/logs/eval_{tag}_{split}.json      全套指标 + 空掩膜统计
    outputs/figures/case_{tag}_{split}.png    案例四联图（好/中/差三例 × 原图/金标准/预测/差异）
    outputs/predictions/{tag}_{split}_boxes.json  每张图预测掩膜的外接矩形（级联提示用）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from common import (CKPT_DIR, FIG_DIR, LOG_DIR, ROOT, autocast_ctx, get_device,
                    set_seed, supports_amp)
from dataset import ISICSeg
from metrics import MetricAccumulator, dice_score
from models import build_model, count_params

PRED_DIR = ROOT / "outputs" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)


def bbox_from_mask(m: np.ndarray):
    """二值掩膜 -> (x0, y0, x1, y1)，闭区间。空掩膜返回 None。"""
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


@torch.no_grad()
def run_eval(model, ds, device, amp=True, batch_size=8, save_masks=False):
    model.eval()
    acc = MetricAccumulator()
    per_sample = []
    boxes = {}
    masks_out = {}
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    for img, mask, ids in loader:
        img = img.to(device)
        with autocast_ctx(device, enabled=amp):
            logits = model(img)
        prob = torch.sigmoid(logits.float()).cpu().numpy()[:, 0]
        gt = mask.numpy()[:, 0].astype(np.uint8)
        pred = (prob > 0.5).astype(np.uint8)
        for i, sid in enumerate(ids):
            d = dice_score(pred[i], gt[i])
            acc.add(pred[i], gt[i], compute_hd=True)
            per_sample.append({"id": sid, "dice": float(d)})
            boxes[sid] = bbox_from_mask(pred[i])
            if save_masks:
                masks_out[sid] = pred[i]
    return acc.summary(), per_sample, boxes, masks_out


def plot_cases(model_tag, split, ds, per_sample, pred_masks, save_path: Path):
    """画案例四联图：取 Dice 最好 / 中位 / 最差三个样本，每行 原图|金标准|预测|差异。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Heiti TC", "Songti SC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    order = sorted(per_sample, key=lambda r: r["dice"])
    picks = [("最佳", order[-1]), ("中位", order[len(order) // 2]), ("最差", order[0])]

    fig, axes = plt.subplots(3, 4, figsize=(13, 10))
    for r, (label, rec) in enumerate(picks):
        idx = ds.ids.index(rec["id"])
        img_raw, gt_raw = ds.raw_pair(idx)                         # 384×384 母版
        img = np.asarray(img_raw).astype(np.uint8)
        gt = (np.asarray(gt_raw) > 127).astype(np.uint8)
        pd = pred_masks[rec["id"]]                                 # 评测分辨率（--size，默认 256）
        # 预测是在 --size 分辨率下算的（指标也在该分辨率下统计），展示时上采样回母版尺寸，
        # 否则与 img/gt 尺寸不一致会 broadcast 报错；最近邻插值不引入新的灰度值。
        if pd.shape != gt.shape:
            pd = cv2.resize(pd, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        diff = np.zeros((*gt.shape, 3), dtype=np.uint8)
        diff[..., 1] = 255 * ((gt == 1) & (pd == 0))    # 绿色 = 漏检（GT 有、预测无）
        diff[..., 0] = 255 * ((gt == 0) & (pd == 1))    # 红色 = 误检
        diff[..., 2] = 255 * ((gt == 1) & (pd == 1))    # 蓝色 = 命中

        axes[r, 0].imshow(img); axes[r, 0].set_title(f"{label}样本 · 原图\n{rec['id']}", fontsize=10)
        axes[r, 1].imshow(gt, cmap="gray"); axes[r, 1].set_title("金标准掩膜", fontsize=10)
        axes[r, 2].imshow(pd, cmap="gray")
        axes[r, 2].set_title(f"预测掩膜\nDice={rec['dice']:.3f}", fontsize=10)
        axes[r, 3].imshow(diff); axes[r, 3].set_title("差异（红=误检 绿=漏检 蓝=命中）", fontsize=10)
        for c in range(4):
            axes[r, c].axis("off")
    fig.suptitle(f"{model_tag} @ {split} 测试案例", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[图] 已存 {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="训练时的实验标签")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--ckpt", default="best", choices=["best", "last"])
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--save-boxes", action="store_true", default=True)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--fold", type=int, default=None,
                    help="五折交叉验证的折号 0-4（在 train+val 合并划分的留出折上评测）")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    set_seed(42)
    device = get_device()
    amp = (not args.no_amp) and supports_amp(device)
    ckpt_path = CKPT_DIR / f"{args.tag}_{args.ckpt}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到权重 {ckpt_path}")

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = ck.get("args", {})
    # 与 train.py:121 同一套条件传参：只有 Attention U-Net 接受 attn_where，
    # 无条件传会让 UNet / ResUNet 构造直接 TypeError（评测链路跑不起来）。
    mname = model_args.get("model", "unet")
    kw = {"attn_where": model_args.get("attn_where", "all")} if mname == "attn_unet" else {}
    model = build_model(mname, **kw).to(device)
    model.load_state_dict(ck["model"])
    total, _ = count_params(model)
    print(f"[评测] {args.tag} @ {args.split}  权重={ckpt_path.name}  参数={total/1e6:.2f}M  设备={device}")

    # 五折模式：在 train+val 合并划分的第 fold 折（留出折）上评测
    split_label = f"fold{args.fold}" if args.fold is not None else args.split
    ds_split = "val" if args.fold is not None else args.split
    ds = ISICSeg(ds_split, size=args.size, augment=False,
                 fold=args.fold, folds=args.folds)
    summary, per_sample, boxes, pred_masks = run_eval(model, ds, device, amp, args.batch, save_masks=True)

    # 画案例图（需要重跑一遍拿掩膜，上面已经存了 pred_masks，直接用）
    plot_cases(args.tag, split_label, ds, per_sample, pred_masks,
               FIG_DIR / f"case_{args.tag}_{split_label}.png")

    if args.save_boxes:
        (PRED_DIR / f"{args.tag}_{split_label}_boxes.json").write_text(
            json.dumps(boxes), encoding="utf-8")

    out = {"tag": args.tag, "split": split_label, "ckpt": args.ckpt,
           "size": args.size, "params_M": round(total / 1e6, 2), "metrics": summary}
    (LOG_DIR / f"eval_{args.tag}_{split_label}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
