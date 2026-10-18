# -*- coding: utf-8 -*-
"""E2：冻结 ViT 图像编码器，只训练 prompt encoder + mask decoder（MedSAM 式）。

**为什么这一阶段可以很快**：图像嵌入与提示无关，已由
`src/precompute_embeddings.py` 预计算并缓存；训练时只跑 prompt encoder +
mask decoder，每 epoch 从"十几分钟"降到"几十秒"。

训练时的提示扰动（很重要）：GT 框随机外扩/内缩 ±20%，逼模型学会在提示不准时
也能定位。这样后面"扰动框鲁棒性曲线"才不会崩。

用法：
    python src/e2_finetune.py --tag e2_box --epochs 20
    python src/e2_finetune.py --tag e2_box --resume
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# DataLoader 在 macOS 默认用 spawn 启动 worker，worker 会重新执行本脚本；
# 这里显式把本作业的 src 放到搜索首位，保证 `from common import …`
# 解析到 hw2_sam/src/common.py 而不是 hw1 的（hw1 的没有 EVAL_SIZE）。
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (CKPT_DIR, CSVLogger, EVAL_SIZE, LOG_DIR, autocast_ctx, get_device,
                    set_seed, supports_amp)
from sam_dataset import EmbeddingDataset
from losses import SegLoss
from metrics import MetricAccumulator
from sam_utils import build_sam, encode_and_decode, gt_box_from_mask, perturb_box, to_1024_box

FIELDS = ["epoch", "lr", "train_loss", "val_dice", "val_iou", "val_hd95", "epoch_sec", "cum_min"]


def flip_emb_and_box(emb: torch.Tensor, box: np.ndarray, h: bool, v: bool, size=EVAL_SIZE):
    """对图像嵌入与框做一致的翻转增强。

    图像嵌入是 (256,64,64) 的特征图，翻转它等价于翻转原图（因为 ViT 的
    patch 网格与图像空间一一对应）—— 这是缓存嵌入后唯一还能做的几何增强。
    """
    if h:
        emb = torch.flip(emb, dims=[-1])
        box = np.array([size - 1 - box[2], box[1], size - 1 - box[0], box[3]], dtype=np.float32)
    if v:
        emb = torch.flip(emb, dims=[-2])
        box = np.array([box[0], size - 1 - box[3], box[2], size - 1 - box[1]], dtype=np.float32)
    return emb, box


def make_batch(batch, rng, train: bool):
    """把 Dataset 采样结果组装成可直接喂给 SAM 的批。"""
    embs, masks, ids = batch
    embs = torch.stack([e.float() for e in embs])
    boxes = []
    for m in masks:
        m = m.numpy().astype(np.uint8)
        b = gt_box_from_mask(m)
        if b is None:
            b = np.array([0, 0, EVAL_SIZE - 1, EVAL_SIZE - 1], dtype=np.float32)
        if train:
            # 训练时给提示加噪声：±20% 随机扰动（鲁棒性的关键）
            b = perturb_box(b, float(rng.uniform(-0.2, 0.2)), size=EVAL_SIZE, rng=rng)
        boxes.append(b)
    boxes = np.stack(boxes).astype(np.float32)
    gt = torch.stack([torch.from_numpy(m.numpy().astype(np.float32)) for m in masks])[:, None]
    return embs, boxes, gt


def run_epoch(sam, loader, optimizer, criterion, device, amp, train, rng, epoch_desc=""):
    if train:
        sam.mask_decoder.train()
        sam.prompt_encoder.train()
    else:
        sam.mask_decoder.eval()
        sam.prompt_encoder.eval()
    losses, acc = [], MetricAccumulator()
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc=epoch_desc, leave=False):
            embs, boxes, gt = make_batch(batch, rng, train)
            B = embs.shape[0]
            # 随机翻转增强（同一张图内所有样本一致，避免额外复杂度）
            if train:
                do_h, do_v = bool(rng.integers(0, 2)), bool(rng.integers(0, 2))
                if do_h or do_v:
                    new_embs, new_boxes = [], []
                    for i in range(B):
                        e, b = flip_emb_and_box(embs[i], boxes[i], do_h, do_v)
                        new_embs.append(e); new_boxes.append(b)
                    embs = torch.stack(new_embs); boxes = np.stack(new_boxes)
            embs = embs.to(device)
            gt = gt.to(device)
            boxes1024 = np.stack([to_1024_box(b) for b in boxes])
            with autocast_ctx(device, enabled=amp):
                low_res, _ = encode_and_decode(sam, embs, box_1024=boxes1024)
                logits = low_res.float()
                if logits.shape[-1] != EVAL_SIZE:
                    import torch.nn.functional as F
                    logits = F.interpolate(logits, (EVAL_SIZE, EVAL_SIZE), mode="bilinear",
                                           align_corners=False)
                loss = criterion(logits, gt)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            pred = (torch.sigmoid(logits.float()) > 0.5).cpu().numpy()[:, 0].astype(np.uint8)
            g = gt.cpu().numpy()[:, 0].astype(np.uint8)
            for p, gg in zip(pred, g):
                acc.add(p, gg, compute_hd=not train)      # 训练时不算 HD95，太慢
    return float(np.mean(losses)), acc.summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="e2_box")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=25.0)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    amp = supports_amp(device)
    print(f"[设备] {device}  AMP={'fp16' if amp else 'off'}")

    sam = build_sam(device)
    # 冻结图像编码器 —— 本阶段的核心设定
    for p in sam.image_encoder.parameters():
        p.requires_grad_(False)
    trainable = sum(p.numel() for p in sam.parameters() if p.requires_grad)
    total = sum(p.numel() for p in sam.parameters())
    print(f"[参数] 可训练 {trainable/1e6:.2f} M / 总计 {total/1e6:.2f} M "
          f"({trainable/total*100:.1f}%)")

    train_ds = EmbeddingDataset("train", augment=True)
    val_ds = EmbeddingDataset("val")
    g = torch.Generator(); g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, generator=g, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers)

    params = list(sam.mask_decoder.parameters()) + list(sam.prompt_encoder.parameters())
    optimizer = AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2)
    criterion = SegLoss(0.5, 0.5)

    ck_last = CKPT_DIR / f"{args.tag}_last.pth"
    ck_best = CKPT_DIR / f"{args.tag}_best.pth"
    logger = CSVLogger(LOG_DIR / f"{args.tag}.csv", FIELDS, append=args.resume)
    rng = np.random.default_rng(args.seed)

    start, best, history = 1, -1.0, []
    if args.resume and ck_last.exists():
        ck = torch.load(ck_last, map_location=device, weights_only=False)
        sam.mask_decoder.load_state_dict(ck["mask_decoder"])
        sam.prompt_encoder.load_state_dict(ck["prompt_encoder"])
        optimizer.load_state_dict(ck["optim"]); scheduler.load_state_dict(ck["sched"])
        start, best, history = ck["epoch"] + 1, ck.get("best_dice", -1.0), ck.get("history", [])
        print(f"[续跑] 从 epoch {start} 继续，最佳 Dice={best:.4f}")

    t0 = time.time()
    for epoch in range(start, args.epochs + 1):
        ep_t = time.time()
        tl, _ = run_epoch(sam, train_loader, optimizer, criterion, device, amp, True, rng,
                          f"train {epoch}")
        _, vs = run_epoch(sam, val_loader, None, criterion, device, amp, False, rng,
                          f"val {epoch}")
        scheduler.step()
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_loss": tl,
               "val_dice": vs["dice"], "val_iou": vs["iou"], "val_hd95": vs["hd95"],
               "epoch_sec": round(time.time() - ep_t, 1),
               "cum_min": round((time.time() - t0) / 60, 1)}
        logger.log(row); history.append(row)
        print(f"  epoch {epoch:3d} | loss {tl:.4f} | dice {vs['dice']:.4f} | "
              f"iou {vs['iou']:.4f} | hd95 {vs['hd95']:.2f} | {row['epoch_sec']}s")

        state = {"epoch": epoch, "mask_decoder": sam.mask_decoder.state_dict(),
                 "prompt_encoder": sam.prompt_encoder.state_dict(),
                 "optim": optimizer.state_dict(), "sched": scheduler.state_dict(),
                 "best_dice": best, "history": history, "args": vars(args)}
        torch.save(state, ck_last)
        if vs["dice"] > best:
            best = vs["dice"]; state["best_dice"] = best
            torch.save(state, ck_best)
            print(f"    ↑ 最佳 Dice={best:.4f}")
        if row["cum_min"] >= args.max_minutes:
            print(f"[时间预算] 已跑 {row['cum_min']} 分钟，落盘。续跑加 --resume")
            break
    logger.close()
    print(f"[结束] {args.tag}  最佳 Dice={best:.4f}")


if __name__ == "__main__":
    main()
