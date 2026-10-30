# -*- coding: utf-8 -*-
"""E3：在 SAM 的 ViT 注意力 q/v 投影上注入 LoRA（r=8），连同 decoder 训练（SAMed 式）。

与 E2 的唯一差别就是"ViT 主干松不松绑"，所以超参、数据、提示策略全部对齐，
两张表放在一起才能干净地回答：**松绑主干到底值不值这个代价**。

代价（必须写进报告）：
- 图像嵌入不能再用缓存（每步都要前向+反向穿过 12 层 ViT）；
- bs=1 + 梯度累积，单 epoch 约 20~30 分钟，10 epoch ≈ 数小时，**必须分段跑**。

用法：
    python src/e3_lora.py --tag e3_lora --epochs 10
    python src/e3_lora.py --tag e3_lora --resume
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 与 e2_finetune.py 同理：spawn worker 重新执行本脚本时，
# 显式把本作业的 src 放到搜索首位，防止 `from common import …` 解析到 hw1 的 common。
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (CKPT_DIR, CSVLogger, EVAL_SIZE, LOG_DIR, autocast_ctx, get_device, write_progress,
                    set_seed, supports_amp)
from sam_dataset import ISICRaw
from losses import SegLoss
from lora import apply_lora_to_sam, collect_lora_params, count_lora_params, lora_state_dict
from metrics import MetricAccumulator
from sam_utils import (build_sam, encode_and_decode, gt_box_from_mask, perturb_box,
                       preprocess_image, to_1024_box)

FIELDS = ["epoch", "lr", "train_loss", "val_dice", "val_iou", "val_hd95", "epoch_sec", "cum_min"]


def collate(batch):
    ids = [b[0] for b in batch]
    imgs = [b[1] for b in batch]
    masks = torch.from_numpy(np.stack([b[2] for b in batch]).astype(np.float32))[:, None]
    return ids, imgs, masks


def forward_batch(sam, imgs, boxes, device, amp):
    """在线跑完整的 SAM：ViT 编码 -> 提示编码 -> mask decoder。"""
    xs = torch.cat([preprocess_image(im, sam, device) for im in imgs], dim=0)
    with autocast_ctx(device, enabled=amp):
        emb = sam.image_encoder(xs)
        boxes1024 = np.stack([to_1024_box(b) for b in boxes])
        low_res, _ = encode_and_decode(sam, emb, box_1024=boxes1024)
        logits = low_res.float()
    if logits.shape[-1] != EVAL_SIZE:
        logits = F.interpolate(logits, (EVAL_SIZE, EVAL_SIZE), mode="bilinear",
                               align_corners=False)
    return logits


def run_epoch(sam, loader, optimizer, criterion, device, amp, train, rng, accum, desc=""):
    sam.train() if train else sam.eval()
    losses, acc, step = [], MetricAccumulator(), 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i, (ids, imgs, masks) in enumerate(tqdm(loader, desc=desc, leave=False)):
            boxes = []
            for m in masks.numpy()[:, 0]:
                b = gt_box_from_mask(m.astype(np.uint8))
                if b is None:
                    b = np.array([0, 0, EVAL_SIZE - 1, EVAL_SIZE - 1], dtype=np.float32)
                if train:
                    b = perturb_box(b, float(rng.uniform(-0.2, 0.2)), size=EVAL_SIZE, rng=rng)
                boxes.append(b)
            logits = forward_batch(sam, imgs, boxes, device, amp)
            gt = masks.to(device)
            loss = criterion(logits, gt) / accum
            if train:
                loss.backward()
                step += 1
                if step % accum == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item() * accum)
            pred = (torch.sigmoid(logits.float()) > 0.5).cpu().numpy()[:, 0].astype(np.uint8)
            g = gt.cpu().numpy()[:, 0].astype(np.uint8)
            for p, gg in zip(pred, g):
                acc.add(p, gg, compute_hd=not train)
            if device.type == "mps":
                torch.mps.empty_cache()
    return float(np.mean(losses)), acc.summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="e3_lora")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8, help="梯度累积，等效 batch = batch×accum")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=25.0)
    ap.add_argument("--val-limit", type=int, default=100)
    ap.add_argument("--train-limit", type=int, default=0,
                    help=">0 时每 epoch 只用前 N 张训练图。E3 要在线穿过 12 层 ViT，"
                         "本机实测约 4 s/张，全量 2594 张需要 2.9 h/epoch，"
                         "因此正式实验用子集控制单 epoch 时长")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    amp = supports_amp(device)
    print(f"[设备] {device}  AMP={'fp16' if amp else 'off'}  LoRA r={args.lora_r}")

    sam = apply_lora_to_sam(build_sam(device), r=args.lora_r, alpha=args.lora_alpha)
    # 冻结 ViT 原始权重；只训练 LoRA + mask decoder + prompt encoder
    for n, p in sam.image_encoder.named_parameters():
        p.requires_grad_(False)
    lora_params = collect_lora_params(sam)
    for p in lora_params:
        p.requires_grad_(True)
    params = lora_params + list(sam.mask_decoder.parameters()) + list(sam.prompt_encoder.parameters())
    trainable = sum(p.numel() for p in params)
    total = sum(p.numel() for p in sam.parameters())
    print(f"[参数] 可训练 {trainable/1e6:.2f} M（其中 LoRA {count_lora_params(sam)/1e6:.3f} M）"
          f" / 总计 {total/1e6:.2f} M ({trainable/total*100:.2f}%)")

    g = torch.Generator(); g.manual_seed(args.seed)
    train_ds = ISICRaw("train", EVAL_SIZE)
    if args.train_limit > 0:
        train_ds = torch.utils.data.Subset(train_ds, range(min(args.train_limit, len(train_ds))))
        print(f"[子集] E3 每 epoch 只用 {len(train_ds)} 张训练图（算力限制）")
    val_ds = ISICRaw("val", EVAL_SIZE)
    if args.val_limit > 0:
        val_ds = torch.utils.data.Subset(val_ds, range(min(args.val_limit, len(val_ds))))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate, generator=g, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            collate_fn=collate, num_workers=0)

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
        sam.load_state_dict(ck["sam_lora"], strict=False)
        sam.mask_decoder.load_state_dict(ck["mask_decoder"])
        sam.prompt_encoder.load_state_dict(ck["prompt_encoder"])
        optimizer.load_state_dict(ck["optim"]); scheduler.load_state_dict(ck["sched"])
        start, best, history = ck["epoch"] + 1, ck.get("best_dice", -1.0), ck.get("history", [])
        print(f"[续跑] 从 epoch {start} 继续，最佳 Dice={best:.4f}")

    t0 = time.time()
    for epoch in range(start, args.epochs + 1):
        ep_t = time.time()
        tl, _ = run_epoch(sam, train_loader, optimizer, criterion, device, amp, True, rng,
                          args.accum, f"train {epoch}")
        _, vs = run_epoch(sam, val_loader, None, criterion, device, amp, False, rng,
                          args.accum, f"val {epoch}")
        scheduler.step()
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_loss": tl,
               "val_dice": vs["dice"], "val_iou": vs["iou"], "val_hd95": vs["hd95"],
               "epoch_sec": round(time.time() - ep_t, 1),
               "cum_min": round((time.time() - t0) / 60, 1)}
        logger.log(row); history.append(row)
        write_progress(args.tag, phase="train", epoch=epoch, epochs_total=args.epochs,
                       detail=f"epoch {epoch}/{args.epochs} · val dice {vs['dice']:.4f}")
        print(f"  epoch {epoch:3d} | loss {tl:.4f} | dice {vs['dice']:.4f} | "
              f"hd95 {vs['hd95']:.2f} | {row['epoch_sec']:.0f}s")

        improved = vs["dice"] > best
        best = max(best, vs["dice"])
        state = {"epoch": epoch, "sam_lora": lora_state_dict(sam),
                 "mask_decoder": sam.mask_decoder.state_dict(),
                 "prompt_encoder": sam.prompt_encoder.state_dict(),
                 "optim": optimizer.state_dict(), "sched": scheduler.state_dict(),
                 "best_dice": best, "history": history, "args": vars(args),
                 "lora_r": args.lora_r, "lora_alpha": args.lora_alpha}
        torch.save(state, ck_last)
        if improved:
            torch.save(state, ck_best)
            print(f"    ↑ 最佳 Dice={best:.4f}")
        if row["cum_min"] >= args.max_minutes:
            print(f"[时间预算] 已跑 {row['cum_min']} 分钟，落盘。续跑加 --resume")
            break
    logger.close()
    print(f"[结束] {args.tag}  最佳 Dice={best:.4f}")
    write_progress(args.tag, phase="done", detail="finished")


if __name__ == "__main__":
    main()
