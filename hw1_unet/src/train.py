# -*- coding: utf-8 -*-
"""训练脚本（手写 U-Net / ResUNet / Attention U-Net 三模型通用）。

工程约束（照作业执行规格实现）：
- 设备 cuda → mps → cpu 三级回退，mps/cuda 下开 fp16 autocast；
- **每个 epoch 都存盘**（last.pth + best.pth），Air 无风扇，长跑必须能续；
- `--max-minutes` 到点自动收工并落盘，下次带 `--resume` 接着跑；
- 早停 patience=8，以验证集 Dice 为准；
- 所有随机性由 --seed 控制，关键对比要跑 3 个种子。

用法示例：
    python src/train.py --model unet --seed 42
    python src/train.py --model unet --loss bce --tag loss_bce
    python src/train.py --model unet --size 384 --tag size384
    python src/train.py --model attn_unet --attn-where skip --tag attn_skip
    python src/train.py --model unet --resume --tag main      # 续跑
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from common import (CKPT_DIR, LOG_DIR, CSVLogger, autocast_ctx, get_device,
                    set_seed, supports_amp)
from dataset import build_loaders
from losses import build_loss
from metrics import MetricAccumulator
from models import build_model, count_params

FIELDS = ["epoch", "lr", "train_loss", "val_loss", "dice", "iou", "hd95",
          "sensitivity", "specificity", "pixel_acc", "epoch_sec", "cum_min"]


@torch.no_grad()
def validate(model, loader, device, criterion, amp=True, compute_hd=True):
    """在验证/测试集上算全套指标。返回 (汇总dict, 平均loss)。"""
    model.eval()
    acc = MetricAccumulator()
    losses = []
    for img, mask, _ in loader:
        img, mask = img.to(device), mask.to(device)
        with autocast_ctx(device, enabled=amp):
            logits = model(img)
        losses.append(criterion(logits, mask).item())
        prob = torch.sigmoid(logits.float())
        pred = (prob > 0.5).cpu().numpy()[:, 0].astype(np.uint8)
        gt = mask.cpu().numpy()[:, 0].astype(np.uint8)
        for p, g in zip(pred, gt):
            acc.add(p, g, compute_hd=compute_hd)
    return acc.summary(), float(np.mean(losses))


def train_one_epoch(model, loader, optimizer, criterion, device, amp=True, scaler=None):
    model.train()
    running, n = 0.0, 0
    for img, mask, _ in loader:
        img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device, enabled=amp):
            logits = model(img)
            loss = criterion(logits, mask)
        # MPS 不支持 GradScaler，直接反向；CUDA 下用 scaler 更稳
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running += loss.item() * img.size(0)
        n += img.size(0)
    return running / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unet", choices=["unet", "resunet", "attn_unet"])
    ap.add_argument("--attn-where", default="all", choices=["all", "skip", "bottleneck", "none"])
    ap.add_argument("--size", type=int, default=256, help="输入分辨率 192/256/384")
    ap.add_argument("--loss", default="bce_dice", choices=["bce", "dice", "bce_dice"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="", help="实验标签，决定 checkpoint / 日志文件名")
    ap.add_argument("--resume", action="store_true", help="从 last.pth 续跑")
    ap.add_argument("--max-minutes", type=float, default=25.0,
                    help="单次连续训练时长上限（MacBook Air 无风扇，防降频/崩溃）")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--subset", type=int, default=0,
                    help=">0 时只用前 N 张训练图（快速诊断用，正式实验不要开）")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    amp = (not args.no_amp) and supports_amp(device)
    tag = args.tag or f"{args.model}_s{args.seed}"
    print(f"[设备] {device}  AMP={'fp16' if amp else 'off'}  标签={tag}")

    train_loader, val_loader, _ = build_loaders(
        size=args.size, batch_size=args.batch, num_workers=args.num_workers, seed=args.seed)
    if args.subset > 0:
        from dataset import ISICSeg
        from torch.utils.data import DataLoader
        sub = torch.utils.data.Subset(train_loader.dataset, range(min(args.subset, len(train_loader.dataset))))
        train_loader = DataLoader(sub, batch_size=args.batch, shuffle=True,
                                  num_workers=0, drop_last=True)

    kw = {"attn_where": args.attn_where} if args.model == "attn_unet" else {}
    model = build_model(args.model, **kw).to(device)
    total, trainable = count_params(model)
    print(f"[模型] {args.model}  参数 {total/1e6:.2f} M（可训练 {trainable/1e6:.2f} M）")

    criterion = build_loss(args.loss)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and amp else None

    ckpt_last = CKPT_DIR / f"{tag}_last.pth"
    ckpt_best = CKPT_DIR / f"{tag}_best.pth"
    log_path = LOG_DIR / f"{tag}.csv"
    logger = CSVLogger(log_path, FIELDS, append=args.resume)

    start_epoch, best_dice, bad_epochs, history = 1, -1.0, 0, []
    if args.resume and ckpt_last.exists():
        ck = torch.load(ckpt_last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optim"])
        scheduler.load_state_dict(ck["sched"])
        start_epoch = ck["epoch"] + 1
        best_dice = ck.get("best_dice", -1.0)
        bad_epochs = ck.get("bad_epochs", 0)
        history = ck.get("history", [])
        print(f"[续跑] 从第 {start_epoch} epoch 继续，历史最佳 Dice={best_dice:.4f}")

    t_start = time.time()
    stop_reason = "completed"
    for epoch in range(start_epoch, args.epochs + 1):
        ep_t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, amp, scaler)
        vsum, val_loss = validate(model, val_loader, device, criterion, amp, compute_hd=True)
        scheduler.step()
        ep_sec = time.time() - ep_t0
        cum_min = (time.time() - t_start) / 60.0

        row = {"epoch": epoch, "lr": lr_now, "train_loss": train_loss, "val_loss": val_loss,
               "dice": vsum["dice"], "iou": vsum["iou"], "hd95": vsum["hd95"],
               "sensitivity": vsum["sensitivity"], "specificity": vsum["specificity"],
               "pixel_acc": vsum["pixel_acc"], "epoch_sec": round(ep_sec, 1),
               "cum_min": round(cum_min, 1)}
        logger.log(row)
        history.append(row)
        print(f"  epoch {epoch:3d} | loss {train_loss:.4f}/{val_loss:.4f} | "
              f"dice {vsum['dice']:.4f} | iou {vsum['iou']:.4f} | hd95 {vsum['hd95']:.2f} | "
              f"{ep_sec:.0f}s")

        # —— 每个 epoch 都存盘（硬性要求）——
        state = {"epoch": epoch, "model": model.state_dict(), "optim": optimizer.state_dict(),
                 "sched": scheduler.state_dict(), "best_dice": best_dice,
                 "bad_epochs": bad_epochs, "history": history, "args": vars(args)}
        torch.save(state, ckpt_last)
        if vsum["dice"] > best_dice:
            best_dice = vsum["dice"]
            bad_epochs = 0
            state["best_dice"] = best_dice
            torch.save(state, ckpt_best)
            print(f"    ↑ 新的最佳 Dice={best_dice:.4f}，已存 {ckpt_best.name}")
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            stop_reason = f"early_stop(patience={args.patience})"
            print(f"[早停] 连续 {bad_epochs} 个 epoch 没提升，停止")
            break
        if cum_min >= args.max_minutes:
            stop_reason = f"time_budget({args.max_minutes}min)"
            print(f"[时间预算] 已跑 {cum_min:.1f} 分钟，先落盘。续跑：python src/train.py "
                  f"--model {args.model} --tag {tag} --resume")
            break

    logger.close()
    # "args" 是 collect_results.py 读取 model / seed / size / loss / attn_where 的来源
    # （checkpoint 里本来就存了 vars(args)，summary 之前漏写会导致汇总表全空）。
    (LOG_DIR / f"{tag}_summary.json").write_text(json.dumps(
        {"tag": tag, "model": args.model, "stop_reason": stop_reason,
         "best_dice": best_dice, "args": vars(args), "history": history},
        indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"[结束] {tag}  原因={stop_reason}  最佳验证 Dice={best_dice:.4f}")


if __name__ == "__main__":
    main()
