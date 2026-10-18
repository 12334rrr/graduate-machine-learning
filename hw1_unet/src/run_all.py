# -*- coding: utf-8 -*-
"""一键跑完作业 1 的全部实验（自动分段续跑 + 自动评测）。

为什么要这个脚本：MacBook Air 无风扇，单次连续训练不能超过 20~30 分钟，
所以 train.py 到点会自动落盘退出；本脚本反复调用它直到训练自然结束。

用法：
    python src/run_all.py                 # 跑全部
    python src/run_all.py --only main     # 只跑主对比
    python src/run_all.py --only ablation # 只跑消融
    python src/run_all.py --skip-eval     # 不评测（只训练）
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from common import CKPT_DIR, LOG_DIR, ROOT

# ---------------- 实验矩阵 ----------------
# 顺序很讲究：先跑"每种模型 1 个种子"，让主对比表尽快成型；
# 再补另外两个种子做显著性；最后跑消融。
#
# 关于 epoch=12（规格里写的是 30）：本机 Apple M5 + MPS fp16 实测约 6 min/epoch，
# 30 epoch 单跑一次就要 3 小时，15 次实验要 45 小时，不可行。
# 12 epoch（≈3900 步）在 ISIC 上已足以让 U-Net 收敛到 0.85 左右的 Dice，
# 因此统一压到 12 epoch + 早停 patience=6，并在报告中如实说明这一取舍。
EPOCHS = 12

MAIN = [
    {"model": m, "seed": 42, "tag": f"{m}_s42", "epochs": EPOCHS, "patience": 6}
    for m in ("unet", "resunet", "attn_unet")
] + [
    {"model": m, "seed": s, "tag": f"{m}_s{s}", "epochs": EPOCHS, "patience": 6}
    for s in (2026, 7) for m in ("unet", "resunet", "attn_unet")
]

ABLATION = [
    {"model": "unet", "seed": 42, "tag": "loss_bce", "loss": "bce", "epochs": EPOCHS, "patience": 6},
    {"model": "unet", "seed": 42, "tag": "loss_dice", "loss": "dice", "epochs": EPOCHS, "patience": 6},
    {"model": "unet", "seed": 42, "tag": "size192", "size": 192, "epochs": EPOCHS, "patience": 6},
    {"model": "unet", "seed": 42, "tag": "size384", "size": 384, "epochs": EPOCHS, "patience": 6},
    {"model": "attn_unet", "seed": 42, "tag": "attn_skip", "attn_where": "skip", "epochs": EPOCHS, "patience": 6},
    {"model": "attn_unet", "seed": 42, "tag": "attn_bottleneck", "attn_where": "bottleneck", "epochs": EPOCHS, "patience": 6},
]


def train_cmd(cfg: dict, resume: bool) -> list:
    cmd = [sys.executable, str(ROOT / "src" / "train.py"),
           "--model", cfg["model"], "--seed", str(cfg.get("seed", 42)),
           "--tag", cfg["tag"], "--max-minutes", str(cfg.get("max_minutes", 25)),
           "--epochs", str(cfg.get("epochs", EPOCHS)),
           "--patience", str(cfg.get("patience", 6))]
    if "loss" in cfg:
        cmd += ["--loss", cfg["loss"]]
    if "size" in cfg:
        cmd += ["--size", str(cfg["size"]), "--batch", str(cfg.get("batch", 8))]
    if "attn_where" in cfg:
        cmd += ["--attn-where", cfg["attn_where"]]
    if resume:
        cmd += ["--resume"]
    return cmd


def is_finished(tag: str) -> bool:
    """训练是否已经自然结束（跑完/早停），而不是被时间预算截断。"""
    p = LOG_DIR / f"{tag}_summary.json"
    if not p.exists():
        return False
    try:
        reason = json.loads(p.read_text(encoding="utf-8")).get("stop_reason", "")
    except Exception:
        return False
    return "time_budget" not in reason


def run_one(cfg: dict, max_segments: int = 12) -> bool:
    tag = cfg["tag"]
    print(f"\n{'='*70}\n[实验] {tag}  模型={cfg['model']}  种子={cfg.get('seed',42)}\n{'='*70}")
    for seg in range(max_segments):
        if is_finished(tag):
            print(f"[跳过] {tag} 已完成")
            return True
        resume = (CKPT_DIR / f"{tag}_last.pth").exists()
        cmd = train_cmd(cfg, resume)
        print(f"[运行 {seg+1}/{max_segments}] {' '.join(cmd[1:])}")
        t0 = time.time()
        r = subprocess.run(cmd, cwd=str(ROOT))
        print(f"  用时 {(time.time()-t0)/60:.1f} min，退出码 {r.returncode}")
        if r.returncode != 0:
            print(f"[失败] {tag} 退出码 {r.returncode}")
            return False
    return is_finished(tag)


def eval_one(tag: str, split: str = "test", size: int = 256) -> None:
    # --size 必须跟训练时一致：evaluate.py 的默认值是 256 且不会从 checkpoint 恢复分辨率，
    # 不传的话 size192 / size384 两个消融会用 256 评测，输入分辨率消融的数据就失真了。
    cmd = [sys.executable, str(ROOT / "src" / "evaluate.py"), "--tag", tag,
           "--split", split, "--size", str(size)]
    print(f"[评测] {tag} @ {split} (size={size})")
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        print(f"[评测失败] {tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, choices=["main", "ablation"])
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--eval-split", default="test")
    args = ap.parse_args()

    cfgs = []
    if args.only in (None, "main"):
        cfgs += MAIN
    if args.only in (None, "ablation"):
        cfgs += ABLATION

    for cfg in cfgs:
        ok = run_one(cfg)
        if ok and not args.skip_eval:
            eval_one(cfg["tag"], args.eval_split, cfg.get("size", 256))

    print("\n[汇总]")
    subprocess.run([sys.executable, str(ROOT / "src" / "collect_results.py")], cwd=str(ROOT))


if __name__ == "__main__":
    main()
