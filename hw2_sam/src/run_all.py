# -*- coding: utf-8 -*-
"""一键跑完作业 2 的全部实验（自动分段续跑 + 自动评测）。

阶段顺序：
    0. 预计算图像嵌入（E1/E2 共用缓存）
    1. E1 零样本：GT 框 / 扰动框 ±10% ±20% / 点提示 1·3·5
    2. E2 冻结编码器微调 -> 同一批提示再评一遍
    3. E3 LoRA 微调（最慢，分段跑）-> 同一批提示再评一遍
    4. 级联：用作业 1 的 U-Net 预测框当提示
    5. 出图

用法：
    python src/run_all.py
    python src/run_all.py --from e2          # 从某个阶段开始
    python src/run_all.py --only eval        # 只评测（权重已训练好）
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from common import CKPT_DIR, LOG_DIR, ROOT

PROMPTS = ["gt", "pert10", "pert20", "pert-10", "pert-20", "point1", "point3", "point5"]


def sh(cmd: list, desc: str = "", check: bool = False) -> int:
    print(f"\n>>> {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    r = subprocess.run([sys.executable] + cmd, cwd=str(ROOT))
    print(f"<<< {desc} 退出码 {r.returncode}，用时 {(time.time()-t0)/60:.1f} min")
    if check and r.returncode != 0:
        raise SystemExit(1)
    return r.returncode


def stage_precompute():
    for split in ("train", "val", "test"):
        sh([str(ROOT / "src" / "precompute_embeddings.py"), "--split", split],
           f"预计算 {split} 嵌入")


def eval_stage(stage: str, tag: str, prompts=PROMPTS, split="test", unet_tag=None):
    for p in prompts:
        cmd = [str(ROOT / "src" / "evaluate_sam.py"), "--stage", stage, "--tag", tag,
               "--prompt", p, "--split", split]
        if p.startswith("auto") and unet_tag:
            cmd += ["--unet-tag", unet_tag]
        sh(cmd, f"{stage}/{p}")


def train_loop(script: str, tag: str, extra: list, max_segments: int = 20):
    for seg in range(max_segments):
        resume = (CKPT_DIR / f"{tag}_last.pth").exists()
        cmd = [str(ROOT / "src" / script), "--tag", tag] + extra
        if resume:
            cmd += ["--resume"]
        print(f"\n[{tag}] 第 {seg+1} 段（resume={resume}）")
        t0 = time.time()
        r = subprocess.run([sys.executable] + cmd, cwd=str(ROOT))
        print(f"  段用时 {(time.time()-t0)/60:.1f} min，退出码 {r.returncode}")
        if r.returncode != 0:
            print(f"[失败] {tag}")
            return False
        # 训练脚本跑满 epoch 后会正常退出；这里用 CSV 行数判断是否结束
        csv = LOG_DIR / f"{tag}.csv"
        if csv.exists():
            lines = csv.read_text(encoding="utf-8").strip().split("\n")
            if len(lines) - 1 >= 10:      # 达到设定 epoch 数就停（默认 10/20）
                print(f"[完成] {tag} 已跑 {len(lines)-1} epoch")
                return True
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="precompute",
                    choices=["precompute", "e1", "e2", "e3", "cascade", "figures"])
    ap.add_argument("--only", default=None, choices=["eval", "figures"])
    ap.add_argument("--unet-tag", default="unet_s42")
    args = ap.parse_args()

    order = ["precompute", "e1", "e2", "e3", "cascade", "figures"]
    if args.only == "figures":
        sh([str(ROOT / "src" / "plot_figures.py"), "--arch", "--robust", "--cascade"], "出图")
        return
    if args.only == "eval":
        eval_stage("e1", "e1")
        eval_stage("e2", "e2_box")
        eval_stage("e3", "e3_lora")
        return

    start_i = order.index(args.start)
    if start_i <= order.index("precompute"):
        stage_precompute()
    if start_i <= order.index("e1"):
        eval_stage("e1", "e1")
    if start_i <= order.index("e2"):
        train_loop("e2_finetune.py", "e2_box", ["--epochs", "20"])
        eval_stage("e2", "e2_box")
    if start_i <= order.index("e3"):
        # E3 要在线穿过 12 层 ViT，全量 2594 张需 2.9 h/epoch，
        # 这里用 600 张子集 × 6 epoch 控制总时长在 4 小时左右（报告中如实说明）
        train_loop("e3_lora.py", "e3_lora",
                   ["--epochs", "6", "--train-limit", "600", "--val-limit", "100"],
                   max_segments=40)
        eval_stage("e3", "e3_lora")
    if start_i <= order.index("cascade"):
        # 自动框级联：零样本 SAM + 微调后的 SAM 各跑一遍
        eval_stage("e1", "e1", prompts=["auto", "auto10"], unet_tag=args.unet_tag)
        eval_stage("e2", "e2_box", prompts=["auto", "auto10"], unet_tag=args.unet_tag)
    if start_i <= order.index("figures"):
        sh([str(ROOT / "src" / "plot_figures.py"), "--arch", "--robust", "--cascade"], "出图")


if __name__ == "__main__":
    main()
