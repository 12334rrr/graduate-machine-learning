# -*- coding: utf-8 -*-
"""hw1 五折交叉验证（只跑主实验里最好的 2 组超参）。

划分方式：把官方 train(2594) + val(100) 合并为 2694 张，用固定种子做确定性五折；
每折用 4 折训练、留出 1 折评测。官方 test(1000) 完全不参与，保持与主实验可比。

用法：
    python src/run_cv.py                 # 2 模型 × 5 折（自动分段续跑 + 自动评测）
    python src/run_cv.py --skip-eval     # 只训练
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

from common import CKPT_DIR, LOG_DIR, REPORT_DIR, ROOT

EPOCHS = 12
FOLDS = 5
# 主实验测试集 Dice：Attention U-Net 0.6080 > ResUNet 0.6077 > U-Net 0.5939
CONFIGS = [
    {"model": "attn_unet", "seed": 42, "prefix": "cv_attn_unet"},
    {"model": "resunet", "seed": 42, "prefix": "cv_resunet"},
]


def fold_tag(prefix: str, fold: int) -> str:
    return f"{prefix}_f{fold}"


def train_cmd(cfg: dict, fold: int, resume: bool) -> list:
    cmd = [sys.executable, str(ROOT / "src" / "train.py"),
           "--model", cfg["model"], "--seed", str(cfg["seed"]),
           "--tag", fold_tag(cfg["prefix"], fold),
           "--max-minutes", "25", "--epochs", str(EPOCHS), "--patience", "6",
           "--fold", str(fold), "--folds", str(FOLDS)]
    if resume:
        cmd += ["--resume"]
    return cmd


def is_finished(tag: str) -> bool:
    p = LOG_DIR / f"{tag}_summary.json"
    if not p.exists():
        return False
    try:
        reason = json.loads(p.read_text(encoding="utf-8")).get("stop_reason", "")
    except Exception:
        return False
    return "time_budget" not in reason


def run_one(cfg: dict, fold: int, max_segments: int = 12) -> bool:
    tag = fold_tag(cfg["prefix"], fold)
    print(f"\n{'='*70}\n[五折] {tag}  模型={cfg['model']}  fold={fold}\n{'='*70}")
    for seg in range(max_segments):
        if is_finished(tag):
            print(f"[跳过] {tag} 已完成")
            return True
        resume = (CKPT_DIR / f"{tag}_last.pth").exists()
        cmd = train_cmd(cfg, fold, resume)
        print(f"[运行 {seg+1}/{max_segments}] {' '.join(cmd[1:])}")
        t0 = time.time()
        r = subprocess.run(cmd, cwd=str(ROOT))
        print(f"  用时 {(time.time()-t0)/60:.1f} min，退出码 {r.returncode}")
        if r.returncode != 0:
            print(f"[失败] {tag} 退出码 {r.returncode}")
            return False
    return is_finished(tag)


def eval_one(cfg: dict, fold: int) -> None:
    tag = fold_tag(cfg["prefix"], fold)
    cmd = [sys.executable, str(ROOT / "src" / "evaluate.py"),
           "--tag", tag, "--split", "val", "--size", "256",
           "--fold", str(fold), "--folds", str(FOLDS)]
    print(f"[评测] {tag} @ 留出折 fold{fold}")
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        print(f"[评测失败] {tag}")


def summarize() -> dict:
    """汇总 eval_cv_*_fold*.json，输出每模型每折指标与均值±标准差。"""
    out = {}
    for cfg in CONFIGS:
        rows = []
        for fold in range(FOLDS):
            p = LOG_DIR / f"eval_{fold_tag(cfg['prefix'], fold)}_fold{fold}.json"
            if not p.exists():
                continue
            m = json.loads(p.read_text(encoding="utf-8"))["metrics"]
            if not all(isinstance(m.get(k), (int, float)) and math.isfinite(float(m[k]))
                       for k in ("dice", "iou", "hd95")):
                print(f"[跳过] fold {fold} 指标含 NaN/缺失")
                continue
            rows.append({"fold": fold, "n": m.get("n"),
                         "dice": m["dice"], "iou": m["iou"], "hd95": m["hd95"]})
        if not rows:
            continue
        def mean_std(key):
            vals = [r[key] for r in rows
                    if isinstance(r[key], (int, float)) and math.isfinite(float(r[key]))]
            if not vals:
                return float("nan"), float("nan")
            return (statistics.mean(vals),
                    statistics.stdev(vals) if len(vals) > 1 else 0.0)
        dice_m, dice_s = mean_std("dice")
        iou_m, iou_s = mean_std("iou")
        hd_m, hd_s = mean_std("hd95")
        out[cfg["prefix"]] = {
            "model": cfg["model"], "n_folds": len(rows), "folds": rows,
            "dice_mean": dice_m, "dice_std": dice_s,
            "iou_mean": iou_m, "iou_std": iou_s,
            "hd95_mean": hd_m, "hd95_std": hd_s,
        }
    return out


def write_summary(res: dict) -> None:
    (LOG_DIR.parent / "cv_results.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# 五折交叉验证结果（自动生成）\n",
             "> 划分：官方 train+val 合并为 2,694 张，固定种子五折；官方 test 不参与。\n",
             "| 模型 | 折数 | Dice (均值±标准差) | IoU | HD95 (px) |",
             "| --- | --- | --- | --- | --- |"]
    CN = {"attn_unet": "Attention U-Net", "resunet": "ResUNet", "unet": "U-Net"}
    for key, v in res.items():
        lines.append(f"| {CN.get(v['model'], v['model'])} | {v['n_folds']} | "
                     f"{v['dice_mean']:.4f} ± {v['dice_std']:.4f} | "
                     f"{v['iou_mean']:.4f} | {v['hd95_mean']:.2f} ± {v['hd95_std']:.2f} |")
    lines.append("\n## 逐折明细\n")
    lines.append("| 模型 | fold | n | Dice | IoU | HD95 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for key, v in res.items():
        for r in v["folds"]:
            lines.append(f"| {CN.get(v['model'], v['model'])} | {r['fold']} | {r['n']} | "
                         f"{r['dice']:.4f} | {r['iou']:.4f} | {r['hd95']:.2f} |")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "cv_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[保存] {LOG_DIR.parent / 'cv_results.json'}")
    print(f"[保存] {REPORT_DIR / 'cv_results.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    for cfg in CONFIGS:
        for fold in range(FOLDS):
            ok = run_one(cfg, fold)
            if ok and not args.skip_eval:
                eval_one(cfg, fold)

    write_summary(summarize())


if __name__ == "__main__":
    main()
