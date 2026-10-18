# -*- coding: utf-8 -*-
"""出图：三张模型结构图、训练曲线、消融对比柱状图。

用法：
    python src/plot_figures.py --arch                      # 三张结构图
    python src/plot_figures.py --curves unet_s42 resunet_s42 attn_s42
    python src/plot_figures.py --ablation outputs/ablation.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import FIG_DIR, LOG_DIR

plt.rcParams["font.sans-serif"] = ["Heiti TC", "Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

C_ENC = "#DCE9F7"
C_BOT = "#F7DCE4"
C_DEC = "#DFF0DC"
C_SKIP = "#9E9E9E"


def _box(ax, x, y, w, h, text, fc, fs=9):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.2, edgecolor="#37474F", facecolor=fc))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color="#102027")


def _arrow(ax, x1, y1, x2, y2, color="#37474F", style="-|>", lw=1.3, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
                                 linewidth=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))


def draw_arch(variant: str, save_path: Path):
    """画 U-Net 家族结构图。variant: unet / resunet / attn_unet"""
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 10)
    ax.axis("off")

    stages = [(64, "256×256"), (128, "128×128"), (256, "64×64"), (512, "32×32")]
    bot = (1024, "16×16")
    xs_enc = [1.4, 3.0, 4.6, 6.2]
    # 解码器横坐标由 bottleneck(7.6) 向右等距展开，下面 xs_dec_full 才是实际用的
    y_top, y_bot = 7.6, 1.6
    ys = [y_top - (y_top - y_bot) * i / 3 for i in range(4)]

    note = {"unet": "两次 3×3 卷积 + BN + ReLU",
            "resunet": "残差块：conv-BN-ReLU-conv-BN + 恒等相加",
            "attn_unet": "两次 3×3 卷积 + BN + ReLU"}[variant]
    title = {"unet": "U-Net（Ronneberger et al., MICCAI 2015）",
             "resunet": "ResUNet（Zhang et al., IEEE GRSL 2018）",
             "attn_unet": "Attention U-Net（Oktay et al., MIDL 2018）"}[variant]
    ax.text(0.05, 9.6, title, fontsize=13, ha="left", va="center", fontweight="bold")
    ax.text(0.05, 9.05, f"积木：{note}", fontsize=10, ha="left", va="center", color="#455A64")

    # 编码器
    for i, ((c, s), x, y) in enumerate(zip(stages, xs_enc, ys)):
        _box(ax, x, y, 1.25, 0.95, f"{c}ch\n{s}", C_ENC)
        if i > 0:
            _arrow(ax, xs_enc[i - 1], ys[i - 1] - 0.5, x, y + 0.5)
        if variant == "resunet":
            ax.text(x, y - 0.78, "残差块", fontsize=7.5, ha="center", color="#1B5E20")
    # bottleneck
    _box(ax, 7.6, 1.6, 1.45, 1.0, f"{bot[0]}ch\n{bot[1]}", C_BOT)
    _arrow(ax, xs_enc[-1] + 0.6, ys[-1] - 0.25, 7.6 - 0.72, 1.75)

    # 解码器（分辨率反向）
    xs_dec_full = [7.6 + 1.3, 7.6 + 2.7, 7.6 + 4.0, 7.6 + 5.2]
    ys_dec = [1.6 + 1.9, 1.6 + 3.4, 1.6 + 4.9, 1.6 + 6.0]
    dec_stages = [(512, "32×32"), (256, "64×64"), (128, "128×128"), (64, "256×256")]
    for i, ((c, s), x, y) in enumerate(zip(dec_stages, xs_dec_full, ys_dec)):
        _box(ax, x, y, 1.25, 0.95, f"{c}ch\n{s}", C_DEC)
        if i == 0:
            _arrow(ax, 7.6 + 0.72, 1.75, x - 0.62, y - 0.35)
        else:
            _arrow(ax, xs_dec_full[i - 1], ys_dec[i - 1] + 0.5, x, y - 0.5)
        if variant == "resunet":
            ax.text(x, y - 0.78, "残差块", fontsize=7.5, ha="center", color="#1B5E20")

    # 跳跃连接（concat）
    for i in range(4):
        _arrow(ax, xs_enc[i] + 0.62, ys[i], xs_dec_full[3 - i] - 0.62, ys_dec[3 - i],
               color=C_SKIP, ls="--", lw=1.1, rad=-0.25)
        mid_x = (xs_enc[i] + 0.62 + xs_dec_full[3 - i] - 0.62) / 2
        mid_y = (ys[i] + ys_dec[3 - i]) / 2 + 0.55
        if variant == "attn_unet":
            ax.plot([mid_x], [mid_y], marker="D", markersize=11, color="#FFF3E0",
                    markeredgecolor="#E65100", zorder=3)
            ax.text(mid_x, mid_y, "AG", ha="center", va="center", fontsize=7, color="#E65100")
        ax.text(mid_x, mid_y - 0.6, "concat", fontsize=7, ha="center", color=C_SKIP)

    # 输出
    _box(ax, xs_dec_full[-1] + 1.55, ys_dec[-1], 1.35, 0.8, "1×1 卷积\n1ch logits", "#FFF8E1")
    _arrow(ax, xs_dec_full[-1] + 0.62, ys_dec[-1], xs_dec_full[-1] + 1.55 - 0.68, ys_dec[-1])

    # 图例说明
    tips = {
        "unet": "跳跃连接 = 通道维 concat（不是 add），输出 1 通道不加 softmax",
        "resunet": "块内恒等相加（残差）+ 跨层 concat；下采样用 stride=2 卷积而非池化",
        "attn_unet": "每条 skip 先过注意力门 AG（gating 来自解码器上一层），再 concat",
    }[variant]
    ax.text(0.05, 0.55, tips, fontsize=9.5, ha="left", va="center", color="#37474F",
            bbox=dict(boxstyle="round,pad=0.35", fc="#F5F5F5", ec="#BDBDBD"))
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[结构图] {save_path}")


def plot_curves(tags, save_path: Path):
    """画训练/验证曲线（loss / dice / hd95）。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for tag in tags:
        p = LOG_DIR / f"{tag}.csv"
        if not p.exists():
            print(f"  [跳过] 缺日志 {p}")
            continue
        import csv as _csv
        rows = list(_csv.DictReader(p.open(encoding="utf-8")))
        ep = [int(r["epoch"]) for r in rows]
        for ax, key, title in zip(axes, ["train_loss", "dice", "hd95"],
                                  ["训练损失", "验证集 Dice", "验证集 HD95 (px)"]):
            vals = [float(r[key]) for r in rows if r[key] not in ("", None)]
            ax.plot(ep[:len(vals)], vals, marker="o", ms=3, label=tag)
            ax.set_title(title)
        if len(rows) and "val_loss" in rows[0]:
            vl = [float(r["val_loss"]) for r in rows if r["val_loss"]]
            axes[0].plot(ep[:len(vl)], vl, ls="--", alpha=0.5, label=f"{tag} (val)")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[曲线] {save_path}")


def plot_ablation(results: dict, save_path: Path):
    """消融柱状图。results 形如 {"损失函数": {"BCE": {...}, ...}, ...}"""
    groups = list(results.keys())
    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4), squeeze=False)
    for gi, g in enumerate(groups):
        ax = axes[0][gi]
        items = results[g]
        names = list(items.keys())
        dice = [items[k].get("dice", float("nan")) for k in names]
        hd = [items[k].get("hd95", float("nan")) for k in names]
        x = np.arange(len(names))
        b1 = ax.bar(x - 0.2, dice, 0.38, label="Dice", color="#2980b9")
        ax2 = ax.twinx()
        b2 = ax2.bar(x + 0.2, hd, 0.38, label="HD95 (px, 越低越好)", color="#c0392b")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel("Dice", color="#2980b9")
        ax2.set_ylabel("HD95 (px)", color="#c0392b")
        ax.set_title(g, fontsize=11)
        ax.set_ylim(0, 1.0)
        for b in b1:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{b.get_height():.3f}", ha="center", fontsize=7.5, color="#154360")
        for b in b2:
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                     f"{b.get_height():.1f}", ha="center", fontsize=7.5, color="#7B241C")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[消融图] {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", action="store_true")
    ap.add_argument("--curves", nargs="*", default=None)
    ap.add_argument("--ablation", default=None, help="消融结果 JSON 路径")
    args = ap.parse_args()

    if args.arch:
        for v in ("unet", "resunet", "attn_unet"):
            draw_arch(v, FIG_DIR / f"arch_{v}.png")
    if args.curves:
        plot_curves(args.curves, FIG_DIR / "training_curves.png")
    if args.ablation:
        plot_ablation(json.loads(Path(args.ablation).read_text(encoding="utf-8")),
                      FIG_DIR / "ablation.png")


if __name__ == "__main__":
    main()
