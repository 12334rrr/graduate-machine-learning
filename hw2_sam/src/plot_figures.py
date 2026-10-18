# -*- coding: utf-8 -*-
"""作业 2 出图：SAM 三件套结构图、mask decoder 数据流图、扰动框鲁棒性曲线、级联系统图。

用法：
    python src/plot_figures.py --arch
    python src/plot_figures.py --robust          # 读 outputs/logs/eval_*.json
    python src/plot_figures.py --cascade
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from common import FIG_DIR, LOG_DIR

plt.rcParams["font.sans-serif"] = ["Heiti TC", "Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

C_VIT = "#E3F2FD"
C_PE = "#FFF8E1"
C_MD = "#E8F5E9"
C_IO = "#FCE4EC"
EDGE = "#37474F"


def _box(ax, x, y, w, h, text, fc, fs=9, ec=EDGE):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.03,rounding_size=0.08",
                                linewidth=1.2, edgecolor=ec, facecolor=fc))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color="#102027")


def _arrow(ax, x1, y1, x2, y2, color=EDGE, ls="-", lw=1.4, rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
                                 linewidth=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))


def draw_sam_arch(save_path: Path):
    """SAM 三件套 + 数据流。"""
    fig, ax = plt.subplots(figsize=(13.5, 6.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(0.1, 9.5, "SAM（Segment Anything）结构：ViT 图像编码器 + 提示编码器 + 轻量掩膜解码器",
            fontsize=13, fontweight="bold")

    # 输入
    _box(ax, 0.9, 5.0, 1.3, 1.1, "输入图像\n1024×1024×3", C_IO)
    # Patch 切分
    _box(ax, 2.5, 5.0, 1.4, 1.1, "Patch 切分\n16×16 → 64×64\n共 4096 token", "#F3E5F5")
    _arrow(ax, 1.6, 5.0, 1.75, 5.0)

    # ViT
    _box(ax, 5.1, 5.0, 3.0, 2.6,
         "ViT-B 图像编码器（≈89 M 参数，占 SAM 97%）\n"
         "12 × Transformer Block：\n"
         "  LayerNorm → 多头自注意力 → 残差 →\n"
         "  LayerNorm → FFN → 残差\n"
         "（第 2/5/8/11 层用全局注意力，其余用窗口注意力）", C_VIT, fs=9)
    _arrow(ax, 3.25, 5.0, 3.55, 5.0)

    # 图像嵌入
    _box(ax, 8.3, 5.0, 1.7, 1.1, "图像嵌入\n256×64×64", "#E1F5FE")
    _arrow(ax, 6.65, 5.0, 7.4, 5.0)

    # 提示
    _box(ax, 5.1, 1.6, 3.0, 1.5, "提示编码器（≈0.01 M）\n框 / 点 / 文本 / 掩膜\n→ 稀疏嵌入 + 稠密嵌入", C_PE, fs=9)
    _arrow(ax, 5.1, 2.4, 5.1, 3.5, ls="--")
    _arrow(ax, 8.3, 4.4, 8.3, 2.3, ls="--", rad=0.25)

    # mask decoder
    _box(ax, 11.0, 5.0, 2.9, 2.9,
         "掩膜解码器（≈4 M）\n"
         "① 提示 token 自注意力\n"
         "② token→image 交叉注意力\n"
         "③ 逐点 FFN\n"
         "④ image→token 交叉注意力\n"
         "⑤ 超分上采样 64→256", C_MD, fs=9)
    _arrow(ax, 9.2, 5.0, 9.5, 5.0)
    _arrow(ax, 6.65, 4.2, 9.5, 5.6, ls="--", color="#1E88E5", rad=-0.2)
    ax.text(7.6, 5.5, "双向交叉注意力", fontsize=8, color="#1E88E5", rotation=-8)

    # 输出
    _box(ax, 13.3, 5.0, 1.2, 1.6, "掩膜\n+IoU 估计", C_IO)
    _arrow(ax, 12.5, 5.0, 12.65, 5.0)

    ax.text(0.1, 0.7,
            "Transformer 落点：① 图像编码器的 12 层 ViT（自注意力）　"
            "② 掩膜解码器的 token↔image 双向交叉注意力（把提示信息注入图像特征）",
            fontsize=9.5, color="#37474F",
            bbox=dict(boxstyle="round,pad=0.4", fc="#F5F5F5", ec="#BDBDBD"))
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[图] {save_path}")


def draw_decoder_flow(save_path: Path):
    """mask decoder 内部数据流（双向交叉注意力）。"""
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(0.1, 9.5, "SAM Mask Decoder 数据流：提示 token 与图像特征的两轮双向交叉注意力",
            fontsize=12.5, fontweight="bold")

    _box(ax, 1.3, 7.6, 2.2, 0.9, "图像嵌入\n256×64×64", "#E1F5FE")
    _box(ax, 1.3, 5.6, 2.2, 0.9, "稠密提示嵌入\n(dense PE)", C_PE)
    _box(ax, 1.3, 3.6, 2.2, 0.9, "提示 token\n(稀疏嵌入 + IoU/掩膜 token)", C_PE)

    _box(ax, 5.0, 6.6, 2.4, 1.0, "① 提示 token\n自注意力", "#E8F5E9")
    _arrow(ax, 2.45, 3.9, 3.75, 6.3, rad=-0.2)
    _box(ax, 8.0, 6.6, 2.6, 1.0, "② token→image\n交叉注意力", "#E8F5E9")
    _arrow(ax, 6.25, 6.6, 6.65, 6.6)
    _arrow(ax, 2.45, 7.4, 6.7, 7.0, ls="--", color="#1E88E5")
    ax.text(4.0, 7.6, "图像特征作为 K/V", fontsize=8, color="#1E88E5")

    _box(ax, 8.0, 4.6, 2.6, 1.0, "③ 逐点 FFN", "#E8F5E9")
    _arrow(ax, 8.0, 6.05, 8.0, 5.15)
    _box(ax, 8.0, 2.6, 2.6, 1.0, "④ image→token\n交叉注意力", "#E8F5E9")
    _arrow(ax, 8.0, 4.05, 8.0, 3.15)
    _arrow(ax, 6.7, 7.2, 6.7, 2.9, ls="--", color="#E65100", rad=0.35)
    ax.text(5.4, 4.7, "token 作为 K/V\n把提示信息写回图像特征", fontsize=8, color="#E65100")

    _box(ax, 11.0, 2.6, 1.6, 2.2, "超分头\n64→256\n掩膜 logits", C_IO)
    _arrow(ax, 9.35, 2.8, 10.15, 2.8)
    _box(ax, 11.0, 6.6, 1.6, 1.0, "IoU 估计头", C_IO)
    _arrow(ax, 8.0, 7.2, 10.15, 6.8, ls="--")

    ax.text(0.1, 0.6, "每跑一次解码还会在 token 里拼一次「上一轮掩膜」，因此 SAM 支持迭代式多轮精修。",
            fontsize=9.5, bbox=dict(boxstyle="round,pad=0.4", fc="#F5F5F5", ec="#BDBDBD"))
    fig.tight_layout(); fig.savefig(save_path, dpi=160); plt.close(fig)
    print(f"[图] {save_path}")


def draw_robustness(save_path: Path):
    """扰动框鲁棒性曲线 + 各提示类型对比（读 outputs/logs/eval_*.json）。"""
    files = sorted(LOG_DIR.glob("eval_*.json"))
    if not files:
        print("[跳过] 还没有评测结果")
        return
    records = [json.loads(f.read_text(encoding="utf-8")) for f in files]

    order = {"pert-20": -20, "pert-10": -10, "gt": 0, "pert10": 10, "pert20": 20}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # 左：扰动曲线（按 stage 分组）
    for stage in ["e1", "e2", "e3"]:
        pts = []
        for r in records:
            if r["stage"] != stage or r["prompt"] not in order:
                continue
            pts.append((order[r["prompt"]], r["metrics"]["dice"], r["metrics"].get("hd95", np.nan)))
        if not pts:
            continue
        pts.sort()
        x = [p[0] for p in pts]
        axes[0].plot(x, [p[1] for p in pts], marker="o", label=f"{stage.upper()} Dice")
    axes[0].axvline(0, color="#999", ls=":", lw=1)
    axes[0].set_xlabel("框扰动幅度（%） 负=内缩 正=外扩")
    axes[0].set_ylabel("Dice")
    axes[0].set_title("提示框扰动鲁棒性曲线")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=9)

    # 右：提示类型对比（GT 框 / 点 / 自动框）
    names, dices, hds = [], [], []
    for r in records:
        if r["prompt"] not in ("gt", "point1", "point3", "point5", "auto", "auto10"):
            continue
        names.append(f"{r['stage'].upper()}\n{r['prompt']}")
        dices.append(r["metrics"]["dice"])
        hds.append(r["metrics"].get("hd95", np.nan))
    if names:
        x = np.arange(len(names))
        axes[1].bar(x, dices, 0.6, color="#2980b9", label="Dice")
        axes[1].set_xticks(x); axes[1].set_xticklabels(names, fontsize=8)
        for i, v in enumerate(dices):
            axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
        axes[1].set_ylim(0, 1.0)
        axes[1].set_ylabel("Dice"); axes[1].set_title("不同提示形式的效果")
        axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"[图] {save_path}")


def draw_cascade(save_path: Path):
    """级联系统图（作业 1 + 作业 2 共用的那一张）。"""
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis("off")
    blocks = [
        (1.4, "① 皮肤镜图像\n256×256", C_IO),
        (4.2, "② U-Net 自动分割\n（作业 1，31 M，~10 ms）\n→ 粗掩膜", "#E8F5E9"),
        (7.0, "③ 掩膜 → 外接矩形框\n（自动提示生成）", C_PE),
        (10.0, "④ SAM 交互式精修\n（作业 2，91 M，~0.5 s）\n→ 精细掩膜", C_VIT),
        (12.8, "⑤ 输出\n给医生复核", C_IO),
    ]
    for i, (x, t, c) in enumerate(blocks):
        _box(ax, x, 2.6, 2.4, 2.0, t, c, fs=9)
        if i > 0:
            _arrow(ax, blocks[i - 1][0] + 1.25, 2.6, x - 1.25, 2.6)
    ax.text(1.4, 0.9, "无需人工", fontsize=9, ha="center", color="#2E7D32")
    ax.text(7.0, 0.9, "把两个作业缝成一条系统", fontsize=9.5, ha="center", color="#1565C0")
    ax.text(10.0, 0.9, "可再叠加点提示迭代", fontsize=9, ha="center", color="#6A1B9A")
    ax.text(0.05, 4.6, "级联系统：小模型出粗框 → 大模型做精修（分工而非替代）",
            fontsize=12.5, fontweight="bold")
    fig.tight_layout(); fig.savefig(save_path, dpi=160); plt.close(fig)
    print(f"[图] {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", action="store_true")
    ap.add_argument("--robust", action="store_true")
    ap.add_argument("--cascade", action="store_true")
    args = ap.parse_args()
    if args.arch:
        draw_sam_arch(FIG_DIR / "arch_sam.png")
        draw_decoder_flow(FIG_DIR / "arch_mask_decoder.png")
    if args.robust:
        draw_robustness(FIG_DIR / "robustness.png")
    if args.cascade:
        draw_cascade(FIG_DIR / "cascade_system.png")


if __name__ == "__main__":
    main()
