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
    """扰动框鲁棒性曲线 + 各提示类型对比（读 outputs/logs/eval_*.json）。

    视觉规范：与门户同一套配色（E1 蓝 / E2 紫 / E3 绿），
    左图用阴影标出两阶段之间的「鲁棒性差距」，右图按阶段着色。
    """
    files = sorted(LOG_DIR.glob("eval_*.json"))
    if not files:
        print("[跳过] 还没有评测结果")
        return
    records = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    # 只取 test 划分（val 冒烟不混入）
    records = [r for r in records if r.get("split") == "test"] or records

    C = {"e1": "#1a73e8", "e2": "#7c3aed", "e3": "#188038"}
    NAME = {"e1": "E1 零样本", "e2": "E2 冻结微调", "e3": "E3 LoRA"}
    order = {"pert-20": -20, "pert-10": -10, "gt": 0, "pert10": 10, "pert20": 20}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ---------- 左：扰动曲线 ----------
    curves = {}
    for stage in ["e1", "e2", "e3"]:
        pts = sorted(
            (order[r["prompt"]], r["metrics"]["dice"])
            for r in records
            if r["stage"] == stage and r["prompt"] in order
        )
        if len(pts) < 2:
            continue
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        curves[stage] = (x, y)
        axes[0].plot(x, y, marker="o", ms=6, lw=2.2, color=C[stage],
                     label=NAME[stage], zorder=3)

    # 用阴影标出 E1 与 E2 之间的差距（微调到底买到了什么）
    if "e1" in curves and "e2" in curves and curves["e1"][0] == curves["e2"][0]:
        xs = curves["e1"][0]
        axes[0].fill_between(xs, curves["e1"][1], curves["e2"][1],
                             color=C["e2"], alpha=0.10, zorder=1,
                             label="微调带来的鲁棒性增益")

    axes[0].axvline(0, color="#9aa0a6", ls=":", lw=1.2)
    axes[0].text(0.6, axes[0].get_ylim()[0] + 0.01, "GT 框", color="#5f6470", fontsize=9, va="bottom")

    # 在最差扰动处标注两阶段差值
    if "e1" in curves and "e2" in curves:
        x1, y1 = curves["e1"][0], curves["e1"][1]
        x2, y2 = curves["e2"][0], curves["e2"][1]
        i = x1.index(min(x1))
        gap = y2[i] - y1[i]
        axes[0].annotate("", xy=(x1[i], y2[i]), xytext=(x1[i], y1[i]),
                         arrowprops=dict(arrowstyle="<->", color="#b06000", lw=1.3))
        axes[0].text(x1[i] + 1.2, (y1[i] + y2[i]) / 2, f"+{gap:.3f}", color="#b06000",
                     fontsize=10, fontweight="bold", va="center")

    axes[0].set_xlabel("框扰动幅度（%）  负 = 内缩，正 = 外扩", fontsize=11)
    axes[0].set_ylabel("Dice", fontsize=11)
    axes[0].set_title("提示框扰动鲁棒性：微调买到的是「不怕提示不准」", fontsize=12.5, pad=10)
    axes[0].set_xticks([-20, -10, 0, 10, 20])
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=9, frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.98))

    # ---------- 右：提示类型对比 ----------
    prompts = ["gt", "pert10", "pert20", "point1", "point3", "point5", "auto", "auto10"]
    PNAME = {"gt": "GT 框", "pert10": "框+10%", "pert20": "框+20%", "point1": "1 点",
             "point3": "3 点", "point5": "5 点", "auto": "自动框", "auto10": "自动框+10%"}
    stages = [st for st in ["e1", "e2", "e3"]
              if any(r["stage"] == st and r["prompt"] in prompts for r in records)]
    if stages:
        x = np.arange(len(prompts))
        w = 0.8 / len(stages)
        for k, st in enumerate(stages):
            vals = []
            for p in prompts:
                hit = next((r for r in records if r["stage"] == st and r["prompt"] == p), None)
                vals.append(hit["metrics"]["dice"] if hit else np.nan)
            bars = axes[1].bar(x + (k - (len(stages) - 1) / 2) * w, vals, w * 0.92,
                               color=C[st], label=NAME[st], zorder=3)
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    axes[1].text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                                 ha="center", fontsize=7.4, color="#3c4043")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([PNAME[p] for p in prompts], fontsize=9, rotation=20, ha="right")
        axes[1].axhline(0.8, color="#9aa0a6", ls="--", lw=1)
        axes[1].text(len(prompts) - 0.4, 0.805, "0.80", color="#5f6470", fontsize=8, ha="right")
        axes[1].set_ylim(0, 1.0)
        axes[1].set_ylabel("Dice", fontsize=11)
        axes[1].set_title("不同提示形式的效果（test 1000 张）", fontsize=12.5, pad=10)
        axes[1].grid(axis="y", alpha=0.25)
        axes[1].legend(fontsize=9, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[图] {save_path}")


def draw_cascade(save_path: Path):
    """级联系统图（作业 1 + 作业 2 共用的那一张）。

    布局按内容动态计算，避免固定坐标导致边缘框被裁、底部大片留白。
    """
    blocks = [
        ("① 皮肤镜图像\n256×256", C_IO),
        ("② U-Net 自动分割\n（作业 1，29 M，~10 ms）\n→ 粗掩膜", "#E8F5E9"),
        ("③ 掩膜 → 外接矩形框\n（自动提示生成）", C_PE),
        ("④ SAM 交互式精修\n（作业 2，94 M，0.02–0.9 s）\n→ 精细掩膜", C_VIT),
        ("⑤ 输出\n给医生复核", C_IO),
    ]
    n = len(blocks)
    w, h, gap, margin = 2.3, 1.9, 0.62, 0.55
    total_w = margin * 2 + n * w + (n - 1) * gap

    fig, ax = plt.subplots(figsize=(13.6, 3.3))
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, 3.75)
    ax.axis("off")

    yc = 1.95
    xs = [margin + w / 2 + i * (w + gap) for i in range(n)]
    for i, ((text, color), x) in enumerate(zip(blocks, xs)):
        _box(ax, x, yc, w, h, text, color, fs=8.8)
        if i > 0:
            _arrow(ax, xs[i - 1] + w / 2 + 0.06, yc, x - w / 2 - 0.06, yc)

    ax.text(xs[0], 0.62, "无需人工", fontsize=9.5, ha="center", color="#2E7D32")
    ax.text(xs[2], 0.62, "把两个作业缝成一条系统", fontsize=9.5, ha="center", color="#1565C0")
    ax.text(xs[3], 0.62, "可再叠加点提示迭代", fontsize=9.5, ha="center", color="#6A1B9A")
    ax.text(margin, 3.42, "级联系统：小模型出粗框 → 大模型做精修（分工而非替代）",
            fontsize=12.5, fontweight="bold", va="center")

    fig.tight_layout()
    fig.savefig(save_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
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
