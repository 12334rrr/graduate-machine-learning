# -*- coding: utf-8 -*-
"""汇总所有实验结果 -> 报告要用的表格（Markdown + JSON）。

读两类文件：
- outputs/logs/{tag}_summary.json   训练过程（含 args：model / size / loss / seed / attn_where）
- outputs/logs/eval_{tag}_{split}.json  测试集评测指标

产出：
- outputs/results_summary.json
- reports/tables.md  （报告直接粘的 Markdown 表）
"""
from __future__ import annotations

import json
from pathlib import Path

from common import LOG_DIR, REPORT_DIR

MODEL_CN = {"unet": "U-Net", "resunet": "ResUNet", "attn_unet": "Attention U-Net"}


def guess_model(tag: str) -> str:
    """从 tag 前缀推断模型名（旧 summary 没有 args 键时的兜底）。"""
    for key in ("attn_unet", "resunet", "unet"):
        if tag.startswith(key):
            return key
    return "?"


def load_all():
    rows = []
    for p in sorted(LOG_DIR.glob("eval_*.json")):
        ev = json.loads(p.read_text(encoding="utf-8"))
        tag = ev["tag"]
        sp = LOG_DIR / f"{tag}_summary.json"
        summ, args = {}, {}
        if sp.exists():
            summ = json.loads(sp.read_text(encoding="utf-8"))
            args = summ.get("args", {}) or {}
        # 兼容：早期 summary 只写了 model 顶层键（无 args），此处回退，避免汇总表全空
        model = args.get("model") or summ.get("model") or guess_model(tag)
        m = ev["metrics"]
        rows.append({
            "tag": tag, "model": model, "seed": args.get("seed", summ.get("seed", "?")),
            "size": args.get("size", summ.get("size", ev.get("size", "?"))),
            "loss": args.get("loss", summ.get("loss", "bce_dice")),
            "attn_where": args.get("attn_where", summ.get("attn_where", "all")),
            "split": ev["split"], "params_M": ev.get("params_M", ""),
            "dice": m["dice"], "dice_std": m.get("dice_std", float("nan")),
            "iou": m["iou"], "hd95": m["hd95"],
            "sensitivity": m["sensitivity"], "specificity": m["specificity"],
            "pixel_acc": m["pixel_acc"],
            "empty_pred": m.get("empty_pred", 0), "n": m.get("n", 0),
        })
    return rows


def agg_models(rows, split="test"):
    """按模型聚合多种子：均值 ± 标准差。"""
    out = {}
    for r in rows:
        if r["split"] != split:
            continue
        out.setdefault(r["model"], []).append(r)
    res = {}
    for k, v in out.items():
        d = [x["dice"] for x in v]
        i = [x["iou"] for x in v]
        h = [x["hd95"] for x in v]
        res[k] = {
            "model_cn": MODEL_CN.get(k, k), "n_runs": len(v),
            "seeds": [x["seed"] for x in v],
            "dice": float(sum(d) / len(d)), "dice_std": float(max(d) - min(d)) if len(d) > 1 else 0.0,
            "dice_sd": float((sum((x - sum(d) / len(d)) ** 2 for x in d) / max(1, len(d) - 1)) ** 0.5),
            "iou": float(sum(i) / len(i)), "hd95": float(sum(h) / len(h)),
            "sensitivity": float(sum(x["sensitivity"] for x in v) / len(v)),
            "specificity": float(sum(x["specificity"] for x in v) / len(v)),
            "pixel_acc": float(sum(x["pixel_acc"] for x in v) / len(v)),
            "params_M": v[0].get("params_M", ""),
            "per_seed": [{"seed": x["seed"], "dice": x["dice"], "iou": x["iou"], "hd95": x["hd95"]}
                         for x in v],
        }
    return res


def build_ablation(rows, split="test"):
    """从评测行拼出消融图所需的结构（plot_figures.py --ablation 的输入）。

    形如 {"损失函数": {"BCE": {"dice":…, "hd95":…}, …}, …}，
    每个维度的默认设置取自对照实验（unet_s42 / attn_unet_s42）。
    """
    by_tag = {r["tag"]: r for r in rows if r["split"] == split}

    def cell(tag):
        r = by_tag.get(tag)
        if not r:
            return None
        return {"dice": round(r["dice"], 4), "hd95": round(r["hd95"], 2), "iou": round(r["iou"], 4)}

    groups = {
        "损失函数": {"BCE": cell("loss_bce"), "Dice": cell("loss_dice"),
                 "0.5BCE+0.5Dice（默认）": cell("unet_s42")},
        "输入分辨率": {"192": cell("size192"), "256（默认）": cell("unet_s42"), "384": cell("size384")},
        "注意力位置": {"仅 skip": cell("attn_skip"), "仅 bottleneck": cell("attn_bottleneck"),
                  "全部（默认）": cell("attn_unet_s42")},
    }
    # 去掉尚未跑出的空项；整组为空则整组丢弃
    out = {}
    for g, items in groups.items():
        kept = {k: v for k, v in items.items() if v}
        if kept:
            out[g] = kept
    return out


def build_markdown(rows, models):
    L = []
    L.append("# 实验结果汇总（自动生成）\n")
    L.append("## 1. 三模型总体比较（测试集，多种子）\n")
    L.append("| 模型 | 种子数 | Dice (均值±标准差) | IoU | HD95 (px) | Sensitivity | Specificity | 像素准确率 | 参数量(M) |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for k in ["unet", "resunet", "attn_unet"]:
        if k not in models:
            continue
        v = models[k]
        L.append(f"| {v['model_cn']} | {v['n_runs']} | {v['dice']:.4f} ± {v['dice_sd']:.4f} | "
                 f"{v['iou']:.4f} | {v['hd95']:.2f} | {v['sensitivity']:.4f} | "
                 f"{v['specificity']:.4f} | {v['pixel_acc']:.4f} | {v['params_M']} |")
    L.append("\n## 2. 逐种子明细\n")
    L.append("| 模型 | 种子 | Dice | IoU | HD95 |")
    L.append("| --- | --- | --- | --- | --- |")
    for k in ["unet", "resunet", "attn_unet"]:
        if k not in models:
            continue
        for ps in models[k]["per_seed"]:
            L.append(f"| {models[k]['model_cn']} | {ps['seed']} | {ps['dice']:.4f} | "
                     f"{ps['iou']:.4f} | {ps['hd95']:.2f} |")
    L.append("\n## 3. 消融实验（单种子）\n")
    L.append("| 消融维度 | 设置 | Dice | IoU | HD95 |")
    L.append("| --- | --- | --- | --- | --- |")
    ABLATION_TAGS = {"loss_bce": "损失函数", "loss_dice": "损失函数",
                     "size192": "输入分辨率", "size384": "输入分辨率",
                     "attn_skip": "注意力位置", "attn_bottleneck": "注意力位置"}
    for r in rows:
        if r["split"] != "test":
            continue
        dim = ABLATION_TAGS.get(r["tag"])
        # 注意：不能只用前缀匹配（attn_ 会把正式实验 attn_unet_s* 误收进来）
        if dim:
            L.append(f"| {dim} | {r['tag']} | {r['dice']:.4f} | {r['iou']:.4f} | {r['hd95']:.2f} |")
    return "\n".join(L)


def main():
    rows = load_all()
    models = agg_models(rows, "test")
    ablation = build_ablation(rows, "test")
    (LOG_DIR.parent / "results_summary.json").write_text(
        json.dumps({"rows": rows, "models": models, "ablation": ablation},
                   indent=2, ensure_ascii=False),
        encoding="utf-8")
    md = build_markdown(rows, models)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "tables.md").write_text(md, encoding="utf-8")
    # 供 plot_figures.py --ablation 直接使用（此前无人生成，导致消融图缺数据源）
    (LOG_DIR.parent / "ablation.json").write_text(
        json.dumps(ablation, indent=2, ensure_ascii=False), encoding="utf-8")
    print(md)
    print(f"\n[保存] {LOG_DIR.parent / 'results_summary.json'}")
    print(f"[保存] {LOG_DIR.parent / 'ablation.json'}（消融图数据源）")
    print(f"[保存] {REPORT_DIR / 'tables.md'}")


if __name__ == "__main__":
    main()
