# -*- coding: utf-8 -*-
"""汇总作业 2 的评测结果 -> 报告用表格（Markdown + JSON）。

读入 outputs/logs/eval_{stage}_{prompt}[_{unet_tag}].json，
产出：
- outputs/results_summary.json
- reports/tables.md

同时把作业 1 的结果（若已生成）并进来，直接给出"U-Net vs SAM"正面对比表。
"""
from __future__ import annotations

import json
from pathlib import Path

from common import LOG_DIR, REPORT_DIR, HW1_ROOT

STAGE_CN = {"e1": "E1 零样本", "e2": "E2 冻结编码器微调", "e3": "E3 LoRA 微调"}
PROMPT_CN = {
    "gt": "GT 外矩形框", "pert10": "外扩 10%", "pert20": "外扩 20%",
    "pert-10": "内缩 10%", "pert-20": "内缩 20%",
    "point1": "中心 1 点", "point3": "3 点", "point5": "5 点",
    "auto": "自动框(U-Net)", "auto10": "自动框外扩 10%",
}


def load_hw2():
    rows = []
    for p in sorted(LOG_DIR.glob("eval_*.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    return rows


def load_hw1():
    p = HW1_ROOT / "outputs" / "results_summary.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def md_hw2(rows):
    L = ["# 作业 2 实验结果汇总（自动生成）\n",
         "## 1. 三阶段总体比较（测试集，GT 框提示）\n",
         "| 阶段 | Dice | IoU | HD95 (px) | Sensitivity | Specificity | 单图推理 (ms) |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        if r["prompt"] != "gt" or r["split"] != "test":
            continue
        m = r["metrics"]
        L.append(f"| {STAGE_CN.get(r['stage'], r['stage'])} | {m['dice']:.4f} | {m['iou']:.4f} | "
                 f"{m['hd95']:.2f} | {m['sensitivity']:.4f} | {m['specificity']:.4f} | "
                 f"{m.get('latency_ms_per_img', float('nan')):.0f} |")

    L += ["\n## 2. 提示框扰动鲁棒性\n",
          "| 阶段 | 内缩 20% | 内缩 10% | GT 框 | 外扩 10% | 外扩 20% |",
          "| --- | --- | --- | --- | --- | --- |"]
    order = ["pert-20", "pert-10", "gt", "pert10", "pert20"]
    for stage in ("e1", "e2", "e3"):
        cells = []
        for pr in order:
            hit = [r for r in rows if r["stage"] == stage and r["prompt"] == pr
                   and r["split"] == "test"]
            cells.append(f"{hit[0]['metrics']['dice']:.4f}" if hit else "—")
        if all(c == "—" for c in cells):
            continue
        L.append(f"| {STAGE_CN[stage]} | " + " | ".join(cells) + " |")

    L += ["\n## 3. 提示形式对比\n",
          "| 阶段 | 提示 | Dice | IoU | HD95 |",
          "| --- | --- | --- | --- | --- |"]
    for r in rows:
        if r["split"] != "test" or r["prompt"] == "gt":
            continue
        m = r["metrics"]
        L.append(f"| {STAGE_CN.get(r['stage'], r['stage'])} | "
                 f"{PROMPT_CN.get(r['prompt'], r['prompt'])} | {m['dice']:.4f} | "
                 f"{m['iou']:.4f} | {m['hd95']:.2f} |")
    return "\n".join(L)


def md_cross(hw1, hw2_rows):
    """作业 1 与作业 2 的正面对比表。"""
    if hw1 is None:
        return "\n## 4. 与作业 1 的正面对比\n\n（作业 1 的汇总结果尚未生成）"
    models = hw1.get("models", {})
    best_unet = models.get("unet", {})
    best_sam = None
    for r in hw2_rows:
        if r["split"] == "test" and r["prompt"] == "gt":
            if best_sam is None or r["metrics"]["dice"] > best_sam["metrics"]["dice"]:
                best_sam = r
    L = ["\n## 4. 与作业 1 的正面对比\n",
         "| 维度 | U-Net（作业 1） | SAM（作业 2） |",
         "| --- | --- | --- |"]
    u_dice = best_unet.get("dice", float("nan"))
    u_hd = best_unet.get("hd95", float("nan"))
    u_par = best_unet.get("params_M", "")
    s_dice = best_sam["metrics"]["dice"] if best_sam else float("nan")
    s_hd = best_sam["metrics"]["hd95"] if best_sam else float("nan")
    s_lat = best_sam["metrics"].get("latency_ms_per_img", float("nan")) if best_sam else float("nan")
    L.append(f"| Dice（越高越好） | {u_dice:.4f} | {s_dice:.4f} |")
    L.append(f"| HD95（越低越好） | {u_hd:.2f} | {s_hd:.2f} |")
    L.append(f"| 可训练参数 | {u_par} M | 91 M（LoRA 可训约 0.5 M） |")
    L.append("| 单图推理时延 | ~10 ms 级 | 见上表（含 1024 ViT 编码） |")
    L.append("| 是否需人工提示 | 否 | 是 |")
    L.append("| 换数据集迁移 | 需重训 | 零样本可用 |")
    return "\n".join(L)


def main():
    rows = load_hw2()
    hw1 = load_hw1()
    (LOG_DIR.parent / "results_summary.json").write_text(
        json.dumps({"rows": rows, "hw1": hw1}, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "tables.md").write_text(md_hw2(rows) + md_cross(hw1, rows), encoding="utf-8")
    print(md_hw2(rows))
    print(f"\n[保存] {REPORT_DIR / 'tables.md'}")


if __name__ == "__main__":
    main()
