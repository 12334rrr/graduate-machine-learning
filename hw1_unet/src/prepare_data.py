# -*- coding: utf-8 -*-
"""数据准备：解压官方压缩包 -> 按 id 配对图像与掩膜 -> resize 存成 npy -> 出统计报告。

产出（落在 data/processed/）：
    {split}_images.npy  uint8, (N, S, S, 3)
    {split}_masks.npy   uint8, (N, S, S)  取值 0/255
    {split}_ids.txt     每行一个 ISIC id，保证图像与掩膜严格对齐

为什么统一存成 384×384：
消融实验要对比 192/256/384 三种输入分辨率。存一份 384 的母版，训练时再
按需下采样，比"每个分辨率各存一份"省 3 倍磁盘，且 384→256 的 resize 在
DataLoader 里只占几毫秒。
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import label

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_PROC, DATA_RAW, REPORT_DIR, FIG_DIR   # noqa: E402

STORE_SIZE = 384          # 母版分辨率
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 官方压缩包 -> (解压目录名, 是图像还是掩膜, split)
PACKAGES = [
    ("ISIC2018_Task1-2_Training_Input.zip", "ISIC2018_Task1-2_Training_Input", "img", "train"),
    ("ISIC2018_Task1_Training_GroundTruth.zip", "ISIC2018_Task1_Training_GroundTruth", "mask", "train"),
    ("ISIC2018_Task1-2_Validation_Input.zip", "ISIC2018_Task1-2_Validation_Input", "img", "val"),
    ("ISIC2018_Task1_Validation_GroundTruth.zip", "ISIC2018_Task1_Validation_GroundTruth", "mask", "val"),
    ("ISIC2018_Task1-2_Test_Input.zip", "ISIC2018_Task1-2_Test_Input", "img", "test"),
    ("ISIC2018_Task1_Test_GroundTruth.zip", "ISIC2018_Task1_Test_GroundTruth", "mask", "test"),
]


def unzip_all(splits=("train", "val", "test")) -> None:
    """解压官方包到 data/raw/<同名目录>/。已存在则跳过。"""
    for zip_name, out_dir, _, sp in PACKAGES:
        if sp not in splits:
            continue
        zp = DATA_RAW / zip_name
        od = DATA_RAW / out_dir
        if not zp.exists():
            raise FileNotFoundError(f"缺压缩包：{zp}，先跑 src/download_isic.py")
        if od.exists() and any(od.iterdir()):
            print(f"[跳过解压] {out_dir} 已存在")
            continue
        print(f"[解压] {zip_name} -> {out_dir}")
        od.mkdir(parents=True, exist_ok=True)
        # macOS 自带的 unzip 对部分压缩参数处理不好，优先用 Python 的 zipfile
        try:
            with zipfile.ZipFile(zp) as z:
                z.extractall(od)
        except Exception as e:                                   # 兜底用系统 ditto
            print(f"  zipfile 失败({e})，改用 ditto")
            subprocess.run(["ditto", "-x", "-k", str(zp), str(od)], check=True)


def _find_files(d: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [p for p in d.rglob("*") if p.suffix.lower() in exts and p.is_file()]


def pair_split(img_dir: Path, mask_dir: Path):
    """按 ISIC id 配对。图像 ISIC_0000000.jpg，掩膜 ISIC_0000000_segmentation.png。"""
    def key(p: Path) -> str:
        stem = p.stem
        return stem.replace("_segmentation", "").replace("_mask", "").replace("_lesion", "")

    imgs = {key(p): p for p in _find_files(img_dir)}
    masks = {key(p): p for p in _find_files(mask_dir)}
    common = sorted(set(imgs) & set(masks))
    missing_img = sorted(set(masks) - set(imgs))
    missing_mask = sorted(set(imgs) - set(masks))
    if missing_img or missing_mask:
        print(f"  [警告] 缺图像 {len(missing_img)} 例 / 缺掩膜 {len(missing_mask)} 例（已丢弃）")
    return common, [imgs[k] for k in common], [masks[k] for k in common]


def _load_mask(p: Path) -> np.ndarray:
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(f"读不了掩膜: {p}")
    return (m > 127).astype(np.uint8)      # 统一二值化，杜绝 0/1 与 0/255 混用


def _process_one(args):
    img_path, mask_path, size = args
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"读不了图像: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = _load_mask(mask_path)
    h0, w0 = img.shape[:2]
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

    m = mask.astype(bool)
    fg_ratio = float(m.mean())
    n_empty = int(not m.any())
    if m.any():
        _, n_comp = label(m)
    else:
        n_comp = 0
    return img.astype(np.uint8), (mask * 255).astype(np.uint8), (h0, w0), fg_ratio, n_empty, n_comp


def build_split(split: str, img_dir: Path, mask_dir: Path, size: int, workers: int = 8):
    ids, img_paths, mask_paths = pair_split(img_dir, mask_dir)
    print(f"[{split}] 配对成功 {len(ids)} 例")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_process_one,
                                [(ip, mp, size) for ip, mp in zip(img_paths, mask_paths)]))

    images = np.stack([r[0] for r in results]).astype(np.uint8)
    masks = np.stack([r[1] for r in results]).astype(np.uint8)
    orig_sizes = np.array([r[2] for r in results], dtype=np.int32)
    fg_ratios = np.array([r[3] for r in results], dtype=np.float64)
    n_empty = int(sum(r[4] for r in results))
    n_comp = np.array([r[5] for r in results], dtype=np.int32)

    np.save(DATA_PROC / f"{split}_images.npy", images)
    np.save(DATA_PROC / f"{split}_masks.npy", masks)
    np.save(DATA_PROC / f"{split}_orig_sizes.npy", orig_sizes)
    (DATA_PROC / f"{split}_ids.txt").write_text("\n".join(ids), encoding="utf-8")
    np.save(DATA_PROC / f"{split}_fg_ratios.npy", fg_ratios)
    np.save(DATA_PROC / f"{split}_ncomp.npy", n_comp)
    print(f"[{split}] 已存 images{masks.shape} -> {DATA_PROC}")
    return dict(split=split, n=len(ids), fg_ratios=fg_ratios, n_empty=n_empty,
                n_comp=n_comp, orig_sizes=orig_sizes)


def write_stats(stats_list, out_path: Path) -> None:
    """把统计结果写成 reports/data_stats.md（首日必做项）。"""
    lines = ["# ISIC 2018 Task 1 数据集统计\n",
             "> 由 `src/prepare_data.py` 自动生成，口径：掩膜 resize 到 384×384 后统计。\n"]
    lines.append("| split | 样本数 | 前景占比均值 | 前景占比中位数 | 前景占比 5%分位 | 前景占比 95%分位 | 空掩膜 | 多连通域(≥2) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in stats_list:
        fr = s["fg_ratios"]
        lines.append(
            f"| {s['split']} | {s['n']} | {fr.mean():.4f} | {np.median(fr):.4f} | "
            f"{np.percentile(fr, 5):.4f} | {np.percentile(fr, 95):.4f} | "
            f"{s['n_empty']} | {int((s['n_comp'] >= 2).sum())} |"
        )
    lines.append("")
    allfr = np.concatenate([s["fg_ratios"] for s in stats_list])
    lines.append("## 关键结论\n")
    lines.append(f"- 全库 {len(allfr)} 例，前景像素占比均值 **{allfr.mean()*100:.2f}%**，"
                 f"中位数 {np.median(allfr)*100:.2f}% —— 背景占绝对多数，"
                 "这正是**像素准确率是个废指标**的根源（全预测背景也能拿到 ~85%）。")
    lines.append(f"- 前景占比 5% 分位仅 {np.percentile(allfr, 5)*100:.2f}%：存在极小病灶样本，"
                 "这类样本是 Dice 的主要拉低项。")
    lines.append(f"- 空掩膜 **{sum(s['n_empty'] for s in stats_list)}** 例；"
                 f"多连通域样本 **{sum(int((s['n_comp'] >= 2).sum()) for s in stats_list)}** 例"
                 "（一处病灶被断开，或标注里含多个皮损）。")
    lines.append("")
    lines.append("## 原始图像尺寸\n")
    lines.append("| split | 高 中位数 | 宽 中位数 | 高 范围 | 宽 范围 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for s in stats_list:
        hs = s["orig_sizes"][:, 0]
        ws = s["orig_sizes"][:, 1]
        lines.append(f"| {s['split']} | {int(np.median(hs))} | {int(np.median(ws))} | "
                     f"{hs.min()}–{hs.max()} | {ws.min()}–{ws.max()} |")
    lines.append("\n> 结论：原图长宽比不一致且分辨率差异大，统一 resize 到正方形是必需步骤；"
                 "代价是会引入形变，这也是后面 HD95 偏大的一个来源。\n")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[统计] 已写入 {out_path}")
    # 同步写一份 JSON，供 dashboard 后端直接读取
    import json as _json
    js = {}
    for s in stats_list:
        fr = s["fg_ratios"]
        js[s["split"]] = {
            "count": int(s["n"]),
            "fg_ratio_mean": float(fr.mean()),
            "fg_ratio_median": float(np.median(fr)),
            "fg_p05": float(np.percentile(fr, 5)),
            "fg_p95": float(np.percentile(fr, 95)),
            "empty_masks": int(s["n_empty"]),
            "multi_component": int((s["n_comp"] >= 2).sum()),
        }
    (out_path.parent / "data_stats.json").write_text(
        _json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_stats(stats_list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Heiti TC", "Songti SC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for s in stats_list:
        axes[0].hist(s["fg_ratios"] * 100, bins=40, alpha=0.55, label=f"{s['split']} (n={s['n']})")
    axes[0].set_xlabel("前景像素占比 (%)")
    axes[0].set_ylabel("样本数")
    axes[0].set_title("病变（前景）像素占比分布")
    axes[0].legend()

    nc = np.concatenate([s["n_comp"] for s in stats_list])
    axes[1].bar(["0\n(空掩膜)", "1", "2", "≥3"],
                [int((nc == 0).sum()), int((nc == 1).sum()),
                 int((nc == 2).sum()), int((nc >= 3).sum())],
                color=["#c0392b", "#2980b9", "#27ae60", "#f39c12"])
    axes[1].set_ylabel("样本数")
    axes[1].set_title("掩膜连通域个数分布")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "data_stats.png", dpi=150)
    plt.close(fig)
    print(f"[图] 已存 {FIG_DIR / 'data_stats.png'}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=STORE_SIZE)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"],
                    help="只处理指定 split（下载还没全时可先跑 train/val）")
    args = ap.parse_args()

    unzip_all(args.splits)
    stats_list = []
    for split in args.splits:
        img_dir = DATA_RAW / [out for name, out, kind, sp in PACKAGES if kind == "img" and sp == split][0]
        mask_dir = DATA_RAW / [out for name, out, kind, sp in PACKAGES if kind == "mask" and sp == split][0]
        stats_list.append(build_split(split, img_dir, mask_dir, args.size, args.workers))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_stats(stats_list, REPORT_DIR / "data_stats.md")
    plot_stats(stats_list)


if __name__ == "__main__":
    main()
