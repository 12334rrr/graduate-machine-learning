# -*- coding: utf-8 -*-
"""分割评价指标，全部手写实现（不依赖 MONAI / medpy），便于在报告里写清定义。

指标口径（逐样本计算后对测试集取平均，除非另有说明）：
- Dice    = 2|P∩G| / (|P|+|G|)
- IoU     = |P∩G| / |P∪G|
- HD95    = 双向 95% 分位 Hausdorff 距离（欧氏距离变换求最近边界距离）
- Sens    = TP/(TP+FN)  召回，漏诊率的反面
- Spec    = TN/(TN+FP)  误诊率的反面
- Acc     = (TP+TN)/总像素  —— 本报告里作为"反面教材"指标使用

空掩膜处理（硬性要求）：预测为空或金标准为空时，Dice/IoU 按约定取值并在
`empty_stats` 里单独计数，绝不参与 HD95 平均，避免除零污染整体结果。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


# ------------------------------------------------------------------ 单样本
def _as_binary(a: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """概率图/0-255 掩膜统一转成 bool。"""
    arr = np.asarray(a)
    if arr.dtype != np.bool_:
        arr = arr > (thr if arr.max() <= 1.0 else thr * 255)
    return arr


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = _as_binary(pred), _as_binary(gt)
    inter = np.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0                      # 双方都为空：约定为完美（ISIC 无此情况）
    return float(2.0 * inter / denom)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = _as_binary(pred), _as_binary(gt)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def _surface(mask: np.ndarray) -> np.ndarray:
    """取二值掩膜的边界像素：原集合减去一次腐蚀结果。"""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return np.logical_and(mask, ~binary_erosion(mask, border_value=0))


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95% 分位 Hausdorff 距离（像素单位）。

    做法：对两个掩膜分别取边界集合 A、B；用欧氏距离变换得到"任意像素到最近
    边界像素"的距离图；把 A 中所有点的距离取 95 分位、B 同理，取两者较大值。
    这样既不用暴力两两求距离，又能天然剔除个别离群点带来的爆炸值。
    """
    p, g = _as_binary(pred), _as_binary(gt)
    if not p.any() or not g.any():
        return float("nan")             # 空掩膜：返回 NaN，由调用方单独统计
    sp, sg = _surface(p), _surface(g)
    dt_g = distance_transform_edt(~sg)  # 任意像素 -> 最近的 GT 边界像素
    dt_p = distance_transform_edt(~sp)
    d_p2g = dt_g[sp]
    d_g2p = dt_p[sg]
    if d_p2g.size == 0 or d_g2p.size == 0:
        return float("nan")
    return float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95)))


def sens_spec_acc(pred: np.ndarray, gt: np.ndarray):
    """逐像素 Sensitivity / Specificity / Accuracy。"""
    p, g = _as_binary(pred), _as_binary(gt)
    tp = np.logical_and(p, g).sum()
    tn = np.logical_and(~p, ~g).sum()
    fp = np.logical_and(p, ~g).sum()
    fn = np.logical_and(~p, g).sum()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    acc = (tp + tn) / (tp + tn + fp + fn)
    return float(sens), float(spec), float(acc)


def foreground_ratio(gt: np.ndarray) -> float:
    """前景（病变）像素占比，用于数据集统计。"""
    return float(_as_binary(gt).mean())


def count_components(mask: np.ndarray) -> int:
    """连通域个数（8 邻域）。用 scipy.ndimage.label，避免引入 skimage 依赖。"""
    from scipy.ndimage import label
    m = _as_binary(mask)
    if not m.any():
        return 0
    lab, n = label(m, structure=np.ones((3, 3), dtype=int))   # structure 全 1 => 8 邻域
    return int(n)


# ------------------------------------------------------------------ 汇总
@dataclass
class MetricAccumulator:
    """累积整个测试集的结果，自动隔离空掩膜样本。"""

    dices: List[float] = field(default_factory=list)
    ious: List[float] = field(default_factory=list)
    hds: List[float] = field(default_factory=list)
    sens: List[float] = field(default_factory=list)
    spec: List[float] = field(default_factory=list)
    acc: List[float] = field(default_factory=list)
    n_empty_pred: int = 0       # 预测全为背景（漏检）
    n_empty_gt: int = 0         # 金标准全为背景（数据本身问题）
    n_total: int = 0
    n_spec_undefined: int = 0   # 预测全前景 => 背景像素为 0 => 特异度 0/0 未定义
    n_sens_undefined: int = 0   # 金标准全背景 => 病灶像素为 0 => 灵敏度 0/0 未定义

    def add(self, pred: np.ndarray, gt: np.ndarray, compute_hd: bool = True) -> None:
        self.n_total += 1
        p, g = _as_binary(pred), _as_binary(gt)
        if not p.any():
            self.n_empty_pred += 1
        if not g.any():
            self.n_empty_gt += 1
        self.dices.append(dice_score(p, g))
        self.ious.append(iou_score(p, g))
        s, sp_, ac = sens_spec_acc(p, g)
        # 与 HD95 同一套哲学：未定义的样本单独计数、不并入平均，
        # 否则一个全前景预测就会把整份评测的 specificity 变成 NaN。
        if np.isnan(s):
            self.n_sens_undefined += 1
        else:
            self.sens.append(s)
        if np.isnan(sp_):
            self.n_spec_undefined += 1
        else:
            self.spec.append(sp_)
        self.acc.append(ac)
        if compute_hd and p.any() and g.any():
            h = hd95(p, g)
            if not np.isnan(h):
                self.hds.append(h)

    def summary(self) -> Dict[str, float]:
        def m(x):
            # nanmean 兜底：万一有漏网的 NaN 也不会污染整体均值
            return float(np.nanmean(x)) if len(x) else float("nan")

        def s(x):
            return float(np.nanstd(x)) if len(x) else float("nan")

        return {
            "n": self.n_total,
            "dice": m(self.dices), "dice_std": s(self.dices),
            "iou": m(self.ious), "iou_std": s(self.ious),
            "hd95": m(self.hds), "hd95_std": s(self.hds),
            "hd95_valid": len(self.hds),
            "sensitivity": m(self.sens), "sens_valid": len(self.sens),
            "specificity": m(self.spec), "spec_valid": len(self.spec),
            "pixel_acc": m(self.acc),
            "empty_pred": self.n_empty_pred,
            "empty_gt": self.n_empty_gt,
            "spec_undefined": self.n_spec_undefined,
            "sens_undefined": self.n_sens_undefined,
        }
