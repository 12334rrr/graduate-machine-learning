# -*- coding: utf-8 -*-
"""损失函数：BCE / Dice / 0.5×BCE + 0.5×Dice（消融实验要对比这三个）。

两个实现细节值得在报告里提一句：
1. 网络输出的是 **logits**（不接 sigmoid/softmax），BCE 用 BCEWithLogitsLoss，
   数值稳定性比自己 sigmoid 再算 BCE 好得多（log-sum-exp 技巧）。
2. Dice 在概率空间上算。分子分母都加平滑项 eps，防止全空预测时梯度为 NaN。
3. 混合精度下 logits 可能是 fp16，进损失前统一 cast 回 fp32，避免 fp16 下
   log() 溢出（这是 MPS 上很常见的"损失突然变 nan"的根因）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss = 1 - Dice(prob, target)，在 (N,1,H,W) 上整体计算。"""
    prob = torch.sigmoid(logits.float())
    target = target.float()
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    return (1.0 - dice).mean()


def bce_loss_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits.float(), target.float())


class SegLoss(nn.Module):
    """按权重混合 BCE 与 Dice。

    loss = w_bce * BCE + w_dice * Dice
    默认 0.5/0.5；消融时分别取 (1,0) 和 (0,1) 得到纯 BCE / 纯 Dice。
    """

    def __init__(self, w_bce: float = 0.5, w_dice: float = 0.5):
        super().__init__()
        self.w_bce = float(w_bce)
        self.w_dice = float(w_dice)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.w_bce > 0:
            loss = loss + self.w_bce * bce_loss_from_logits(logits, target)
        if self.w_dice > 0:
            loss = loss + self.w_dice * dice_loss_from_logits(logits, target)
        return loss

    def __repr__(self) -> str:
        return f"SegLoss(bce={self.w_bce}, dice={self.w_dice})"


def build_loss(name: str) -> nn.Module:
    """按名字拿损失函数，消融实验直接传 --loss bce / dice / bce_dice。"""
    name = name.lower()
    if name == "bce":
        return SegLoss(1.0, 0.0)
    if name == "dice":
        return SegLoss(0.0, 1.0)
    if name in ("bce_dice", "bcedice", "combo"):
        return SegLoss(0.5, 0.5)
    raise ValueError(f"未知损失: {name}（可选 bce / dice / bce_dice）")
