# -*- coding: utf-8 -*-
"""Attention U-Net（Oktay et al., MIDL 2018）—— 手写实现。

骨架与 U-Net 完全一致，唯一区别：每条跳跃连接上挂一个注意力门（AttentionGate）。
gating 信号来自解码器上一层输出（更"高级"的语义），用来给 skip 特征逐通道"打分"。

**消融开关 attn_where**（报告"不同超参数设置比较"要用）：
    all         —— 4 条跳跃连接都加门（论文默认）
    skip        —— 只在 enc1/enc2 两条浅层（分辨率高、噪声多）加门
    bottleneck  —— 只在 enc4（最靠近 bottleneck 的那条）加门
    none        —— 不加门（退化成普通 U-Net，用于对照）
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import AttentionGate, DoubleConv, Down, OutConv, Up


class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 base: int = 64, bilinear: bool = True,
                 attn_where: str = "all"):
        super().__init__()
        self.name = "AttentionUNet"
        self.attn_where = attn_where
        self.inc = DoubleConv(in_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.down4 = Down(base * 8, base * 16)
        self.up1 = Up(base * 16, base * 8, bilinear)
        self.up2 = Up(base * 8, base * 4, bilinear)
        self.up3 = Up(base * 4, base * 2, bilinear)
        self.up4 = Up(base * 2, base, bilinear)
        self.outc = OutConv(base, out_channels)

        # 每条门的输入通道 = (skip 通道, gating 通道)。
        # gating 来自解码器更粗一级的输出，通道数比 skip 多一倍，注意力门内部的
        # 两个 1×1 卷积会把它们统一到 inter_ch 后再相加。
        specs = {
            "g4": (base * 8, base * 16),   # skip=x4(512), gating=x5(1024)
            "g3": (base * 4, base * 8),    # skip=x3(256), gating=dec32(512)
            "g2": (base * 2, base * 4),    # skip=x2(128), gating=dec64(256)
            "g1": (base, base * 2),        # skip=x1(64),  gating=dec128(128)
        }
        enable = {"all": {"g1", "g2", "g3", "g4"},
                  "skip": {"g1", "g2"},
                  "bottleneck": {"g4"},
                  "none": set()}[attn_where]
        self.enable = enable
        for k, (cx, cg) in specs.items():
            if k in enable:
                setattr(self, k, AttentionGate(cx, cg, max(cx // 2, 1)))
            else:
                setattr(self, k, None)

        # 保存最近一次前向的注意力权重，供报告画图
        self.last_alphas = {}

    def _gate(self, key, skip, gating):
        """统一的门控调用：门不存在就直接返回原 skip。"""
        g = getattr(self, key)
        if g is None:
            return skip
        out, alpha = g(skip, gating)
        self.last_alphas[key] = alpha.detach()
        return out

    def forward(self, x):
        x1 = self.inc(x)        # 256, 64
        x2 = self.down1(x1)     # 128, 128
        x3 = self.down2(x2)     # 64, 256
        x4 = self.down3(x3)     # 32, 512
        x5 = self.down4(x4)     # 16, 1024

        x4 = self._gate("g4", x4, x5)     # 用 bottleneck 语义给最深层 skip 打分
        x = self.up1(x5, x4)              # 32, 512
        x3 = self._gate("g3", x3, x)
        x = self.up2(x, x3)               # 64, 256
        x2 = self._gate("g2", x2, x)
        x = self.up3(x, x2)               # 128, 128
        x1 = self._gate("g1", x1, x)
        x = self.up4(x, x1)               # 256, 64
        return self.outc(x)


def build_attn_unet(**kw) -> nn.Module:
    return AttentionUNet(**kw)
