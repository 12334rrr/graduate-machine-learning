# -*- coding: utf-8 -*-
"""ResUNet（Zhang et al., IEEE GRSL 2018）—— 手写实现。

与 U-Net 的两处差别：
1. 每个编码/解码块换成 **残差块**（conv-bn-relu-conv-bn + 跳跃相加），
   下采样用 stride=2 的卷积而不是最大池化；
2. 编码器与解码器之间仍用 **concat** 传细节（"残差相加"发生在块内部）。

残差块带来的好处：在 ISIC 这种"前景占比小、边界模糊"的任务上，深层梯度
更稳，收敛更快；代价是参数量比同宽度 U-Net 略多。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import OutConv, ResidualBlock


class UpResBlock(nn.Module):
    """解码侧的一个上采样块：插值放大 -> 与 skip concat -> 1×1 卷积降通道 -> 残差块。"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, bias=False),
        )
        self.conv = ResidualBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        dh = skip.shape[-2] - x.shape[-2]
        dw = skip.shape[-1] - x.shape[-1]
        if dh or dw:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ResUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base: int = 64):
        super().__init__()
        self.name = "ResUNet"
        self.inc = ResidualBlock(in_channels, base)          # 256
        self.down1 = ResidualBlock(base, base * 2, stride=2)      # 128
        self.down2 = ResidualBlock(base * 2, base * 4, stride=2)  # 64
        self.down3 = ResidualBlock(base * 4, base * 8, stride=2)  # 32
        self.down4 = ResidualBlock(base * 8, base * 16, stride=2)  # 16
        self.up1 = UpResBlock(base * 16, base * 8)           # 32
        self.up2 = UpResBlock(base * 8, base * 4)            # 64
        self.up3 = UpResBlock(base * 4, base * 2)            # 128
        self.up4 = UpResBlock(base * 2, base)                # 256
        self.outc = OutConv(base, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def build_resunet(**kw) -> nn.Module:
    return ResUNet(**kw)
