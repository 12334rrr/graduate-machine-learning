# -*- coding: utf-8 -*-
"""经典 U-Net（Ronneberger et al., MICCAI 2015）—— 手写实现。

结构（输入 3×256×256）：
    编码器 64 -> 128 -> 256 -> 512 -> 1024（bottleneck）
    解码器 512 -> 256 -> 128 -> 64，每层与编码器同分辨率特征 concat
    1×1 卷积输出 1 通道 logits

张量形状变化表（默认 base=64、4 次下采样、输入 256×256）：
     stage     空间尺寸      通道
     enc1      256×256       64
     enc2      128×128       128
     enc3      64×64         256
     enc4      32×32         512
     bottleneck 16×16        1024
     dec4      32×32         512
     dec3      64×64         256
     dec2      128×128       128
     dec1      256×256       64
     out       256×256       1
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import DoubleConv, Down, OutConv, Up


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 base: int = 64, bilinear: bool = True):
        super().__init__()
        self.name = "UNet"
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

    def forward(self, x):
        x1 = self.inc(x)      # 256
        x2 = self.down1(x1)   # 128
        x3 = self.down2(x2)   # 64
        x4 = self.down3(x3)   # 32
        x5 = self.down4(x4)   # 16
        x = self.up1(x5, x4)  # 32
        x = self.up2(x, x3)   # 64
        x = self.up3(x, x2)   # 128
        x = self.up4(x, x1)   # 256
        return self.outc(x)


def build_unet(**kw) -> nn.Module:
    return UNet(**kw)
