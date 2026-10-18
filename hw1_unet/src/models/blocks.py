# -*- coding: utf-8 -*-
"""三种 U-Net 变体共用的基础积木：卷积块、下采样、上采样、注意力门。

全部手写，不调用 MONAI / segmentation_models_pytorch。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """U-Net 的"两次卷积"单元：(3×3 conv -> BN -> ReLU) × 2。

    为什么用 BN：256×256 输入、batch 只有 8，BN 能显著稳住训练；
    为什么 padding=1：保证卷积前后空间尺寸不变，跳跃连接才能直接 concat。
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """下采样：2×2 最大池化把分辨率砍半，再接 DoubleConv 把通道翻倍。"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """上采样 + 跳跃连接。

    两种上采样方式：
    - bilinear=True  用双线性插值再接 1×1 卷积（无棋盘伪影，参数少）
    - bilinear=False 用转置卷积 ConvTranspose2d（可学习，但容易出棋盘效应）

    跳跃连接是 **concat（沿通道维拼接）不是 add**：U-Net 原论文的设计意图是让
    解码器自己决定用多少浅层细节，add 会强制二者通道数一致并丢失信息。
    """

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)   # 拼接后通道数 = in_ch//2 + in_ch//2

    def forward(self, x, skip):
        x = self.up(x)
        # 由于 resize/池化可能差 1 个像素，这里做一次中心对齐的 pad
        dh = skip.shape[-2] - x.shape[-2]
        dw = skip.shape[-1] - x.shape[-1]
        if dh or dw:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1×1 卷积把通道压到 1，**不加 sigmoid**。

    输出 logits：训练时交给 BCEWithLogitsLoss（数值稳定），
    推理时再 sigmoid 取阈值。这是分割任务的标准做法。
    """

    def __init__(self, in_ch: int, out_ch: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    """ResUNet 的残差块：conv-bn-relu-conv-bn，再与输入**相加**。

    相加要求通道数一致，所以入口先用 1×1 卷积把通道对齐（shortcut）。
    残差连接的意义：梯度可以沿恒等通路直达浅层，缓解深层网络的退化问题。
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Identity()
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.shortcut(x))


class AttentionGate(nn.Module):
    """Attention U-Net 的注意力门（Oktay et al., MIDL 2018）。

    输入：
      x —— 编码器同层的跳跃特征（skip）
      g —— 解码器上一层的输出（gating signal，携带更"高级"的语义）
    做法：
      1. 两个 1×1 卷积把通道统一到 inter_ch，逐元素相加后过 ReLU；
      2. 再一个 1×1 卷积压成 1 通道，过 sigmoid 得到 0~1 的注意力权重 α；
      3. skip 逐通道乘以 α —— 与当前解码语义无关的区域（背景、伪影）被压低。
    直观理解：让网络在拼接之前先"擦掉"跳跃特征里不相关的部分。
    """

    def __init__(self, in_ch_x: int, in_ch_g: int, inter_ch: int):
        super().__init__()
        self.w_x = nn.Conv2d(in_ch_x, inter_ch, kernel_size=1, stride=1, bias=False)
        self.w_g = nn.Conv2d(in_ch_g, inter_ch, kernel_size=1, stride=1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # g 来自更深的层，分辨率更低，先插值到与 x 一致
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=True)
        alpha = self.psi(self.act(self.w_x(x) + self.w_g(g)))
        return x * alpha, alpha
