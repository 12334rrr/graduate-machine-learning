# -*- coding: utf-8 -*-
"""手写 LoRA（Low-Rank Adaptation）实现，注入 SAM 的 ViT 注意力 q/v 投影。

原理（Hu et al., ICLR 2022）：
    冻结原始权重 W0 ∈ R^{d×k}，旁路加一个低秩分解 ΔW = B·A，
    A ∈ R^{r×k}（高斯初始化），B ∈ R^{d×r}（**零初始化**，保证训练起点等价原模型）。
    前向：h = W0·x + (α/r)·B·A·x
训练时只更新 A、B，参数量从 d×k 降到 r×(d+k)。

为什么只动 q、v 不动 k、out_proj：
    SAMed 的实验结论是 q/v 就够了，再多不涨点反而更容易过拟合；
    在 2594 张的小数据集上，这一点尤其重要。

SAM 的 qkv 是**一个** Linear 同时产出 q/k/v（out_features = 3×dim），
所以用 `slices` 掩码只把 LoRA 增量加到 q 段和 v 段，k 段保持原样。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """把一个 nn.Linear 包成带 LoRA 旁路的线性层。"""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0,
                 slices=None, dropout: float = 0.0):
        super().__init__()
        assert r > 0
        # ⚠️ LoRA 参数/mask 必须建在 base 权重所在设备上：
        # nn.Parameter 默认建在 CPU，而 build_sam 已把模型放到 MPS，
        # 不跟随的话前向第一步就报 "weight is on cpu but expected on mps"。
        dev = base.weight.device
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)          # 原始权重冻结

        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if slices is None:
            self.register_buffer("mask", torch.ones(base.out_features, device=dev))
        else:
            m = torch.zeros(base.out_features, device=dev)
            for s in slices:
                m[slice(*s)] = 1.0
            self.register_buffer("mask", m)

    def forward(self, x):
        out = self.base(x)
        delta = F.linear(F.linear(self.drop(x), self.lora_A), self.lora_B)  # (..., out)
        return out + delta * self.scaling * self.mask

    @property
    def lora_params(self):
        return [self.lora_A, self.lora_B]


def apply_lora_to_sam(sam, r: int = 8, alpha: float = 16.0, target_modules=("qkv",)):
    """把 LoRA 注入 SAM 的 ViT 图像编码器（qkv 的 q/v 段）。

    返回注入了 LoRA 的 sam；可训练参数只有 lora_A / lora_B。
    """
    n_injected = 0
    blocks = sam.image_encoder.blocks
    for blk in blocks:
        attn = blk.attn
        if "qkv" in target_modules:
            d = attn.qkv.out_features // 3          # dim
            slices = [(0, d), (2 * d, 3 * d)]       # q 段 与 v 段
            attn.qkv = LoRALinear(attn.qkv, r=r, alpha=alpha, slices=slices)
            n_injected += 1
    print(f"[LoRA] 已在 {n_injected} 个 Transformer block 的 qkv(q,v) 注入 r={r}")
    return sam


def collect_lora_params(model):
    """收集所有 LoRA 参数，交给优化器。"""
    params = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            params += m.lora_params
    return params


def count_lora_params(model):
    return sum(p.numel() for p in collect_lora_params(model))


def lora_state_dict(model):
    """只保存 LoRA 相关参数（几 MB，方便备份）。"""
    sd = {}
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            sd[name + ".lora_A"] = m.lora_A
            sd[name + ".lora_B"] = m.lora_B
    return sd
