# -*- coding: utf-8 -*-
"""模型工厂：按名字构造手写网络。"""
from __future__ import annotations

import torch.nn as nn

from .attn_unet import AttentionUNet
from .resunet import ResUNet
from .unet import UNet

__all__ = ["UNet", "ResUNet", "AttentionUNet", "build_model"]


def build_model(name: str, **kwargs) -> nn.Module:
    name = name.lower().replace("-", "").replace("_", "")
    if name in ("unet",):
        return UNet(**kwargs)
    if name in ("resunet",):
        return ResUNet(**kwargs)
    if name in ("attnunet", "attentionunet", "attunet"):
        return AttentionUNet(**kwargs)
    raise ValueError(f"未知模型: {name}（可选 unet / resunet / attn_unet）")


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
