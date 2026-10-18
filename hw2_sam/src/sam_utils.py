# -*- coding: utf-8 -*-
"""SAM 相关封装：建模型、图像预处理、提示（框/点）编码、掩膜解码。

SAM 的三个部件（这也是报告里要讲清的 Transformer 落点）：
1. **image encoder** —— ViT-B/16：把 1024×1024 图像切成 64×64 个 16×16 patch，
   经 12 层 **Transformer** block（多头自注意力 + FFN + LayerNorm）得到
   256×64×64 的图像嵌入。参数量约 89 M，占 SAM 总参数的 97%。
2. **prompt encoder** —— 把提示（点/框/掩膜）变成嵌入：稀疏提示用可学习的
   token + 位置编码，稠密提示（掩膜）用卷积下采样。
3. **mask decoder** —— 轻量 Transformer：先对图像嵌入做自注意力，再用
   **双向交叉注意力**（token→image 与 image→token 两个方向交替）把提示信息
   注入，最后上采样出掩膜。参数约 4 M。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import EVAL_SIZE, SAM_CKPT, SAM_INPUT, get_device

MODEL_TYPE = "vit_b"


def build_sam(device=None, ckpt: Path = SAM_CKPT):
    from segment_anything import sam_model_registry
    if not ckpt.exists():
        raise FileNotFoundError(f"缺 SAM 权重：{ckpt}，先跑 src/download_sam.py")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(ckpt))
    sam.to(device or get_device())
    return sam


def preprocess_image(img_u8: np.ndarray, sam, device) -> torch.Tensor:
    """(H,W,3) uint8 -> (1,3,1024,1024) float 归一化张量。"""
    x = cv2.resize(img_u8, (SAM_INPUT, SAM_INPUT), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(x.transpose(2, 0, 1)).float()[None].to(device)
    return sam.preprocess(t)          # 减 pixel_mean / 除 pixel_std


def to_1024_box(box_xyxy: np.ndarray, src_size: int = EVAL_SIZE) -> np.ndarray:
    """把 EVAL_SIZE 分辨率下的框坐标线性放大到 SAM 的 1024 空间。

    注意：SAM 的 prompt encoder 内部会做坐标归一化，但要求坐标是**预处理后
    图像空间**（即 1024）下的值，所以这里必须自己乘 scale。
    """
    scale = SAM_INPUT / float(src_size)
    return np.asarray(box_xyxy, dtype=np.float32) * scale


def perturb_box(box_xyxy, scale: float, size: int = EVAL_SIZE, rng: np.random.Generator = None):
    """对 GT 框做外扩(scale>0)/内缩(scale<0)扰动，用于鲁棒性实验。

    scale=+0.1 表示四条边各向外扩 10% 的框宽/高；scale=-0.2 表示内缩 20%。
    内缩后框退化（宽或高 < 2 px）时，退回中心点框，避免出现非法框。
    """
    rng = rng or np.random.default_rng(42)
    x0, y0, x1, y1 = [float(v) for v in box_xyxy]
    w, h = x1 - x0, y1 - y0
    dx, dy = scale * w, scale * h
    nx0, ny0, nx1, ny1 = x0 - dx, y0 - dy, x1 + dx, y1 + dy
    # 内缩时加一点点随机抖动，模拟"标注不准"而不是"等比缩放"
    jitter = rng.uniform(-0.02, 0.02, 4) * np.array([w, h, w, h])
    nx0, ny0, nx1, ny1 = (np.array([nx0, ny0, nx1, ny1]) + jitter)
    nx0, ny0 = max(0.0, nx0), max(0.0, ny0)
    nx1, ny1 = min(size - 1.0, nx1), min(size - 1.0, ny1)
    if nx1 - nx0 < 2 or ny1 - ny0 < 2:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        nx0, ny0, nx1, ny1 = cx - 1, cy - 1, cx + 1, cy + 1
    return np.array([nx0, ny0, nx1, ny1], dtype=np.float32)


def points_from_mask(mask: np.ndarray, k: int = 1, rng: np.random.Generator = None):
    """从掩膜里采样 k 个点提示：k=1 取重心，k>1 取"最远点采样"的近似。

    重心可能落在掩膜外（凹形病灶），因此先检查、退回随机前景点。
    """
    rng = rng or np.random.default_rng(42)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    if k == 1:
        cy, cx = int(ys.mean()), int(xs.mean())
        if mask[cy, cx] == 0:
            i = rng.integers(len(xs))
            cx, cy = int(xs[i]), int(ys[i])
        return np.array([[cx, cy]], dtype=np.float32)
    # 多中心点：把前景按 k-means(k) 粗聚类，每类取重心（用简单的均匀网格近似）
    H, W = mask.shape
    step = max(1, int(np.sqrt(len(xs) / k)))
    pts = []
    for gy in range(0, H, max(1, H // step)):
        for gx in range(0, W, max(1, W // step)):
            sub = mask[gy:gy + H // step, gx:gx + W // step]
            sy, sx = np.where(sub > 0)
            if len(sx) > 0:
                pts.append([gx + int(sx.mean()), gy + int(sy.mean())])
    pts = np.array(pts, dtype=np.float32)
    if len(pts) == 0:
        return np.array([[int(xs.mean()), int(ys.mean())]], dtype=np.float32)
    if len(pts) > k:
        idx = rng.choice(len(pts), k, replace=False)
        pts = pts[idx]
    return pts


def gt_box_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def encode_and_decode(sam, image_embedding, box_1024=None, points_1024=None,
                      point_labels=None, multimask=False):
    """给定图像嵌入 + 提示 -> 掩膜 logits。

    image_embedding: (B,256,64,64)
    返回 masks_logits: (B,1,256,256)
    """
    device = image_embedding.device
    B = image_embedding.shape[0]
    pe = sam.prompt_encoder.get_dense_pe().to(device).expand(B, -1, -1, -1)

    boxes_t = None
    if box_1024 is not None:
        boxes_t = torch.as_tensor(np.asarray(box_1024), dtype=torch.float32, device=device)
        if boxes_t.ndim == 1:
            boxes_t = boxes_t[None]
        boxes_t = boxes_t[:B]

    pts_t, lbl_t = None, None
    if points_1024 is not None:
        pts_list, lbl_list = [], []
        for i in range(B):
            p = points_1024[i] if isinstance(points_1024, (list, tuple)) else points_1024
            p = np.asarray(p, dtype=np.float32)
            if p.ndim == 1:
                p = p[None]
            lab = point_labels[i] if point_labels is not None else np.ones(len(p), dtype=np.int32)
            lab = np.asarray(lab, dtype=np.int32)
            pts_list.append(p)
            lbl_list.append(lab)
        pts_t = torch.as_tensor(np.stack(pts_list), dtype=torch.float32, device=device)
        lbl_t = torch.as_tensor(np.stack(lbl_list), dtype=torch.int32, device=device)

    # 注意：points 必须是 None 或 (coords, labels) 二元组；
    # 传 (None, None) 会让 prompt_encoder 里的 _get_batch_size 直接崩。
    pts_arg = (pts_t, lbl_t) if pts_t is not None else None
    sparse, dense = sam.prompt_encoder(points=pts_arg, boxes=boxes_t, masks=None)

    # SAM 的 mask_decoder 是「单图 + K 组掩膜 token」设计：
    # predict_masks 里会 repeat_interleave(image_embeddings, tokens.shape[0])。
    # 批量 B>1 时 image(B=8) 会被复制成 (64,…) 与 dense (8,…) 冲突——必须逐图解码。
    # （E1 评测逐图 B=1 从未暴露；E2 训练 batch=8 一上来就崩。）
    image_embedding = image_embedding.to(device)
    pe = sam.prompt_encoder.get_dense_pe().to(device)
    if B == 1:
        low_res, iou_pred = sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=multimask,
        )
        return low_res, iou_pred

    lows, ious = [], []
    for i in range(B):
        # ⚠️ image_pe 恒为 (1,256,64,64)（与图像内容无关的固定正弦网格），
        # 不能用 pe[i:i+1] 切片——i≥1 时会切出空张量，two-way attention 输出 size 0。
        low, iou = sam.mask_decoder(
            image_embeddings=image_embedding[i : i + 1],
            image_pe=pe[:1],
            sparse_prompt_embeddings=sparse[i : i + 1],
            dense_prompt_embeddings=dense[i : i + 1],
            multimask_output=multimask,
        )
        lows.append(low)
        ious.append(iou)
    return torch.cat(lows, dim=0), torch.cat(ious, dim=0)


def resize_logits_to_eval(low_res: torch.Tensor, size: int = EVAL_SIZE) -> torch.Tensor:
    return F.interpolate(low_res.float(), (size, size), mode="bilinear", align_corners=False)
