# -*- coding: utf-8 -*-
"""作业 2 的数据读取：直接复用作业 1 生成的 processed npy（保证同一测试集）。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from common import EVAL_SIZE, HW1_PROC


class ISICRaw(Dataset):
    """返回 (id, image_uint8(H,W,3), mask01(H,W))，分辨率 EVAL_SIZE=256。"""

    def __init__(self, split: str = "test", size: int = EVAL_SIZE, data_dir: Path = HW1_PROC):
        self.split = split
        self.size = size
        self.images = np.load(data_dir / f"{split}_images.npy", mmap_mode="r")
        self.masks = np.load(data_dir / f"{split}_masks.npy", mmap_mode="r")
        self.ids = (data_dir / f"{split}_ids.txt").read_text(encoding="utf-8").split()
        assert len(self.images) == len(self.masks) == len(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = np.asarray(self.images[i])
        mask = np.asarray(self.masks[i])
        if self.size != img.shape[0]:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return self.ids[i], img.astype(np.uint8), (mask > 127).astype(np.uint8)


class EmbeddingDataset(Dataset):
    """E2 用：直接读预计算好的 SAM 图像嵌入（fp16，256×64×64）。

    为什么能这么做（这是本作业最重要的工程优化）：
    冻结 image encoder 后，图像嵌入**与提示无关**，一张图只需要算一次。
    E1 评测和 E2 训练共用同一份缓存，E2 每 epoch 从十几分钟降到几十秒。
    """

    def __init__(self, split: str = "train", size: int = EVAL_SIZE,
                 emb_dir: Path = None, data_dir: Path = HW1_PROC, augment: bool = False):
        from common import EMB_DIR
        self.split = split
        self.size = size
        self.augment = augment
        # 缓存按 split 分目录存放：data/embeddings/{train,val,test}/{id}.npy
        # （与 precompute_embeddings.py 的写入路径一致）。默认不能指向父目录。
        self.emb_dir = Path(emb_dir) if emb_dir else (EMB_DIR / split)
        self.images = np.load(data_dir / f"{split}_images.npy", mmap_mode="r")
        self.masks = np.load(data_dir / f"{split}_masks.npy", mmap_mode="r")
        self.ids = (data_dir / f"{split}_ids.txt").read_text(encoding="utf-8").split()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        emb = np.load(self.emb_dir / f"{self.ids[i]}.npy").astype(np.float16)  # (256,64,64)
        mask = np.asarray(self.masks[i])
        if self.size != mask.shape[0]:
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8)
        return torch.from_numpy(np.ascontiguousarray(emb)), torch.from_numpy(mask), self.ids[i]
