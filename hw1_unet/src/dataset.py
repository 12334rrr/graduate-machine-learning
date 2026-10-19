# -*- coding: utf-8 -*-
"""ISIC 分割数据集与数据增强。

设计要点：
1. 母版存 384×384，按 --size 下采样到 192/256/384（消融"输入分辨率"用）；
2. **增强只作用在训练集**，验证/测试集只做 resize + 归一化（否则指标不可比）；
3. 图像与掩膜必须用**同一套几何变换**（同一个仿射矩阵），否则掩膜就错位了；
   色彩抖动只作用在图像上，掩膜不受影响；
4. 归一化用 ImageNet 均值方差，因为三个模型都是随机初始化从零训练，
   用 ImageNet 统计量能让输入分布落在合理区间，收敛更快。
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from common import DATA_PROC

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 五折交叉验证的固定划分种子：train+val 合并后做一次确定性置换，
# 所有 fold / 所有模型看到的分折完全一致（与训练随机种子无关）。
CV_SPLIT_SEED = 12345
CV_FOLDS = 5


class RandomAugment:
    """手写几何 + 色彩增强。

    几何部分用一个 2×3 仿射矩阵统一描述「缩放 + 旋转 + 平移 + 翻转」，
    图像用 INTER_LINEAR、掩膜用 INTER_NEAREST 同一矩阵 warp，保证严格对齐。
    """

    def __init__(self, p_flip=0.5, rot_deg=30, scale_range=(0.8, 1.2), color=True):
        self.p_flip = p_flip
        self.rot_deg = rot_deg
        self.scale_range = scale_range
        self.color = color

    def __call__(self, img: np.ndarray, mask: np.ndarray):
        h, w = img.shape[:2]
        cy, cx = h / 2.0, w / 2.0

        # 1) 随机缩放
        s = random.uniform(*self.scale_range)
        # 2) 随机旋转
        ang = random.uniform(-self.rot_deg, self.rot_deg)
        # 3) 随机平移（±10% 边长）
        tx = random.uniform(-0.1, 0.1) * w
        ty = random.uniform(-0.1, 0.1) * h

        M = cv2.getRotationMatrix2D((cx, cy), ang, s)
        M[:, 2] += [tx, ty]

        # 4) 翻转直接融进仿射矩阵：把缩放系数取反等价于镜像
        if random.random() < self.p_flip:
            M[:, 0] = -M[:, 0]
            M[0, 2] += w
        if random.random() < self.p_flip:
            M[:, 1] = -M[:, 1]
            M[1, 2] += h

        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # 5) 色彩抖动（只改图像，掩膜是语义标签不能动）
        if self.color:
            if random.random() < 0.5:
                b = random.uniform(0.8, 1.2)
                c = random.uniform(0.8, 1.2)
                img = np.clip(img.astype(np.float32) * b, 0, 255).astype(np.uint8)
                img = np.clip((img.astype(np.float32) - 127.5) * c + 127.5, 0, 255).astype(np.uint8)
            if random.random() < 0.3:
                # 轻微高斯噪声：模拟皮肤镜图像的采集噪声
                sigma = random.uniform(1.0, 5.0)
                img = np.clip(img.astype(np.float32) +
                              np.random.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)
        return img, mask


class ISICSeg(Dataset):
    """从 npy 母版读取，按 size 下采样。

    fold=None 时沿用官方划分（train 2594 / val 100 / test 1000）；
    fold=k 时把 train+val 合并为 2694 张，做确定性五折划分：
    split="train" 返回其余 4 折，split="val" 返回第 k 折。test 不受 fold 影响。
    """

    def __init__(self, split: str = "train", size: int = 256, augment: bool = False,
                 data_dir: Path = DATA_PROC, fold: int | None = None,
                 folds: int = CV_FOLDS):
        self.split = split
        self.size = size
        self.augment = augment
        self.fold = fold
        self._parts = None
        self._index = None

        if fold is None:
            self.images = np.load(data_dir / f"{split}_images.npy", mmap_mode="r")
            self.masks = np.load(data_dir / f"{split}_masks.npy", mmap_mode="r")
            self.ids = (data_dir / f"{split}_ids.txt").read_text(encoding="utf-8").split()
            assert len(self.images) == len(self.masks) == len(self.ids)
        else:
            assert 0 <= fold < folds
            assert split in ("train", "val"), "五折模式只用于 train/val"
            parts = []
            for sp in ("train", "val"):
                parts.append((
                    np.load(data_dir / f"{sp}_images.npy", mmap_mode="r"),
                    np.load(data_dir / f"{sp}_masks.npy", mmap_mode="r"),
                    (data_dir / f"{sp}_ids.txt").read_text(encoding="utf-8").split(),
                ))
            all_ids = parts[0][2] + parts[1][2]
            n = len(all_ids)
            perm = np.random.RandomState(CV_SPLIT_SEED).permutation(n)
            chunks = np.array_split(perm, folds)
            keep = (np.concatenate([chunks[i] for i in range(folds) if i != fold])
                    if split == "train" else chunks[fold])
            offsets = (0, len(parts[0][2]))
            self._parts = parts
            self._index = [(0, int(g)) if g < offsets[1] else (1, int(g) - offsets[1])
                           for g in keep]
            self.ids = [all_ids[int(g)] for g in keep]

        self.aug = RandomAugment() if augment else None

    def __len__(self) -> int:
        return len(self.ids)

    def raw_pair(self, i: int):
        """第 i 个样本的 384 母版原图与掩膜（画案例图用，不受 fold/size 影响）。"""
        if self._parts is not None:
            part, local = self._index[i]
            return (np.asarray(self._parts[part][0][local]),
                    np.asarray(self._parts[part][1][local]))
        return np.asarray(self.images[i]), np.asarray(self.masks[i])

    def __getitem__(self, i: int):
        if self._parts is not None:
            part, local = self._index[i]
            img = np.asarray(self._parts[part][0][local])     # (384,384,3) uint8
            mask = np.asarray(self._parts[part][1][local])    # (384,384) uint8 0/255
        else:
            img = np.asarray(self.images[i])                  # (384,384,3) uint8
            mask = np.asarray(self.masks[i])                  # (384,384) uint8 0/255

        if self.size != img.shape[0]:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        if self.aug is not None:
            img, mask = self.aug(img, mask)

        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))   # (3,H,W)

        mask = (mask > 127).astype(np.float32)          # 阈值化回 0/1，杜绝 0/255 混用
        mask = torch.from_numpy(mask[None, ...])        # (1,H,W)
        return img, mask, self.ids[i]


def seed_worker(worker_id):
    """DataLoader worker 的种子初始化。

    必须定义在模块顶层：用 spawn 启动的多进程要 pickle 这个函数，
    定义在 build_loaders 内部会直接报 "Can't pickle local object"。
    """
    worker_seed = (torch.initial_seed() + worker_id) % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loaders(size=256, batch_size=8, num_workers=4, seed=42, data_dir=DATA_PROC,
                  fold=None, folds=CV_FOLDS):
    """构造 train/val/test 三个 DataLoader。test 只在最终评测时用。

    fold=k 时 train/val 改为「train+val 合并后的五折划分」，test 不变。
    """
    import torch
    from torch.utils.data import DataLoader

    g = torch.Generator()
    g.manual_seed(seed)

    train_ds = ISICSeg("train", size=size, augment=True, data_dir=data_dir,
                       fold=fold, folds=folds)
    val_ds = ISICSeg("val", size=size, augment=False, data_dir=data_dir,
                     fold=fold, folds=folds)
    test_ds = ISICSeg("test", size=size, augment=False, data_dir=data_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, worker_init_fn=seed_worker,
                              generator=g, drop_last=True, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=False)
    return train_loader, val_loader, test_loader
