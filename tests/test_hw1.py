# -*- coding: utf-8 -*-
"""作业 1 单元测试（在 hw1-unet-medseg 环境运行）。

覆盖：指标计算、五折划分、三模型前向形状、汇总口径。
不训练、不读大图，全部 < 1 分钟。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hw1_unet" / "src"))

from dataset import ISICSeg  # noqa: E402
from metrics import count_components, dice_score, foreground_ratio, hd95, iou_score  # noqa: E402
from models import build_model  # noqa: E402
from collect_results import is_main_run  # noqa: E402


class TestMetrics(unittest.TestCase):
    def test_dice_iou_known(self):
        pred = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        gt = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        self.assertAlmostEqual(dice_score(pred, gt), 0.5, places=6)
        self.assertAlmostEqual(iou_score(pred, gt), 1 / 3, places=6)

    def test_identical_and_disjoint(self):
        x = np.zeros((8, 8), dtype=np.uint8)
        x[2:5, 2:5] = 1
        self.assertAlmostEqual(dice_score(x, x), 1.0, places=6)
        self.assertAlmostEqual(iou_score(x, x), 1.0, places=6)
        self.assertAlmostEqual(hd95(x, x), 0.0, places=6)
        y = np.zeros_like(x)
        y[6, 6] = 1
        self.assertAlmostEqual(dice_score(x, y), 0.0, places=6)

    def test_foreground_and_components(self):
        x = np.zeros((10, 10), dtype=np.uint8)
        x[1:3, 1:3] = 1
        x[6:8, 6:8] = 1
        self.assertAlmostEqual(foreground_ratio(x), 8 / 100, places=6)
        self.assertEqual(count_components(x), 2)


class TestFoldSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train0 = ISICSeg("train", size=64, fold=0)
        cls.val0 = ISICSeg("val", size=64, fold=0)
        cls.test = ISICSeg("test", size=64)

    def test_sizes(self):
        self.assertEqual(len(self.train0) + len(self.val0), 2694)  # 官方 train+val
        self.assertEqual(len(self.val0), 539)

    def test_disjoint_and_no_test_leak(self):
        self.assertEqual(len(set(self.train0.ids) & set(self.val0.ids)), 0)
        self.assertEqual(len((set(self.train0.ids) | set(self.val0.ids)) & set(self.test.ids)), 0)

    def test_item_shape(self):
        img, mask, sid = self.train0[0]
        self.assertEqual(tuple(img.shape), (3, 64, 64))
        self.assertEqual(tuple(mask.shape), (1, 64, 64))
        self.assertIsInstance(sid, str)


class TestModels(unittest.TestCase):
    def test_forward_shapes(self):
        x = torch.randn(1, 3, 64, 64)
        for name in ("unet", "resunet", "attn_unet"):
            model = build_model(name).eval()
            with torch.no_grad():
                y = model(x)
            self.assertEqual(tuple(y.shape), (1, 1, 64, 64), name)


class TestAggregation(unittest.TestCase):
    def test_main_tag_filter(self):
        for tag in ("unet_s42", "resunet_s2026", "attn_unet_s7"):
            self.assertTrue(is_main_run(tag), tag)
        for tag in ("loss_bce", "size192", "attn_skip", "diag_1e-3_amp", "smoke", "cv_attn_unet_f0"):
            self.assertFalse(is_main_run(tag), tag)


if __name__ == "__main__":
    unittest.main(verbosity=2)
