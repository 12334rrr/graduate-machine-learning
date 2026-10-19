# -*- coding: utf-8 -*-
"""作业 2 单元测试（在 hw2-sam-transformer 环境运行）。

覆盖：提示构造（GT/扰动/点/自动框）、坐标变换、LoRA 零初始化等价性。
不加载 SAM 权重（375MB），全部 < 30 秒。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hw2_sam" / "src"))

from evaluate_sam import build_prompt  # noqa: E402
from lora import LoRALinear  # noqa: E402
from metrics import dice_score  # noqa: E402
from sam_utils import to_1024_box  # noqa: E402


def _mask(size=64, x0=16, y0=16, x1=48, y1=48):
    m = np.zeros((size, size), dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


class TestPrompts(unittest.TestCase):
    def setUp(self):
        self.mask = _mask()
        self.rng = np.random.default_rng(42)

    def test_gt_box(self):
        box, pts = build_prompt("gt", self.mask, 64, self.rng)
        self.assertIsNone(pts)
        # 外接框取像素索引：x/y 从 16 到 47（48 是开区间）
        np.testing.assert_allclose(box, [16, 16, 47, 47])

    def test_perturb_shrinks_and_expands(self):
        b_gt, _ = build_prompt("gt", self.mask, 64, self.rng)
        b_in, _ = build_prompt("pert-20", self.mask, 64, np.random.default_rng(1))
        b_out, _ = build_prompt("pert20", self.mask, 64, np.random.default_rng(1))
        area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
        self.assertLess(area(b_in), area(b_gt))
        self.assertGreater(area(b_out), area(b_gt))

    def test_points(self):
        box, pts = build_prompt("point3", self.mask, 64, self.rng)
        self.assertIsNone(box)
        self.assertEqual(pts.shape, (3, 2))

    def test_auto_and_auto10(self):
        boxes = {"ISIC_x": [10, 10, 40, 40]}
        b_auto, _ = build_prompt("auto", self.mask, 64, self.rng, boxes, "ISIC_x")
        b_a10, _ = build_prompt("auto10", self.mask, 64, self.rng, boxes, "ISIC_x")
        np.testing.assert_allclose(b_auto, [10, 10, 40, 40])
        self.assertLessEqual(b_a10[0], b_auto[0])
        self.assertGreaterEqual(b_a10[2], b_auto[2])

    def test_missing_auto_returns_none(self):
        box, pts = build_prompt("auto", self.mask, 64, self.rng, {}, "missing")
        self.assertIsNone(box)
        self.assertIsNone(pts)


class TestCoords(unittest.TestCase):
    def test_to_1024(self):
        np.testing.assert_allclose(to_1024_box(np.array([0, 0, 255, 255], dtype=np.float32)),
                                   [0, 0, 1020, 1020])  # 256 → 1024 是 ×4


class TestLoRA(unittest.TestCase):
    def test_zero_init_equivalence_and_delta(self):
        base = nn.Linear(8, 8)
        layer = LoRALinear(base, r=2, alpha=4.0).eval()
        x = torch.randn(3, 8)
        with torch.no_grad():
            self.assertTrue(torch.allclose(layer(x), base(x), atol=1e-6))
            layer.lora_B.data.normal_()
            self.assertFalse(torch.allclose(layer(x), base(x), atol=1e-6))
        # 原始权重被冻结，只有 LoRA 参数可训练
        self.assertFalse(any(p.requires_grad for p in layer.base.parameters()))
        self.assertTrue(all(p.requires_grad for p in layer.lora_params))


class TestMetrics(unittest.TestCase):
    def test_dice(self):
        pred = np.zeros((4, 4), dtype=np.uint8)
        gt = np.zeros((4, 4), dtype=np.uint8)
        pred[0, 0] = gt[0, 0] = 1
        self.assertAlmostEqual(dice_score(pred, gt), 1.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
