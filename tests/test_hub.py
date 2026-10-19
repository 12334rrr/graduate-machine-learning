# -*- coding: utf-8 -*-
"""门户单元/接口测试（用系统 /usr/bin/python3 运行，零第三方依赖）。

覆盖：JSON 清洗、均值/标准差口径、实验状态注入、/api 聚合与 HTTP 冒烟。
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataset_hub"))

import server  # noqa: E402

# 沙箱/CI 环境常注入 HTTP_PROXY，localhost 请求必须绕过代理（否则 502/连接被拒）
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class TestHelpers(unittest.TestCase):
    def test_sanitize_json(self):
        out = server._sanitize_json({"a": float("nan"), "b": [float("inf"), 1.0]})
        self.assertIsNone(out["a"])
        self.assertIsNone(out["b"][0])
        self.assertEqual(out["b"][1], 1.0)

    def test_mean_std(self):
        mean, std = server._mean_std([1.0, 2.0, 3.0])
        self.assertAlmostEqual(mean, 2.0, places=6)
        self.assertAlmostEqual(std, 1.0, places=6)
        self.assertEqual(server._mean_std([5.0]), (5.0, None))
        self.assertEqual(server._mean_std([]), (None, None))

    def test_experiment_status_contains_live_numbers(self):
        text = server.experiment_status_text()
        self.assertIn("当前实验状态", text)
        self.assertIn("作业1", text)
        self.assertTrue("作业2" in text or "E1" in text)


class TestAggregates(unittest.TestCase):
    def test_live_state_shape(self):
        d = server.live_state()
        self.assertIsInstance(d, dict)
        self.assertIn("items", d)
        for it in d["items"]:
            self.assertIn("proj", it)
            self.assertIn(it["proj"], ("hw1", "hw2"))

    def test_hw1_state_keys(self):
        d = server.hw1_state()
        for k in ("runs", "evals", "checkpoints", "figures", "code"):
            self.assertIn(k, d)
        self.assertIsInstance(d["runs"], list)

    def test_hw2_state_keys(self):
        d = server.hw2_state()
        for k in ("stages", "embeddings", "evals", "figures"):
            self.assertIn(k, d)
        self.assertIn("e1", d["stages"])


class TestHTTP(unittest.TestCase):
    BASE = "http://127.0.0.1:8800"

    def _get(self, path):
        with _OPENER.open(self.BASE + path, timeout=5) as r:
            self.assertEqual(r.status, 200)
            return json.loads(r.read().decode("utf-8"))

    def test_api_endpoints(self):
        try:
            ov = self._get("/api/overview")
        except (urllib.error.URLError, ConnectionError) as exc:
            self.skipTest(f"门户未启动：{exc}")
        self.assertIn("datasets", ov)
        self.assertEqual(self._get("/api/live")["count"] >= 0, True)
        self.assertIn("runs", self._get("/api/hw1"))
        self.assertIn("stages", self._get("/api/hw2"))


class TestFrontendAssets(unittest.TestCase):
    def test_index_has_live_and_map(self):
        html = (ROOT / "dataset_hub" / "static" / "index.html").read_text(encoding="utf-8")
        for token in ("nav-live", "实验地图", "function renderLive", "eta_sec", "expMapHw2"):
            self.assertIn(token, html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
