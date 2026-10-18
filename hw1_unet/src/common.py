# -*- coding: utf-8 -*-
"""公共工具：路径常量、设备探测、随机种子、CSV 日志。

设计原则（硬性要求）：
1. 全部路径由本文件推导，代码里禁止出现 /Users/... 这类绝对路径；
2. 设备按 cuda -> mps -> cpu 三级回退，禁止硬编码 mps；
3. 随机种子统一在这里固定，保证实验可复现。
"""
from __future__ import annotations

import csv
import os
import random
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------- 路径常量
# 本文件位于 <项目根>/hw1_unet/src/common.py
ROOT = Path(__file__).resolve().parents[1]          # .../hw1_unet
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
LOG_DIR = ROOT / "outputs" / "logs"
FIG_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "reports"

for _d in (DATA_RAW, DATA_PROC, CKPT_DIR, LOG_DIR, FIG_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 设备探测
def get_device() -> torch.device:
    """cuda 优先，其次 mps（Apple Silicon），最后 cpu。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_ctx(device: torch.device, enabled: bool = True):
    """混合精度上下文。只有真跑到 mps/cuda 时才开 fp16，cpu 下开了没意义还容易出数值问题。"""
    if not enabled or device.type == "cpu":
        import contextlib
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def supports_amp(device: torch.device) -> bool:
    return device.type in ("cuda", "mps")


# ---------------------------------------------------------------- 随机种子
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # MPS 后端没有单独的种子接口，manual_seed 已覆盖
    torch.use_deterministic_algorithms(False, warn_only=True)


# ---------------------------------------------------------------- CSV 日志
class CSVLogger:
    """每个实验一个 CSV。

    ⚠️ 续跑必须用追加模式：之前固定用 "w" 打开，每次 --resume 都会清空
    之前所有 epoch（实测 9/12 个正式运行的 CSV 被截断，甚至有 0 行），
    而 summary.json 里的 history 是完整的——门户图表读 CSV 就会少点。
    """

    def __init__(self, path, fieldnames, append=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        if append is None:                      # 未显式指定时按文件是否已存在推断
            append = self.path.exists() and self.path.stat().st_size > 0
        self._fp = open(self.path, "a" if append else "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fp, fieldnames=self.fieldnames)
        if not append:
            self._writer.writeheader()
        self._fp.flush()

    def log(self, row):
        self._writer.writerow({k: row.get(k, "") for k in self.fieldnames})
        self._fp.flush()

    def close(self):
        self._fp.close()


# ---------------------------------------------------------------- 实时进度心跳
# 供 dataset_hub 门户的「实时运行」面板读取：每个训练/评测进程周期性写一个
# tiny JSON（几 KB），服务端扫描 outputs/progress/*.json 判断谁在跑、跑到哪。
# 设计原则：失败绝不影响训练（任何异常都吞掉）。
def write_progress(tag: str, **fields) -> None:
    """原子写入 outputs/progress/{tag}.json。"""
    import json as _json
    import os as _os
    import time as _time
    try:
        d = ROOT / "outputs" / "progress"
        d.mkdir(parents=True, exist_ok=True)
        payload = {"tag": tag, "ts": _time.time(), **fields}
        tmp = d / f".{tag}.tmp"
        tmp.write_text(_json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        _os.replace(tmp, d / f"{tag}.json")
    except Exception:
        pass
