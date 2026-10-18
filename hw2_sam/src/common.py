# -*- coding: utf-8 -*-
"""作业 2 公共工具：路径、设备、种子、CSV 日志。

与作业 1 的关系：本作业**复用作业 1 的 processed 数据与指标实现**，
保证两个作业用同一套测试集、同一套指标口径（报告里那张正面对比表才有意义）。
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]          # .../hw2_sam
PROJECT_ROOT = ROOT.parent                          # .../作业
HW1_ROOT = PROJECT_ROOT / "hw1_unet"              # 作业1（级联提示来源）
HW1_SRC = HW1_ROOT / "src"
HW1_PROC = HW1_ROOT / "data" / "processed"

# 直接 import 作业 1 的 metrics，杜绝两版指标不一致。
# ⚠️ 必须 append 而非 insert(0)：insert 会把 hw1 的 src 排到本作业前面，
# DataLoader 的 spawn worker 重新执行主脚本时 `from common import …`
# 会解析到 hw1 的 common.py（没有 EVAL_SIZE）而直接 ImportError。
# hw2 的 src/metrics.py 与 hw1 的保持逐字节同步（cp 同步），因此 append 语义等价。
if str(HW1_SRC) not in sys.path:
    sys.path.append(str(HW1_SRC))

DATA_RAW = ROOT / "data" / "raw"
EMB_DIR = ROOT / "data" / "embeddings"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
LOG_DIR = ROOT / "outputs" / "logs"
FIG_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "reports"
PRED_DIR = ROOT / "outputs" / "predictions"

for _d in (DATA_RAW, EMB_DIR, CKPT_DIR, LOG_DIR, FIG_DIR, REPORT_DIR, PRED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SAM_CKPT = DATA_RAW / "sam_vit_b_01ec64.pth"
SAM_INPUT = 1024        # SAM 图像编码器的固定输入边长
EVAL_SIZE = 256         # 评测分辨率，与作业 1 保持一致


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_ctx(device: torch.device, enabled: bool = True):
    if not enabled or device.type == "cpu":
        import contextlib
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def supports_amp(device: torch.device) -> bool:
    return device.type in ("cuda", "mps")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
