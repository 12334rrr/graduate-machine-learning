#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
作业1（hw1_unet）监控仪表盘后端
================================
基于 Python 标准库 http.server 的零依赖聚合服务：
  - 扫描 outputs/logs 下的训练 CSV / 评测 JSON，汇总为实时状态
  - 暴露 src 代码浏览、checkpoints 模型下载、figures/reports 查看
  - 前端 dashboard/index.html 每 5 秒轮询 /api/state 实现实时刷新

启动：  python dashboard/server.py [--port 8801]
"""
import argparse
import csv
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent.parent          # hw1_unet/
DASH = Path(__file__).resolve().parent                  # hw1_unet/dashboard/
LOGS = ROOT / "outputs" / "logs"
CKPT = ROOT / "outputs" / "checkpoints"
FIGS = ROOT / "outputs" / "figures"
REPORTS = ROOT / "reports"
SRC = ROOT / "src"

PROJECT = {
    "id": "hw1",
    "title": "作业 1 · 医学图像分割（手写 U-Net 系列）",
    "subtitle": "ISIC 2018 Task 1 皮肤病变分割 · U-Net / ResUNet / Attention U-Net 三模型对比",
    "env": "hw1-unet-medseg",
    "accent": "#4f8cff",
}


def read_csv_series(path: Path):
    """读取训练日志 CSV -> [{epoch, lr, train_loss, ...}]，容错返回 []。"""
    rows = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                row = {}
                for k, v in r.items():
                    try:
                        row[k] = float(v)
                    except (TypeError, ValueError):
                        row[k] = v
                rows.append(row)
    except Exception:
        pass
    return rows


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def collect_runs():
    """汇总所有训练/评测运行。status: running / finished。"""
    runs = []
    now = time.time()
    for csv_path in sorted(LOGS.glob("*.csv")):
        tag = csv_path.stem
        series = read_csv_series(csv_path)
        if not series:
            continue
        summary = read_json(LOGS / f"{tag}_summary.json") or {}
        mtime = csv_path.stat().st_mtime
        finished = bool(summary) or (now - mtime > 180)
        last = series[-1]
        best_dice = max((r.get("dice") or 0) for r in series)
        config = {k: v for k, v in summary.items() if k != "history"}
        runs.append({
            "tag": tag,
            "model": summary.get("model", tag.split("_")[0]),
            "seed": summary.get("seed"),
            "config": config,
            "epochs_done": int(last.get("epoch", len(series))),
            "total_epochs": summary.get("epochs"),
            "best_val_dice": round(best_dice, 4),
            "last": last,
            "cum_min": round(last.get("cum_min") or 0, 2),
            "status": "finished" if finished else "running",
            "updated": time.strftime("%m-%d %H:%M:%S", time.localtime(mtime)),
            "has_eval": (LOGS / f"eval_{tag}.json").exists(),
            "series": series,
        })
    return runs


def collect_evals():
    """测试集评测结果（eval_*.json）。"""
    out = []
    for p in sorted(LOGS.glob("eval_*.json")):
        d = read_json(p)
        if d:
            d["_file"] = p.name
            out.append(d)
    return out


def list_tree(base: Path, exts=None):
    items = []
    if not base.exists():
        return items
    for p in sorted(base.rglob("*")):
        if p.is_file() and (exts is None or p.suffix.lower() in exts):
            st = p.stat()
            items.append({
                "path": str(p.relative_to(ROOT)),
                "name": p.name,
                "size": st.st_size,
                "size_h": human_size(st.st_size),
                "mtime": time.strftime("%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
            })
    return items


def human_size(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f} {u}" if u != "B" else f"{n} B"
        n /= 1024


def safe_join(base: Path, rel: str) -> Path:
    """防目录穿越：解析后必须仍在 base 内。"""
    p = (ROOT / rel).resolve()
    if not str(p).startswith(str(ROOT.resolve())):
        raise PermissionError(rel)
    return p


def build_state():
    data_stats = read_json(REPORTS / "data_stats.json")
    return {
        "project": PROJECT,
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs": collect_runs(),
        "evals": collect_evals(),
        "checkpoints": list_tree(CKPT, {".pth", ".pt"}),
        "figures": list_tree(FIGS, {".png", ".jpg", ".svg"}),
        "reports": list_tree(REPORTS, {".md", ".json"}),
        "code": list_tree(SRC, {".py"}),
        "data_stats": data_stats,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 静音
        pass

    def _send(self, body: bytes, ctype="application/json; charset=utf-8", code=200,
              extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"), code=code)

    def do_GET(self):
        u = urlparse(self.path)
        path, qs = u.path, parse_qs(u.query)
        try:
            if path in ("/", "/index.html"):
                f = DASH / "index.html"
                self._send(f.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json(build_state())
            elif path == "/api/code":
                rel = qs.get("path", [""])[0]
                p = safe_join(SRC.parent, rel)
                if not str(p).startswith(str(SRC.resolve())):
                    raise PermissionError
                self._json({"path": rel, "content": p.read_text(encoding="utf-8", errors="replace")})
            elif path == "/api/report":
                rel = qs.get("path", [""])[0]
                p = safe_join(REPORTS.parent, rel)
                if not str(p).startswith(str(REPORTS.resolve())):
                    raise PermissionError
                self._json({"path": rel, "content": p.read_text(encoding="utf-8", errors="replace")})
            elif path.startswith("/figures/"):
                p = safe_join(FIGS.parent, str(Path("outputs/figures") / path.split("/")[-1]))
                ctype = "image/png" if p.suffix == ".png" else "image/jpeg"
                self._send(p.read_bytes(), ctype)
            elif path.startswith("/download/"):
                rel = qs.get("path", [""])[0]
                p = safe_join(CKPT.parent, rel)
                if not str(p).startswith(str(CKPT.resolve())) or not p.is_file():
                    raise PermissionError
                self._send(p.read_bytes(), "application/octet-stream",
                           extra={"Content-Disposition": f'attachment; filename="{p.name}"'})
            else:
                self._json({"error": "not found"}, 404)
        except PermissionError:
            self._json({"error": "forbidden"}, 403)
        except FileNotFoundError:
            self._json({"error": "file not found"}, 404)
        except Exception as e:  # noqa: BLE001
            self._json({"error": str(e)}, 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8801)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[hw1 dashboard] http://127.0.0.1:{args.port}  (Ctrl+C 停止)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
