#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
作业2（hw2_sam）监控仪表盘后端
================================
与 hw1 仪表盘同一套设计系统，但聚合逻辑针对 SAM 三阶段实验：
  - E1 零样本 / E2 冻结编码器微调 / E3 LoRA 微调的运行状态
  - 图像嵌入预计算缓存进度（train 2594 / val 100 / test 1000）
  - 评测 JSON（嵌套 metrics）展平、模型权重下载、代码与报告浏览

启动：  python dashboard/server.py [--port 8802]
"""
import argparse
import csv
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent.parent          # hw2_sam/
DASH = Path(__file__).resolve().parent
LOGS = ROOT / "outputs" / "logs"
CKPT = ROOT / "outputs" / "checkpoints"
FIGS = ROOT / "outputs" / "figures"
REPORTS = ROOT / "reports"
SRC = ROOT / "src"
EMB = ROOT / "data" / "embeddings"
SAM_WEIGHTS = ROOT / "data" / "raw" / "sam_vit_b_01ec64.pth"

EMB_TARGETS = {"train": 2594, "val": 100, "test": 1000}

PROJECT = {
    "id": "hw2",
    "title": "作业 2 · SAM 提示分割（Transformer）",
    "subtitle": "Segment Anything (ViT-B) · E1 零样本 / E2 冻结微调 / E3 LoRA · 与作业1级联",
    "env": "hw2-sam-transformer",
    "accent": "#bc8cff",
}

STAGE_META = {
    "e1": {"name": "E1 零样本评测", "desc": "原版 SAM 不训练，GT 外接矩形框提示，量化域差距"},
    "e2": {"name": "E2 冻结编码器微调", "desc": "冻结 ViT 图像编码器，仅训练 prompt encoder + mask decoder（MedSAM 式，约 4% 参数）"},
    "e3": {"name": "E3 LoRA 微调", "desc": "ViT 的 q/v 投影注入 LoRA(r=8)，连同 decoder 一起训练（SAMed 式）"},
}


def read_csv_series(path: Path):
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


def human_size(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f} {u}" if u != "B" else f"{n} B"
        n /= 1024


def collect_runs():
    """训练类运行（E2/E3 的 CSV 日志）。"""
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
        best_dice = max((r.get("val_dice") or r.get("dice") or 0) for r in series)
        stage = tag.split("_")[0] if tag[:2] in ("e1", "e2", "e3") else "other"
        runs.append({
            "tag": tag, "stage": stage,
            "model": summary.get("model", "SAM ViT-B"),
            "config": {k: v for k, v in summary.items() if k != "history"},
            "epochs_done": int(last.get("epoch", len(series))),
            "total_epochs": summary.get("epochs"),
            "best_val_dice": round(best_dice, 4),
            "last": last,
            "cum_min": round(last.get("cum_min") or 0, 2),
            "status": "finished" if finished else "running",
            "updated": time.strftime("%m-%d %H:%M:%S", time.localtime(mtime)),
            "series": series,
        })
    return runs


def collect_evals():
    """评测 JSON（E1/E2/E3 × 提示类型 × split），metrics 嵌套展平。"""
    out = []
    for p in sorted(LOGS.glob("eval_*.json")):
        d = read_json(p)
        if not d:
            continue
        m = d.get("metrics", {}) or {}
        out.append({
            "file": p.name,
            "stage": d.get("stage", "?"),
            "tag": d.get("tag", p.stem),
            "prompt": d.get("prompt", "?"),
            "split": d.get("split", "?"),
            "n": d.get("n", m.get("n")),
            "dice": m.get("dice"), "dice_std": m.get("dice_std"),
            "iou": m.get("iou"), "hd95": m.get("hd95"),
            "sensitivity": m.get("sensitivity"), "specificity": m.get("specificity"),
            "pixel_acc": m.get("pixel_acc"),
            "latency_ms": m.get("latency_ms_per_img") or m.get("latency_ms"),
            "mtime": time.strftime("%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)),
        })
    return out


def collect_embeddings():
    """嵌入缓存进度：已生成文件数 / 目标数。"""
    out = {}
    for split, target in EMB_TARGETS.items():
        d = EMB / split
        done = len(list(d.glob("*.npy"))) + len(list(d.glob("*.npz"))) + \
               len(list(d.glob("*.pt"))) if d.exists() else 0
        # 缓存可能是任意扩展名，退化为直接数文件
        if d.exists():
            done = len([p for p in d.iterdir() if p.is_file()])
        size = sum(p.stat().st_size for p in d.iterdir() if p.is_file()) if d.exists() else 0
        out[split] = {"done": done, "target": target,
                      "pct": round(done / target * 100, 1), "size_h": human_size(size)}
    return out


def collect_stages(runs, evals):
    """E1/E2/E3 三阶段状态汇总。"""
    stages = {}
    for sid, meta in STAGE_META.items():
        s_runs = [r for r in runs if r["stage"] == sid]
        s_evals = [e for e in evals if e["stage"] == sid]
        best = None
        for e in s_evals:
            if e["dice"] is not None and (best is None or e["dice"] > best["dice"]):
                best = e
        running = any(r["status"] == "running" for r in s_runs)
        stages[sid] = {**meta,
                       "status": "running" if running else ("done" if (s_evals or s_runs) else "pending"),
                       "n_runs": len(s_runs), "n_evals": len(s_evals),
                       "best_eval": best}
    return stages


def list_tree(base: Path, exts=None):
    items = []
    if not base.exists():
        return items
    for p in sorted(base.rglob("*")):
        if p.is_file() and (exts is None or p.suffix.lower() in exts):
            st = p.stat()
            items.append({"path": str(p.relative_to(ROOT)), "name": p.name,
                          "size": st.st_size, "size_h": human_size(st.st_size),
                          "mtime": time.strftime("%m-%d %H:%M:%S", time.localtime(st.st_mtime))})
    return items


def safe_join(rel: str) -> Path:
    p = (ROOT / rel).resolve()
    if not str(p).startswith(str(ROOT.resolve())):
        raise PermissionError(rel)
    return p


def build_state():
    runs = collect_runs()
    evals = collect_evals()
    # SAM 原版权重也列入可下载产物
    weights = list_tree(CKPT, {".pth", ".pt"})
    if SAM_WEIGHTS.exists():
        st = SAM_WEIGHTS.stat()
        weights.insert(0, {"path": str(SAM_WEIGHTS.relative_to(ROOT)),
                           "name": SAM_WEIGHTS.name + "（SAM 原版 ViT-B 权重）",
                           "size": st.st_size, "size_h": human_size(st.st_size),
                           "mtime": time.strftime("%m-%d %H:%M:%S", time.localtime(st.st_mtime))})
    return {
        "project": PROJECT,
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stages": collect_stages(runs, evals),
        "embeddings": collect_embeddings(),
        "runs": runs,
        "evals": evals,
        "checkpoints": weights,
        "figures": list_tree(FIGS, {".png", ".jpg", ".svg"}),
        "reports": list_tree(REPORTS, {".md", ".json"}),
        "code": list_tree(SRC, {".py"}),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype="application/json; charset=utf-8", code=200, extra=None):
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
                self._send((DASH / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json(build_state())
            elif path == "/api/code":
                rel = qs.get("path", [""])[0]
                p = safe_join(rel)
                if not str(p).startswith(str(SRC.resolve())):
                    raise PermissionError
                self._json({"path": rel, "content": p.read_text(encoding="utf-8", errors="replace")})
            elif path == "/api/report":
                rel = qs.get("path", [""])[0]
                p = safe_join(rel)
                if not str(p).startswith(str(REPORTS.resolve())):
                    raise PermissionError
                self._json({"path": rel, "content": p.read_text(encoding="utf-8", errors="replace")})
            elif path == "/api/eval":
                f = qs.get("file", [""])[0]
                p = safe_join(f"outputs/logs/{f}")
                self._json(read_json(p) or {})
            elif path.startswith("/figures/"):
                p = safe_join(str(Path("outputs/figures") / path.split("/")[-1]))
                ctype = "image/png" if p.suffix == ".png" else "image/jpeg"
                self._send(p.read_bytes(), ctype)
            elif path.startswith("/download/"):
                rel = qs.get("path", [""])[0]
                p = safe_join(rel)
                allowed = str(CKPT.resolve())
                if not (str(p).startswith(allowed) or p == SAM_WEIGHTS.resolve()) or not p.is_file():
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
    ap.add_argument("--port", type=int, default=8802)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[hw2 dashboard] http://127.0.0.1:{args.port}  (Ctrl+C 停止)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
