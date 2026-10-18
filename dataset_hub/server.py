#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MedSeg Dataset Hub —— 科研数据集展示平台后端
端口 8800 · 仅 Python 标准库 · ThreadingHTTPServer · 绑定 127.0.0.1

路由：
  GET  /                     静态前端（static/index.html，单文件 SPA）
  GET  /static/<file>        静态资源（限 static/ 目录内，防目录穿越）
  GET  /api/overview         平台总览数字 + AI 助手可用性
  GET  /api/datasets         数据集全量（data/datasets.json）
  GET  /api/glossary         术语表全量（data/glossary.json）
  GET  /api/check_link?url=  上游链接可达性探测（HEAD 优先，10s 超时）
  GET  /api/hw1              作业1 实验聚合（只读 hw1_unet/outputs + reports）
  GET  /api/hw2              作业2 实验聚合（只读 hw2_sam/outputs + 嵌入缓存）
  GET  /api/hw_report        作业报告内容（限 reports/ 目录）
  GET  /api/hw_code          作业源码内容（限 src/ 目录）
  GET  /api/hw_eval          评测 JSON 原文（限 outputs/logs/）
  GET  /exp/<proj>/figures/<f>   实验图片（只读 outputs/figures/）
  GET  /exp/<proj>/download      大文件流式下载（限 checkpoints/ 与 SAM 原版权重）
  POST /api/chat             DeepSeek 流式问答代理（SSE 原样透传）

配置：
  DEEPSEEK_API_KEY / DEEPSEEK_MODEL —— 先读环境变量，再解析同目录 .env
  （os.environ.setdefault，不覆盖已有环境变量）
"""

import csv
import json
import math
import os
import shutil
import socket
import sys
import time
import urllib.parse
from datetime import datetime
from http.client import HTTPConnection, HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")

HOST, PORT = "127.0.0.1", 8800
LINK_CHECK_TIMEOUT = 10      # /api/check_link 超时（秒）
CHAT_TIMEOUT = 90            # DeepSeek 上游连接/读超时（秒）
MAX_BODY_BYTES = 1_000_000   # 请求体上限 1MB
MAX_HISTORY_MESSAGES = 20    # 聊天历史保留条数

DEEPSEEK_HOST = "api.deepseek.com"
DEEPSEEK_PATH = "/chat/completions"

FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
    '<rect width="64" height="64" rx="14" fill="#1a73e8"/>'
    '<text x="32" y="43" font-size="34" text-anchor="middle">🔬</text></svg>'
).encode("utf-8")

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}

SYSTEM_PROMPT = """你是「MedSeg Dataset Hub」科研数据集展示平台内置的研究助手，服务一个皮肤病变分割研究项目。

【项目背景】
- 作业1（hw1_unet）：在 ISIC 2018 Task 1 皮肤镜数据集上手写实现并对比三个分割模型：U-Net、ResUNet、Attention U-Net；评价指标为 Dice / IoU / HD95。
- 作业2（hw2_sam）：基于 SAM ViT-B（375MB，91M 参数）的三阶段实验：E1 零样本评测（量化 SAM 从自然图像到皮肤镜的域差距）、E2 冻结图像编码器只微调提示编码器与掩膜解码器、E3 用 LoRA(r=8) 微调 ViT 的 q/v 投影（约 0.5M 可训练参数）。
- 核心叙事「级联系统」：U-Net 小模型全自动出粗框 → 作为提示输入 SAM 精修——小模型提框、大模型精修，分工而非替代。

【平台收录内容】
6 个条目：ISIC 2018 Task 1（主数据集，train 2594 / val 100 / test 1000，前景占比均值约 23.3%）、ISIC 2019、HAM10000、PH²、SA-1B（背景知识）、SAM ViT-B 预训练权重；每个条目含官方/镜像下载链接（镜像：hf-mirror、ModelScope）、许可证与官方榜单来源；另有 30+ 条术语解释（Dice、IoU、HD95、LoRA、提示分割、零样本等）。

【实验观测视图】
平台还实时聚合两个作业的实验产物（只读）："作业 1" 视图展示训练运行（tag/最优 Dice/损失曲线对比，如 diag_1e-3_amp 等学习率与 AMP 诊断）、数据统计与 checkpoint；"作业 2" 视图展示 E1/E2/E3 三阶段状态（E1 零样本已有 GT 框提示评测：val 5 张 Dice 约 0.757、HD95 约 14.6）、嵌入缓存进度（train/val/test 已 100% 预计算）。用户问实验进展时按这些视图的实际数据回答，没跑的实验（如 E2/E3）如实说"尚未运行"。

【回答要求】
1. 始终用简体中文，专业且通俗易懂，回答尽量简洁（聊天面板场景，可用 Markdown 粗体与行内代码）。
2. 涉及数据集下载：优先说明官方直链，官方不可达时提示镜像（hf-mirror / ModelScope），并提醒许可证限制（ISIC 系列为 CC BY-NC 4.0，仅限非商业研究；SAM 权重为 Apache 2.0）。
3. 绝不编造榜单排名、指标数字或许可证条款；不确定时明确说"平台数据中没有收录，建议查阅官方来源"。"""


# ---------------------------------------------------------------- 配置加载

def load_env():
    """解析同目录 .env（KEY=VALUE，支持 # 注释）。已有环境变量优先，不覆盖。"""
    path = os.path.join(BASE_DIR, ".env")
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(key, value)
    except OSError as exc:
        print(f"[warn] 读取 .env 失败：{exc}", file=sys.stderr)


def read_json(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 链接探测

def _open_conn(parsed, timeout):
    if parsed.scheme == "https":
        return HTTPSConnection(parsed.hostname, parsed.port or 443, timeout=timeout)
    return HTTPConnection(parsed.hostname, parsed.port or 80, timeout=timeout)


def _to_int(value):
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _content_range_total(crange):
    """'bytes 0-0/12345' -> 12345"""
    if not crange or "/" not in crange:
        return None
    return _to_int(crange.rsplit("/", 1)[-1])


def check_link(url):
    """HEAD 探测上游链接可达性。返回 {ok, status, size_bytes, message}（中文原因分类）。"""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return {"ok": False, "status": None, "size_bytes": None,
                "message": "链接格式无效：仅支持 http/https 地址"}

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) DatasetHub/1.0",
        "Accept": "*/*",
    }
    try:
        hops = 0
        while True:
            path = parsed.path or "/"
            if parsed.query:
                path += "?" + parsed.query
            conn = _open_conn(parsed, LINK_CHECK_TIMEOUT)
            try:
                conn.request("HEAD", path, headers=headers)
                resp = conn.getresponse()
                resp.read()
                status = resp.status
                clen = resp.getheader("Content-Length")
                crange = resp.getheader("Content-Range")
                loc = resp.getheader("Location")
            finally:
                conn.close()

            # 跟随至多 2 次重定向
            if status in (301, 302, 303, 307, 308) and loc and hops < 2:
                hops += 1
                new = urllib.parse.urlsplit(urllib.parse.urljoin(url, loc))
                if new.scheme in ("http", "https"):
                    parsed = new
                    continue
            break

        if 200 <= status < 300:
            return {"ok": True, "status": status, "size_bytes": _to_int(clen),
                    "message": "链接可达（HEAD 探测成功）"}

        # 部分站点（如需登录的页面、禁用 HEAD 的静态站）退化为 Range GET 探测
        if status in (403, 405, 501):
            get_headers = dict(headers)
            get_headers["Range"] = "bytes=0-0"
            path = parsed.path or "/"
            if parsed.query:
                path += "?" + parsed.query
            conn = _open_conn(parsed, LINK_CHECK_TIMEOUT)
            try:
                conn.request("GET", path, headers=get_headers)
                resp = conn.getresponse()
                resp.read()
                status2 = resp.status
                clen2 = resp.getheader("Content-Length")
                crange2 = resp.getheader("Content-Range")
            finally:
                conn.close()
            if status2 in (200, 206):
                size = _content_range_total(crange2) or _to_int(clen2)
                return {"ok": True, "status": status2, "size_bytes": size,
                        "message": "链接可达（该站点不支持 HEAD，已改用 GET 探测成功）"}
            return {"ok": False, "status": status, "size_bytes": None,
                    "message": f"服务器拒绝访问（HTTP {status}）：该资源可能需要登录或有限流，可尝试下方镜像源"}

        if 400 <= status < 500:
            return {"ok": False, "status": status, "size_bytes": None,
                    "message": f"服务器返回 HTTP {status}：资源不存在或无权限访问，可尝试下方镜像源"}
        return {"ok": False, "status": status, "size_bytes": None,
                "message": f"服务器返回 HTTP {status}：服务端异常，可稍后重试或尝试镜像源"}

    except socket.timeout:
        return {"ok": False, "status": None, "size_bytes": None,
                "message": "探测超时（10 秒内无响应）：网络不可达或对方服务缓慢，可尝试下方镜像源"}
    except socket.gaierror:
        return {"ok": False, "status": None, "size_bytes": None,
                "message": "DNS 解析失败：域名不存在或本机当前无法解析（可检查网络/代理设置）"}
    except ConnectionRefusedError:
        return {"ok": False, "status": None, "size_bytes": None,
                "message": "连接被拒绝：目标服务未在该端口响应"}
    except OSError as exc:
        return {"ok": False, "status": None, "size_bytes": None,
                "message": f"网络错误（{exc.__class__.__name__}）：连接未能建立，可尝试下方镜像源"}


# ---------------------------------------------------------------- 实验聚合（只读引用 hw1/hw2 产物）

WORKSPACE = Path(BASE_DIR).parent
HW1_DIR = WORKSPACE / "hw1_unet"
HW2_DIR = WORKSPACE / "hw2_sam"

HW1_PROJECT = {
    "id": "hw1",
    "title": "作业 1 · 手写 U-Net 系列分割",
    "subtitle": "ISIC 2018 Task 1 · U-Net / ResUNet / Attention U-Net 三模型对比",
}
HW2_PROJECT = {
    "id": "hw2",
    "title": "作业 2 · SAM 提示分割（Transformer）",
    "subtitle": "SAM ViT-B · E1 零样本 / E2 冻结微调 / E3 LoRA · 与作业 1 级联",
}
STAGE_META = {
    "e1": {"name": "E1 零样本评测", "desc": "原版 SAM 不训练，GT 外接矩形框提示，量化域差距"},
    "e2": {"name": "E2 冻结编码器微调", "desc": "冻结 ViT 编码器，仅训练提示编码器 + 掩膜解码器（约 4% 参数）"},
    "e3": {"name": "E3 LoRA 微调", "desc": "ViT q/v 投影注入 LoRA(r=8)，连同解码器一起训练（约 0.5M 可训练参数）"},
}
EMB_TARGETS = {"train": 2594, "val": 100, "test": 1000}
FIG_EXTS = {".png", ".jpg", ".jpeg", ".svg"}
FIG_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml"}


def _read_json_file(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _sanitize_json(v):
    """把 NaN / Infinity 替换为 null。

    实验评测结果里可能出现 NaN（如预测全前景时 specificity 0/0）。
    Python 的 json.dumps 会把它输出成字面量 NaN——那不是合法 JSON，
    浏览器 res.json() 会解析失败，导致前端拿到 null。
    """
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _sanitize_json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_sanitize_json(x) for x in v]
    return v


def _read_csv_series(path):
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


def _list_tree(base, exts=None, rel_to=None):
    """列出目录树。path 相对 rel_to（应为项目根），保证与 /exp/<proj>/download 的守卫口径一致。"""
    items = []
    base = Path(base)
    rel_to = Path(rel_to) if rel_to else base.parent
    if not base.exists():
        return items
    for p in sorted(base.rglob("*")):
        if p.is_file() and (exts is None or p.suffix.lower() in exts):
            st = p.stat()
            items.append({
                "path": str(p.relative_to(rel_to)),
                "name": p.name,
                "size": st.st_size,
                "size_h": _human_size(st.st_size),
                "mtime": time.strftime("%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
            })
    return items


def _human_size(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f} {u}" if u != "B" else f"{n} B"
        n /= 1024


def _agg_runs(logs_dir, with_stage=False):
    """扫描 outputs/logs 下的训练 CSV + 摘要 JSON，与两个监控台同一套聚合口径。"""
    runs = []
    logs_dir = Path(logs_dir)
    if not logs_dir.exists():
        return runs
    now = time.time()
    for csv_path in sorted(logs_dir.glob("*.csv")):
        series = _read_csv_series(csv_path)
        tag = csv_path.stem
        summary = _read_json_file(logs_dir / f"{tag}_summary.json") or {}
        # 防御：历史上有过续跑把 CSV 截断的 bug（CSVLogger 用 "w" 模式），
        # summary 的 history 才是完整记录。哪个更长就用哪个。
        hist = summary.get("history") or []
        if len(hist) > len(series):
            series = hist
        if not series:
            continue
        mtime = csv_path.stat().st_mtime
        last = series[-1]
        best = max((r.get("val_dice") or r.get("dice") or 0) for r in series)
        run = {
            "tag": tag,
            "model": summary.get("model", tag.split("_")[0]),
            "config": {k: v for k, v in summary.items() if k != "history"},
            "epochs_done": int(last.get("epoch", len(series))),
            "total_epochs": summary.get("epochs"),
            "best_val_dice": round(best, 4),
            "last": last,
            "cum_min": round(last.get("cum_min") or 0, 2),
            "status": "finished" if (summary.get("stop_reason") or bool(summary) or now - mtime > 180) else "running",
            "updated": time.strftime("%m-%d %H:%M:%S", time.localtime(mtime)),
            "series": series,
        }
        if with_stage:
            run["stage"] = tag.split("_")[0] if tag[:2] in ("e1", "e2", "e3") else "other"
        runs.append(run)
    return runs


def _agg_evals_hw1(logs_dir):
    out = []
    for p in sorted(Path(logs_dir).glob("eval_*.json")):
        d = _read_json_file(p)
        if d:
            out.append({"file": p.name, **d})
    return out


def _agg_evals_hw2(logs_dir):
    """评测 JSON（E1/E2/E3 × 提示 × split），metrics 嵌套展平。"""
    out = []
    for p in sorted(Path(logs_dir).glob("eval_*.json")):
        d = _read_json_file(p)
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


def _agg_embeddings(hw2_dir):
    out = {}
    for split, target in EMB_TARGETS.items():
        d = Path(hw2_dir) / "data" / "embeddings" / split
        done = size = 0
        if d.exists():
            files = [p for p in d.iterdir() if p.is_file()]
            done = len(files)
            size = sum(p.stat().st_size for p in files)
        out[split] = {"done": done, "target": target,
                      "pct": round(done / target * 100, 1),
                      "size": size, "size_h": _human_size(size)}
    return out


def _hw_stages(runs, evals):
    stages = {}
    for sid, meta in STAGE_META.items():
        s_runs = [r for r in runs if r.get("stage") == sid]
        s_evals = [e for e in evals if e.get("stage") == sid]
        best = None
        for e in s_evals:
            if e.get("dice") is not None and (best is None or e["dice"] > best["dice"]):
                best = e
        running = any(r["status"] == "running" for r in s_runs)
        stages[sid] = {**meta,
                       "status": "running" if running else ("done" if (s_evals or s_runs) else "pending"),
                       "n_runs": len(s_runs), "n_evals": len(s_evals),
                       "best_eval": best}
    return stages


def hw1_state():
    logs = HW1_DIR / "outputs" / "logs"
    runs = _agg_runs(logs)
    return {
        "project": HW1_PROJECT,
        "server_time": datetime.now().isoformat(timespec="seconds"),
        "runs": runs,
        "evals": _agg_evals_hw1(logs),
        # path 相对项目根（outputs/checkpoints/…），与 /exp/<proj>/download 的守卫口径一致
        "checkpoints": _list_tree(HW1_DIR / "outputs" / "checkpoints", {".pth", ".pt"}, rel_to=HW1_DIR),
        "figures": _list_tree(HW1_DIR / "outputs" / "figures", FIG_EXTS, rel_to=HW1_DIR),
        "reports": _list_tree(HW1_DIR / "reports", {".md", ".json"}, rel_to=HW1_DIR),
        "code": _list_tree(HW1_DIR / "src", {".py"}, rel_to=HW1_DIR),
        "predictions": _list_tree(HW1_DIR / "outputs" / "predictions", {".json"}, rel_to=HW1_DIR),
        "data_stats": _read_json_file(HW1_DIR / "reports" / "data_stats.json"),
        "ablation": _read_json_file(HW1_DIR / "outputs" / "ablation.json") or {},
    }


def hw2_state():
    logs = HW2_DIR / "outputs" / "logs"
    runs = _agg_runs(logs, with_stage=True)
    evals = _agg_evals_hw2(logs)
    weights = _list_tree(HW2_DIR / "outputs" / "checkpoints", {".pth", ".pt"}, rel_to=HW2_DIR)
    sam_w = HW2_DIR / "data" / "raw" / "sam_vit_b_01ec64.pth"
    if sam_w.exists():
        st = sam_w.stat()
        weights.insert(0, {"path": str(sam_w.relative_to(HW2_DIR)),
                           "name": sam_w.name + "（SAM 原版 ViT-B 权重）",
                           "size": st.st_size, "size_h": _human_size(st.st_size),
                           "mtime": time.strftime("%m-%d %H:%M:%S", time.localtime(st.st_mtime))})
    return {
        "project": HW2_PROJECT,
        "server_time": datetime.now().isoformat(timespec="seconds"),
        "stages": _hw_stages(runs, evals),
        "embeddings": _agg_embeddings(HW2_DIR),
        "runs": runs,
        "evals": evals,
        "checkpoints": weights,
        "figures": _list_tree(HW2_DIR / "outputs" / "figures", FIG_EXTS, rel_to=HW2_DIR),
        "reports": _list_tree(HW2_DIR / "reports", {".md", ".json"}, rel_to=HW2_DIR),
        "code": _list_tree(HW2_DIR / "src", {".py"}, rel_to=HW2_DIR),
    }


# ---------------------------------------------------------------- 请求处理

class Handler(BaseHTTPRequestHandler):
    server_version = "DatasetHub/1.0"
    protocol_version = "HTTP/1.1"

    # ---- 基础输出 ----
    def _send(self, status, ctype, body, extra=None):
        if getattr(self, "_head_only", False):
            body = b""  # HEAD：头照发、体不写
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _send_json(self, obj, status=200):
        payload = json.dumps(_sanitize_json(obj), ensure_ascii=False).encode("utf-8")
        self._send(status, "application/json; charset=utf-8", payload)

    def _send_error_json(self, status, message):
        self._send_json({"error": message}, status)

    def log_message(self, fmt, *args):
        sys.stdout.write("[%s] %s\n" % (datetime.now().strftime("%H:%M:%S"), fmt % args))
        sys.stdout.flush()

    # ---- GET / HEAD ----
    def do_GET(self):
        self._handle_get(head=False)

    def do_HEAD(self):
        # 与 GET 同路由，只回头部不写 body（并保证 no-store 等响应头一致）
        self._handle_get(head=True)

    def _handle_get(self, head=False):
        self._head_only = head
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path
        try:
            if path in ("/", "/index.html"):
                return self._serve_static("index.html")
            if path == "/favicon.ico":
                return self._send(200, "image/svg+xml", FAVICON_SVG)
            if path.startswith("/static/"):
                return self._serve_static(path[len("/static/"):])
            if path == "/api/overview":
                return self._api_overview()
            if path == "/api/datasets":
                return self._send_json(read_json("datasets.json"))
            if path == "/api/glossary":
                return self._send_json(read_json("glossary.json"))
            if path == "/api/check_link":
                return self._api_check_link(parsed)
            if path == "/api/hw1":
                return self._send_json(hw1_state())
            if path == "/api/hw2":
                return self._send_json(hw2_state())
            if path == "/api/hw_report":
                return self._api_hw_report(parsed)
            if path == "/api/hw_code":
                return self._api_hw_code(parsed)
            if path == "/api/hw_eval":
                return self._api_hw_eval(parsed)
            if path.startswith("/exp/") and "/figures/" in path:
                return self._serve_exp_figure(path)
            if path.startswith("/exp/") and path.endswith("/download"):
                return self._api_exp_download(parsed)
            return self._send_error_json(404, f"未找到该路径：{path}（可用接口见 README）")
        except BrokenPipeError:
            raise
        except Exception as exc:  # 统一兜底，保证前端不白屏
            sys.stderr.write(f"[error] {path}: {exc.__class__.__name__}: {exc}\n")
            return self._send_error_json(500, f"服务器内部错误（{exc.__class__.__name__}），请查看终端日志")

    def _serve_static(self, rel):
        root = os.path.realpath(STATIC_DIR)
        target = os.path.realpath(os.path.join(root, rel))
        if not target.startswith(root + os.sep) or not os.path.isfile(target):
            return self._send_error_json(404, f"静态文件不存在：{rel}")
        ext = os.path.splitext(target)[1].lower()
        with open(target, "rb") as f:
            body = f.read()
        self._send(200, MIME.get(ext, "application/octet-stream"), body)

    def _api_overview(self):
        datasets = read_json("datasets.json")
        glossary = read_json("glossary.json")
        self._send_json({
            "datasets": len(datasets.get("datasets", [])),
            "terms": len(glossary.get("terms", [])),
            "leaderboard_sources": sum(len(d.get("leaderboards", []))
                                       for d in datasets.get("datasets", [])),
            "chat_enabled": bool(os.environ.get("DEEPSEEK_API_KEY")),
            "server_time": datetime.now().isoformat(timespec="seconds"),
        })

    def _api_check_link(self, parsed):
        qs = urllib.parse.parse_qs(parsed.query)
        url = (qs.get("url") or [""])[0].strip()
        if not url:
            return self._send_error_json(400, "缺少 url 参数，用法：/api/check_link?url=https://…")
        result = check_link(url)
        self._send_json(result)

    # ---- 实验聚合辅助（只读 hw1/hw2）----
    def _exp_root(self, proj):
        if proj == "hw1":
            return HW1_DIR
        if proj == "hw2":
            return HW2_DIR
        return None

    def _api_hw_report(self, parsed):
        """返回 hw1/hw2 下 reports/ 目录中某个文本文件的内容（只读、防穿越）。"""
        qs = urllib.parse.parse_qs(parsed.query)
        proj = (qs.get("proj") or [""])[0]
        rel = (qs.get("path") or [""])[0]
        root = self._exp_root(proj)
        if root is None or not rel:
            return self._send_error_json(400, "用法：/api/hw_report?proj=hw1&path=reports/data_stats.md")
        reports = (root / "reports").resolve()
        target = (root / rel).resolve()
        if not str(target).startswith(str(reports) + os.sep) or not target.is_file():
            return self._send_error_json(404, f"报告不存在或不在 reports/ 内：{rel}")
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return self._send_error_json(500, f"报告读取失败：{exc}")
        self._send_json({"path": rel, "content": content})

    def _api_hw_code(self, parsed):
        """返回 hw1/hw2 下 src/ 目录中某个源码文件的内容（只读、防穿越）。"""
        qs = urllib.parse.parse_qs(parsed.query)
        proj = (qs.get("proj") or [""])[0]
        rel = (qs.get("path") or [""])[0]
        root = self._exp_root(proj)
        if root is None or not rel:
            return self._send_error_json(400, "用法：/api/hw_code?proj=hw1&path=src/train.py")
        src = (root / "src").resolve()
        target = (root / rel).resolve()
        if not str(target).startswith(str(src) + os.sep) or not target.is_file():
            return self._send_error_json(404, f"源码不存在或不在 src/ 内：{rel}")
        if target.suffix.lower() != ".py":
            return self._send_error_json(404, "仅支持查看 .py 源码")
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return self._send_error_json(500, f"源码读取失败：{exc}")
        self._send_json({"path": rel, "content": content, "lines": content.count("\n") + 1})

    def _api_hw_eval(self, parsed):
        """返回 hw1/hw2 下 outputs/logs/ 中某个评测 JSON 的原文（只读、防穿越）。"""
        qs = urllib.parse.parse_qs(parsed.query)
        proj = (qs.get("proj") or [""])[0]
        name = (qs.get("file") or [""])[0]
        root = self._exp_root(proj)
        if root is None or not name:
            return self._send_error_json(400, "用法：/api/hw_eval?proj=hw1&file=eval_unet_s42_test.json")
        logs = (root / "outputs" / "logs").resolve()
        target = (logs / name).resolve()
        if not str(target).startswith(str(logs) + os.sep) or not target.is_file():
            return self._send_error_json(404, f"评测文件不存在：{name}")
        data = _read_json_file(target)
        if data is None:
            return self._send_error_json(500, f"评测文件解析失败：{name}")
        self._send_json({"file": name, "data": data})

    def _serve_exp_figure(self, path):
        """/exp/<proj>/figures/<文件名> —— 只读输出 figures/ 目录下的图片。"""
        parts = path.split("/")          # ["", "exp", proj, "figures", name]
        if len(parts) != 5:
            return self._send_error_json(404, "图片路径格式：/exp/hw1/figures/<文件名>")
        root = self._exp_root(parts[2])
        name = parts[4]
        if root is None:
            return self._send_error_json(404, f"未知项目：{parts[2]}")
        figs = (root / "outputs" / "figures").resolve()
        target = (figs / name).resolve()
        if not str(target).startswith(str(figs) + os.sep) or not target.is_file():
            return self._send_error_json(404, f"图片不存在：{name}")
        ext = target.suffix.lower()
        if ext not in FIG_MIME:
            return self._send_error_json(404, "仅支持 png/jpg/svg 图片")
        self._send(200, FIG_MIME[ext], target.read_bytes())

    def _api_exp_download(self, parsed):
        """大文件（checkpoints）流式下载：只允许 outputs/checkpoints/ 与 hw2 的 SAM 原版权重。
        proj 可写在路径段（/exp/hw1/download）或查询串（/exp/download?proj=hw1），两种形式等价；
        路径段存在但不是 hw1/hw2 时直接 404，不静默回退到查询参数。"""
        qs = urllib.parse.parse_qs(parsed.query)
        parts = parsed.path.split("/")          # 路径段形式 ["", "exp", proj, "download"]；查询串形式 ["", "exp", "download"]
        if len(parts) == 4:
            if parts[2] not in ("hw1", "hw2"):
                return self._send_error_json(404, f"未知项目：{parts[2]}（仅 hw1 / hw2）")
            proj = parts[2]
        else:
            proj = (qs.get("proj") or [""])[0]
        rel = (qs.get("path") or [""])[0]
        root = self._exp_root(proj)
        if root is None or not rel:
            return self._send_error_json(400, "用法：/exp/hw1/download?proj=hw1&path=outputs/checkpoints/xxx.pth")
        ckpt = (root / "outputs" / "checkpoints").resolve()
        target = (root / rel).resolve()
        sam_w = (HW2_DIR / "data" / "raw" / "sam_vit_b_01ec64.pth").resolve() if proj == "hw2" else None
        allowed = str(target).startswith(str(ckpt) + os.sep) or (sam_w and target == sam_w)
        if not allowed or not target.is_file():
            return self._send_error_json(404, "文件不存在或不在允许的目录内")
        size = target.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Disposition", f'attachment; filename="{target.name}"')
        self.end_headers()
        with open(target, "rb") as f:
            shutil.copyfileobj(f, self.wfile, 1024 * 1024)  # 1MB 流式，不占内存

    # ---- POST /api/chat（DeepSeek SSE 透传代理）----
    def do_POST(self):
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path != "/api/chat":
            return self._send_error_json(404, f"不支持的 POST 路径：{parsed.path}")
        try:
            return self._handle_chat()
        except BrokenPipeError:
            self.close_connection = True
        except Exception as exc:
            sys.stderr.write(f"[error] /api/chat: {exc.__class__.__name__}: {exc}\n")
            try:
                self._send_error_json(500, f"服务器内部错误（{exc.__class__.__name__}），请查看终端日志")
            except OSError:
                self.close_connection = True

    def _read_body(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            raise ValueError("请求体为空")
        if length > MAX_BODY_BYTES:
            raise ValueError("请求体过大（上限 1MB）")
        return self.rfile.read(length)

    def _handle_chat(self):
        try:
            raw = json.loads(self._read_body().decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            return self._send_error_json(400, f"请求体不是合法 JSON：{exc}")

        incoming = raw.get("messages") if isinstance(raw, dict) else None
        if not isinstance(incoming, list):
            return self._send_error_json(400, '请求体格式应为 {"messages": [{role, content}, …]}')
        # 只保留 user/assistant 轮次（system 提示词由服务端注入，防止客户端覆盖）
        history = [{"role": m.get("role"), "content": str(m.get("content", ""))}
                   for m in incoming
                   if isinstance(m, dict) and m.get("role") in ("user", "assistant")
                   and str(m.get("content", "")).strip()]
        if not history:
            return self._send_error_json(400, "messages 不能为空（至少包含一条 user 消息）")
        history = history[-MAX_HISTORY_MESSAGES:]

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            # 以 SSE 错误事件返回，前端可统一渲染
            self._begin_sse()
            return self._sse({"error": "AI 助手未启用：未检测到 DEEPSEEK_API_KEY。"
                                       "请在 dataset_hub/.env 中配置后重启服务（参见 .env.example）"})

        model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + history,
            "stream": True,
            "temperature": 0.3,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {api_key}",
        }

        self._begin_sse()
        conn = HTTPSConnection(DEEPSEEK_HOST, timeout=CHAT_TIMEOUT)
        try:
            conn.request("POST", DEEPSEEK_PATH, body=body, headers=headers)
            resp = conn.getresponse()
            if resp.status != 200:
                raw_err = resp.read(2000).decode("utf-8", "replace")
                detail = raw_err
                try:
                    err = json.loads(raw_err).get("error")
                    if isinstance(err, dict):
                        detail = err.get("message") or raw_err
                except ValueError:
                    pass
                friendly = {
                    401: f"DeepSeek API 认证失败（401）：{detail}。请检查 dataset_hub/.env 中的 DEEPSEEK_API_KEY",
                    402: f"DeepSeek 账户余额不足（402）：{detail}",
                    422: f"DeepSeek 拒绝了请求参数（422）：{detail}",
                    429: f"请求过于频繁（429）：{detail}，请稍后再试",
                }.get(resp.status, f"DeepSeek API 返回 HTTP {resp.status}：{detail}")
                return self._sse({"error": friendly})

            # SSE 原样透传：逐行读、逐行写、逐行 flush
            while True:
                line = resp.readline()
                if not line:
                    break
                self.wfile.write(line)
                self.wfile.flush()
        except socket.timeout:
            self._sse({"error": "连接 DeepSeek 超时：网络不稳定或服务响应缓慢，请稍后重试"})
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True  # 客户端已断开（如点了停止）
        except OSError as exc:
            self._sse({"error": f"连接 DeepSeek 失败（{exc.__class__.__name__}）：请检查本机网络后重试"})
        finally:
            conn.close()

    def _begin_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _sse(self, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.wfile.write(b"data: " + data + b"\n\n")
        self.wfile.flush()


# ---------------------------------------------------------------- 入口

def main():
    load_env()
    has_key = bool(os.environ.get("DEEPSEEK_API_KEY"))
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True

    print("=" * 62)
    print("  MedSeg Dataset Hub · 科研数据集展示平台")
    print(f"  地址        http://{HOST}:{PORT}/")
    print(f"  数据        {DATA_DIR}")
    print(f"  AI 助手     {'已启用 · model=' + model if has_key else '未启用（缺少 DEEPSEEK_API_KEY）'}")
    print("  配置方式    dataset_hub/.env 或环境变量（详见 README）")
    print("  停止        Ctrl+C")
    print("=" * 62)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[stop] 服务已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
