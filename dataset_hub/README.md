# MedSeg Dataset Hub · 一站式科研门户

> 为**皮肤病变分割研究项目**服务的单端口门户：研究物料（数据集、权重、许可证、官方榜单、术语）
> **+ 两个作业的完整实验观测**（训练曲线、评测档案、权重下载、源码浏览、图表画廊）
> **+ DeepSeek 流式问答助手**，全部收在一个支持**浅色 / 深色双主题**的 Web 平台里。

- 访问地址：<http://127.0.0.1:8800/>（启动后浏览器打开即可）
- **单一服务**：原先的两个暗色监控台（8801/8802）已废弃，功能全部并入本门户的 `#/hw1` 与 `#/hw2`
  （迁移对照见 [hw1](../hw1_unet/dashboard/DEPRECATED.md) / [hw2](../hw2_sam/dashboard/DEPRECATED.md)）

---

## 核心功能

| 功能 | 说明 |
| --- | --- |
| 🗂 数据集下载中心 | 6 个条目（ISIC 2018 Task 1 / ISIC 2019 / HAM10000 / PH² / SA-1B / SAM ViT-B 权重），每行含格式、体积、来源徽标与【复制链接】【打开下载】【检测】三按钮；镜像源单独分组并注明适用场景 |
| 🔗 链接可达性检测 | `GET /api/check_link` 后端 HEAD 探测（10s 超时、跟随 ≤2 次重定向、403/405/501 退化 Range GET），前端展示 🟢 可达（含真实体积）/ 🔴 不可达（中文原因 + 镜像引导） |
| 📖 术语系统 | 33 条术语、15 个分类：正文虚线词悬浮即现释义卡，点击直达术语表锚点；术语表支持分类筛选 + 搜索，公式等宽字体展示 |
| 🏆 官方榜单 | 每个条目展示 ISIC 官方榜单 / Papers with Code / 标杆论文（MedSAM、Segment Anything）入口与排名依据指标；**只收录可考来源，不编造数字** |
| 💬 AI 问答助手 | 右下角悬浮面板，DeepSeek 流式（SSE）回答项目结构、数据集下载、指标口径等问题；多轮历史、快捷问题、可中途停止；密钥只存在服务端 |
| 🧪 实验观测（hw1/hw2 全量融合） | `#/hw1`：Grafana 风统计面板 + Val Dice 曲线对比 + 损失曲线（可切换运行）+ 模型对比柱状图 + 消融柱状图 + 运行详情弹窗 + 评测档案 + 预测框导出；`#/hw2`：E1/E2/E3 阶段卡 + 提示类型对比柱状图 + **扰动框鲁棒性曲线** + 评测档案 + 嵌入缓存进度。服务端实时只读扫描两作业 `outputs/` |
| 📊 Grafana 风格图表 | 自研纯 SVG 图表引擎：面积渐变、平滑曲线、**悬停十字线 + 共享 tooltip**、图例可点击隐藏、nice 刻度、阈值参考线、末值标注；另有分组柱状图与带迷你趋势线的统计面板 |
| 🎨 浅色 / 深色双主题 | 导航栏 🌙/☀️ 一键切换，记忆偏好并跟随系统设置；全部颜色走语义化 CSS 变量，图表颜色同步切换 |
| 🧭 代码与产物浏览 | 源码浏览器（点击 `src/*.py` 直接看内容）、图表画廊（缩略图 + 灯箱放大）、评测档案（完整指标 + 原始 JSON）、运行详情（逐 epoch 明细）、权重流式下载、报告查看 |
| ⏱ 实时刷新 | 实时运行面板每 **3s** 轮询 `/api/live`（进度条 + 已运行/预计剩余时间，只更新面板不重绘整页）；**顶栏全局徽标在任何页面显示“有几个任务在跑、是什么、还需多久”**；每个实验视图顶部有「实验地图」（实验矩阵 + 依赖关系 + 当前进度）；另有「▶ 自动刷新（10s）」重新聚合，离开视图自动停表 |
| 🏛 架构说明页 | 内嵌 SVG 分层架构图、模块划分表、前后端链路时序（`#/about`） |

## 页面（hash 路由，单文件 SPA）

| 路由 | 视图 | 内容 |
| --- | --- | --- |
| `#/` | 首页 | hero + 三个关键数字 + 精选条目 + 实验观测摘要 + 榜单速览横条 + 平台能力四宫格 |
| `#/datasets` | 列表页 | 分类 chips + 搜索 + 响应式卡片网格（3/2/1 列） |
| `#/dataset/<id>` | 详情页 | 面包屑、简介（术语悬浮释义）、统计表、下载中心、镜像组、榜单卡、关联术语、侧边信息栏 |
| `#/hw1` | 作业 1 实验 | 运行表、Val Dice 曲线对比、损失曲线（运行切换）、数据统计与分布图、checkpoints、报告查看 |
| `#/hw2` | 作业 2 实验 | E1/E2/E3 阶段卡、嵌入缓存进度、评测表、级联叙事、checkpoints（含 SAM 原版权重下载） |
| `#/glossary` | 术语表 | 分类分组 + 筛选 + 搜索；支持 `#/glossary?term=Dice` 直达锚点 |
| `#/about` | 架构说明 | SVG 架构图（含 progress 心跳节点）、模块表、五条链路、三服务关系卡 |

## 技术栈

| 层 | 技术 | 说明 |
| --- | --- | --- |
| 后端 | Python 3 标准库（`http.server` / `http.client` / `urllib` / `json`） | 零第三方依赖，`ThreadingHTTPServer`，绑定 `127.0.0.1:8800`，全部响应 `Cache-Control: no-store`，静态服务防目录穿越 |
| 前端 | 原生 HTML/CSS/JS 单文件（`static/index.html`） | 无构建链、无 CDN、离线可开；**浅色 / 深色双主题**（语义化 CSS 变量，`localStorage` 记忆 + 跟随系统）；自研纯 SVG 图表引擎（对标 Grafana） |
| AI | DeepSeek `chat/completions`（`stream=true`） | 服务端注入系统提示词并逐行透传 SSE；模型/密钥由环境配置 |
| 数据 | `data/datasets.json`（6 条目）、`data/glossary.json`（33 术语） | 唯一事实来源，前端启动时拉取一次并缓存内存 |

## 目录结构

```
dataset_hub/
├── server.py           # 后端：静态服务 + JSON API + 链接检测 + DeepSeek SSE 代理（端口 8800）
├── static/
│   └── index.html      # 前端：单文件 SPA（七视图 + 术语 tooltip + 聊天面板，内联 CSS/JS）
├── data/
│   ├── datasets.json   # 数据集条目：描述/统计/格式/大小/许可证/下载(含镜像)/榜单/关联术语
│   └── glossary.json   # 术语表：术语/英文/分类/定义/公式
├── .env                # 本地密钥配置（已被 .gitignore 排除，绝不入库）
├── .env.example        # 配置模板（无真实值）
├── .gitignore          # 排除 .env 等
└── README.md           # 本文件
```

## 数据流

```
浏览器（SPA）
   │ ① GET / · /api/overview · /api/datasets · /api/glossary   （启动各一次，缓存内存）
   │ ② GET /api/check_link?url=…                               （点【检测】时）
   │ ③ POST /api/chat {messages:[…]}                           （聊天时，SSE 流式响应）
   │ ④ GET /api/hw1 · /api/hw2 · /exp/…                        （作业 1 / 作业 2 实验视图）
   │ ⑤ GET /api/live                                            （实时运行面板，每 3s 轮询）
   ▼
server.py (127.0.0.1:8800)
   ├─ 读 data/*.json ──────── 数据来源（唯一事实来源）
   ├─ 读 .env / 环境变量 ───── DEEPSEEK_API_KEY / DEEPSEEK_MODEL（仅服务端持有）
   ├─ HEAD 上游（10s 超时）── 链接探测：{ok, status, size_bytes, message}
   ├─ 只读扫描 hw1_unet/ · hw2_sam/ ── outputs/logs（CSV+摘要）、嵌入缓存、figures、reports
   └─ 转发 api.deepseek.com ── SSE 逐行透传（注入系统提示词、截留最近 20 条历史）
```

- **① 数据链路**：hash 路由切换 → 未缓存才请求 → 之后导航零请求。
- **② 检测链路**：超时/DNS/HTTP 状态分类为中文文案，前端 🟢/🔴 呈现。
- **③ 聊天链路**：前端 `fetch + ReadableStream` 逐行解析 `data:`，增量渲染至 `[DONE]`；`{"error": …}` 事件统一渲染为友好错误。
- **④ 实验聚合链路**：`/api/hw1`、`/api/hw2` 每次请求实时扫描两作业的 `outputs/`（logs/figures/checkpoints/reports）与 hw2 的 `data/embeddings/` 嵌入缓存（与暗色监控台同一套聚合口径），训练进行中时视图随之刷新；图片经 `/exp/<proj>/figures/<f>`，权重经 `/exp/<proj>/download` 流式下载（1MB 分块；白名单为各作业 `outputs/checkpoints/` 与 hw2 的 SAM 原版权重，防目录穿越）。
- **⑤ 实时监控链路**：训练/评测进程每 epoch（评测每 25 张）向 `outputs/progress/*.json` 写心跳 → 前端每 **3 秒**轮询 `/api/live`（10 分钟心跳窗口；`phase=done` 保留 10 分钟展示「刚完成」）→ 只更新实时运行面板，不打断图表悬停与滚动；实验视图的「自动刷新（10s）」则重新聚合 `/api/hw1|hw2`，离开视图会自动停表。

## 启动

```bash
cd dataset_hub
/usr/bin/python3 server.py        # 无需任何第三方包，Python ≥ 3.9
# → http://127.0.0.1:8800/
```

停止：终端 `Ctrl+C`。端口固定 8800（绑 `127.0.0.1`；原 8801/8802 监控台已废弃，功能已并入本门户）。

## 配置（AI 助手）

两种方式，**环境变量优先**（`.env` 用 `os.environ.setdefault` 解析，不覆盖已有环境变量）：

```bash
# 方式一：.env 文件（推荐，本地持久）
cp .env.example .env
# 编辑 .env：
# DEEPSEEK_API_KEY=sk-你的密钥
# DEEPSEEK_MODEL=deepseek-chat

# 方式二：shell 环境变量（临时/覆盖）
export DEEPSEEK_API_KEY=sk-你的密钥
/usr/bin/python3 server.py
```

- 未配置密钥时平台其余功能完全可用；导航栏状态点显示 🟡「AI 未配置」，聊天面板会给出引导文案。
- `DEEPSEEK_MODEL` 缺省为 `deepseek-chat`。
- ⚠️ **安全红线**：密钥只允许出现在 `.env`（已 gitignore）或你自己的 shell 环境变量中；前端页面、git 提交里一律不得出现明文密钥。

## API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 返回 `static/index.html` |
| GET | `/api/overview` | `{datasets, terms, leaderboard_sources, chat_enabled, server_time}` |
| GET | `/api/datasets` | `data/datasets.json` 全文 |
| GET | `/api/glossary` | `data/glossary.json` 全文 |
| GET | `/api/check_link?url=` | HEAD 探测（10s 超时）→ `{ok, status, size_bytes, message}` |
| GET | `/api/hw1` | 作业 1 实验聚合（runs/evals/checkpoints/figures/reports/data_stats，只读） |
| GET | `/api/hw2` | 作业 2 实验聚合（stages/embeddings/runs/evals/checkpoints/figures/reports，只读） |
| GET | `/api/live` | 实时进度：扫描两作业 `outputs/progress/*.json` 心跳，只返回 10 分钟内的条目（`phase=done` 保留 10 分钟，且同一阶段/提示已完成时自动隐藏残留的 eval 心跳）；每条含 `elapsed_sec` / `eta_sec`（训练按最近 epoch 实测耗时、评测按相邻心跳速率估算） |
| GET | `/api/hw_report?proj=&path=` | 报告文本内容（限各作业 reports/ 目录） |
| GET | `/api/hw_code?proj=&path=` | 源码内容（限各作业 src/，仅 .py） |
| GET | `/api/hw_eval?proj=&file=` | 评测 JSON 原文（限各作业 outputs/logs/） |
| GET | `/exp/<proj>/figures/<f>` | 实验图片（限各作业 outputs/figures/） |
| GET | `/exp/<proj>/download?path=` | 权重流式下载（限 outputs/checkpoints/ 与 hw2 的 SAM 原版权重；`proj` 写在路径段或 `?proj=` 均可） |
| POST | `/api/chat` | 请求体 `{"messages":[{role,content},…]}`，响应 `text/event-stream` 透传 |

错误统一返回 `{"error": "中文说明"}` 与合适状态码。

## 维护备注

- 新增数据集/术语：直接编辑 `data/*.json`（服务端每次请求实时读盘，无需重启）；`glossary.json` 中新增分类时记得同步 `categories` 数组（前端另有未声明分类的兜底渲染，不会丢词条）。
- 榜单纪律：只添加官方或论文可考的来源，禁止编造排名与指标数字。
- 本平台对 `hw1_unet/`、`hw2_sam/` **只读引用**（叙述与事实数据），绝不修改实验代码。
- 安全边界：`/api/check_link` 可探测任意 http/https 目标（含本机端口），属本地工具的有意设计；服务仅绑定 `127.0.0.1`，**切勿**改为 `0.0.0.0` 对外暴露。
