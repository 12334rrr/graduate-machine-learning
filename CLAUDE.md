# CLAUDE.md · 项目要求与工作约定

> 给未来的 AI 会话（以及人类协作者）看的项目说明。**动手前先读本文件与 `documents/作业执行总提示词.md`。**

---

## 1. 这是什么

一门机器学习课程的两份平时作业 + 一个把它们缝在一起的一站式科研门户。
核心叙事是**级联系统**：作业 1 的小模型（U-Net）全自动出粗框 → 作为提示喂给作业 2 的大模型（SAM）精修——
「小模型提框、大模型精修，分工而非替代」。

| 子项目 | 内容 | 入口 |
| --- | --- | --- |
| `hw1_unet/` | 手写 U-Net / ResUNet / Attention U-Net，ISIC 2018 Task 1 皮肤病变分割 | 门户 `#/hw1` |
| `hw2_sam/` | SAM ViT-B 三阶段：E1 零样本 / E2 冻结编码器微调 / E3 LoRA(r=8) | 门户 `#/hw2` |
| `dataset_hub/` | 一站式门户：数据集、术语、榜单、AI 问答、两作业实验观测，浅色/深色双主题 | `http://127.0.0.1:8800` |

**截止时间**：2026-11-30。两份报告统一六段式（摘要 / 引言 / 模型原理+结构图 / 实验设计 / 结果分析三小节 / 总结含高亮观点）。

---

## 2. 环境（已就绪，不要重装）

| 项 | 事实 |
| --- | --- |
| 机器 | MacBook Air，Apple M5，24GB 统一内存，**无风扇**，macOS 26.6 |
| 算力 | 仅 MPS（无 CUDA），fp16 实测约 13.6 TFLOPS |
| conda | 具名环境：`hw1-unet-medseg`、`hw2-sam-transformer`（Python 3.11.17 / torch 2.14.1 / MPS True） |
| 门户 | `dataset_hub` 用系统 `/usr/bin/python3`（3.9），**零第三方依赖** |
| 网络 | huggingface.co 不可达；`hf-mirror.com`、ModelScope、阿里云 pypi 可用；`api.deepseek.com` 可用 |

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate hw1-unet-medseg        # 作业 1
conda activate hw2-sam-transformer    # 作业 2
/usr/bin/python3 dataset_hub/server.py                            # 门户（无需 conda）
```

---

## 3. 硬性约束（违反即返工）

| 约束 | 说明 |
| --- | --- |
| **必须手写模型** | 禁用 MONAI / segmentation_models_pytorch 的现成 U-Net；要能逐行讲清实现 |
| **单次连续训练 ≤ 25 分钟** | Air 无风扇会降频；`train.py --max-minutes` 强制，`run_all.py` 自动分段续跑 |
| **不编造指标** | 指标不好就如实写并分析原因；差异 <0.5 点必须用 3 种子均值±标准差或说"不显著" |
| **不删 checkpoint** | 续跑靠 `{tag}_last.pth`；删了就前功尽弃 |
| **数据/权重不入库** | 见 `.gitignore`：`*/data/`、`*/outputs/checkpoints/`、`.conda/`、`.env` 全部排除 |
| **密钥只存 `.env`** | `dataset_hub/.env` 已被 gitignore；前端、文档、提交里不得出现明文密钥 |
| **门户只读实验产物** | `dataset_hub` 每次请求扫描两作业 `outputs/`，**不改实验代码** |
| **推送前必须单元测试** | 任何代码修改后，先跑 `bash tests/run_all.sh`（语法 + hw1/hw2/hub 单元测试 + HTTP/前端渲染冒烟 + 交付物结构），**全绿才允许 `git commit` / `git push`**；测试失败不得推送，禁止 `--no-verify` 绕过（已装 `.githooks/pre-push` 自动门禁） |

---

## 4. 常用命令

### 作业 1
```bash
cd hw1_unet && conda activate hw1-unet-medseg
python src/run_all.py --only main        # 主对比 9 组（3 模型 × 3 种子，12 epoch）
python src/run_all.py --only ablation    # 消融 6 组（损失/分辨率/注意力位置）
python src/run_all.py                    # 全部
python src/train.py --model unet --seed 42 --tag unet_s42 --resume   # 单实验续跑
python src/evaluate.py --tag unet_s42 --split test --size 256        # 评测 + 四联图 + 预测框
python src/collect_results.py            # → reports/tables.md + outputs/ablation.json
python src/plot_figures.py --arch        # 三张结构图
```

### 作业 2
```bash
cd hw2_sam && conda activate hw2-sam-transformer
python src/run_all.py                    # 全流程（E1 评测 → E2 → E3 → 级联 → 出图）
python src/run_all.py --from e2          # 从 E2 跑到最后
python src/run_all.py --from e3          # E3 → 级联 → 出图
python src/evaluate_sam.py --stage e1 --prompt gt --split test
python src/plot_figures.py --arch --robust --cascade
```

### 门户
```bash
cd dataset_hub && /usr/bin/python3 server.py      # 127.0.0.1:8800
curl -s localhost:8800/api/hw1 | python3 -m json.tool | head
```

### 测试（git commit / push 前必须全绿）
```bash
bash tests/run_all.sh        # 语法检查 + hw1/hw2/hub 单元测试 + HTTP 冒烟 + 交付物结构
```

---

## 5. 当前状态（2026-10-19）

| 项 | 状态 |
| --- | --- |
| hw1 数据 + 三模型实现 | ✅ 完成（384 母版 npy，train 2594 / val 100 / test 1000） |
| hw1 主对比 9 组 | ✅ 完成（U-Net 0.594±0.022 / ResUNet 0.608±0.019 / Attn U-Net 0.608±0.015，**低于 0.75 目标**，欠训练） |
| hw1 消融 6 组 | ✅ 完成（单种子 42；损失/分辨率/注意力位置各组 3 设置，结果见 `outputs/ablation.json`） |
| hw1 报告 | ✅ 已写（六段式，含消融分析；数字与 `tables.md` 一致） |
| hw2 E1 零样本 | ✅ test 1000 张 **Dice 0.869** |
| hw2 E2 冻结微调 | ✅ test 1000 张 **Dice 0.900**（超作业 1 最佳 0.608，W5 验收达成） |
| hw2 E3 LoRA | ✅ 8 组提示 test 评测完成（GT 框 **Dice 0.898** / HD95 17.5） |
| hw2 级联 | ✅ E1/E2/E3 的 auto/auto10 全部完成（auto：E1 0.384 / E2 0.659 / E3 **0.703**） |
| hw1 五折 CV | ✅ 完成（2 模型 × 5 折；Attn U-Net 0.563±0.068 / ResUNet 0.570±0.035，差异不显著） |
| hw2 报告 | ✅ 已写（六段式，E3 评测数字已补全） |
| 门户 | ✅ 七视图 + 双主题 + Grafana 风格图表 + 代码浏览/评测档案/画廊/自动刷新 |

---

## 6. 踩过的坑（改代码前务必知道）

| 坑 | 症状 | 正确做法 |
| --- | --- | --- |
| **CSV 日志续跑截断** | 续跑后图表只剩最后几个 epoch（曾 9/12 个运行被截断） | `CSVLogger(..., append=args.resume)`；已修，历史数据已从 summary/checkpoint 恢复 |
| **SAM mask_decoder 是单图设计** | batch>1 报 `size of tensor a (64) must match b (8)` | `encode_and_decode` 内逐图解码；`image_pe` 恒为 `(1,256,64,64)`，**不能按 batch 切片**（i≥1 会切出空张量） |
| **LoRA 参数设备** | `weight is on cpu but expected on mps` | LoRA 的 `nn.Parameter` 必须建在 `base.weight.device` 上 |
| **DataLoader spawn + sys.path** | worker 里 `from common import EVAL_SIZE` 解析到 hw1 的 common | hw2 的 `common.py` 用 `sys.path.append` 而非 `insert(0)`；训练脚本顶部再显式插入本目录 |
| **NaN 污染指标** | 整份评测 `specificity = NaN`（test #641 的 GT 全前景 → 0/0） | `MetricAccumulator` 剔除未定义样本并计数，`nanmean` 兜底 |
| **NaN 进入 JSON** | 前端 `res.json()` 解析失败 → 整页"聚合失败" | 服务端 `_send_json` 统一 `_sanitize_json`（NaN/Inf → null） |
| **JS 函数重名** | `fmtNum` 被图表版覆盖，数据集详情页整页崩 | 统计格式化用 `fmtStatNum`，图表刻度用 `fmtNum` |
| **热节流** | 长跑后评测从 0.9 s/图 掉到 15 s/图 | 让机器冷却 20-30 分钟；E3 评测放在训练之后单独跑 |
| **pkill 自匹配** | 命令文本含脚本名，把自己 shell 杀了 | 用 `pgrep -f "run_al[l]"` 括号技巧或按端口/PID 精确清理 |

---

## 7. 门户架构速览

```
server.py (127.0.0.1:8800, 标准库)
  ├─ GET /                     单文件 SPA（static/index.html，无构建/CDN）
  ├─ /api/overview|datasets|glossary
  ├─ /api/hw1|hw2              实时扫描两作业 outputs/（CSV+summary、评测、嵌入缓存）
  ├─ /api/hw_code|hw_eval|hw_report   源码 / 评测原文 / 报告（均限目录、防穿越）
  ├─ /api/check_link           HEAD 探测（10s 超时，403/405/501 退化 Range GET）
  ├─ /exp/<proj>/figures|download     图片 / 权重流式下载
  └─ POST /api/chat            DeepSeek SSE 透传（key 只在服务端 .env）
```

前端：hash 路由七视图；Grafana 风格图表引擎（`gchart/gbar/gstat`，悬停十字线+共享 tooltip、图例可点击隐藏、阈值线、面积渐变）；浅色/深色双主题（`data-theme` + CSS 变量 + localStorage `dh-theme`）。

---

## 8. 工作方式

1. **先读** `documents/作业执行总提示词.md`（规格与约束）与 `documents/作业收尾执行提示词.md`（从当前状态到交付的路线）。
2. **分段交付**：每个阶段跑完对照验收标准核对，不超过 5 行汇报（产出文件、关键数字、下一步）。
3. **长任务后台跑**：`nohup ... &`，不要占着终端；被中断就重跑同一条命令（自动续跑），**绝不删 checkpoint**。
4. **改实验代码只允许修 bug**，改动必须记录在提交信息里；改完先 `python -m py_compile` 再跑。
5. **测试门禁与推送**：任何代码修改后先跑 `bash tests/run_all.sh`，全部通过后才 `git add/commit/push`；
   仓库已配置 `.githooks/pre-push`，`git push` 时会再自动跑一遍，失败即拒绝推送；
   提交信息里写明改动与测试结果；推送前 `git status` 确认没有数据、权重、`.env` 或明文密钥。
6. **报告纪律**：数字全部来自 `outputs/logs/*.json`，不得手填；口径（12 epoch、E3 子集、GT 框提示）必须在报告中披露。
