# 作业 2｜SAM 提示分割（零样本 → 冻结编码器微调 → LoRA）

**模型**：SAM ViT-B（91M，`sam_vit_b_01ec64.pth`，375 MB）
**数据**：与作业 1 共用 ISIC 2018，**必须共用同一测试集、同一指标口径**
**（直接复用作业 1 的 `metrics.py`，杜绝两版指标不一致）**
**完整执行规格**：见 `../documents/作业执行总提示词.md` §5 与 §6–§8  
**实验进度与结果**：打开一站式门户 <http://127.0.0.1:8800/#/hw2>（阶段卡、评测档案、嵌入缓存、源码浏览）  
> `dashboard/` 目录已废弃（端口 8802），功能全部并入门户，见 [dashboard/DEPRECATED.md](dashboard/DEPRECATED.md)

## 环境（按用途命名）

| 环境名 | 用途 | 目录 |
| --- | --- | --- |
| `hw1-unet-medseg` | 作业 1：手写医学图像分割（U-Net 系列） | `/opt/anaconda3/envs/hw1-unet-medseg` |
| `hw2-sam-transformer` | 作业 2：SAM(ViT) 提示分割与 LoRA 微调 | `/opt/anaconda3/envs/hw2-sam-transformer` |

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate hw2-sam-transformer
python -c "import torch, segment_anything;print(torch.__version__, torch.backends.mps.is_available())"
```

Python 3.11，conda-forge 源，核心依赖：torch / torchvision / numpy / pandas / matplotlib /
opencv-python-headless / scipy / tqdm / segment-anything / einops

## 目录

| 路径 | 用途 |
| --- | --- |
| `data/raw/` | `sam_vit_b_01ec64.pth`（375 MB） |
| `data/embeddings/{split}/` | **预计算的 SAM image embedding**（约 7 GB，E1/E2 共用） |
| `src/` | E1 / E2 / E3 + 预计算 + 评测 + 出图 |
| `outputs/checkpoints/` | 按 epoch 存盘（E3 必须分段续跑） |
| `outputs/logs/` | `eval_{stage}_{prompt}.json` + 每实验一个 CSV |
| `outputs/figures/` | SAM 三件套结构图、mask decoder 数据流图、扰动框曲线、级联系统图 |
| `reports/` | 最终实验报告 |

## 三阶段实验

| 实验 | 设置 | 预期耗时 |
| --- | --- | --- |
| E1 零样本 | 不训练，GT 框提示 | 评测 1,000 张，约 10 分钟 |
| E2 冻结编码器 | 只训 prompt encoder + mask decoder | 用缓存嵌入，**每 epoch 数十秒** |
| E3 LoRA | 在 ViT 的 q/v 投影注入 LoRA（r=8） | bs=1 + 累积 8，约 20~30 min/epoch，**分段跑** |

## 关键工程点

**冻结 image encoder 时图像嵌入与提示无关 → 预计算并缓存**（2,594 张 × 256×64×64 fp16 ≈ 5 GB）。
这一步让 E2 从"每 epoch 十几分钟"降到"几十秒"，是整套方案能在本机跑完的前提。
E3 动了主干，缓存失效，必须在线前向。

## 复现步骤

```bash
conda activate hw2-sam-transformer
python src/run_all.py                       # 全流程（自动分段续跑）
python src/run_all.py --from e2             # 从 E2 开始（嵌入已算好）
python src/run_all.py --only eval           # 只评测
```

单独跑：

```bash
python src/download_sam.py                  # 下载 SAM ViT-B 权重（已存在则跳过）
python src/precompute_embeddings.py --split train
python src/evaluate_sam.py --stage e1 --prompt gt     --split test
python src/evaluate_sam.py --stage e1 --prompt pert20 --split test
python src/e2_finetune.py --tag e2_box --epochs 20
python src/e3_lora.py     --tag e3_lora --epochs 10
python src/evaluate_sam.py --stage e2 --tag e2_box --prompt auto --unet-tag unet_s42
python src/plot_figures.py --arch --robust --cascade
python src/plot_cases.py                   # 定性案例图（GT 框，E1/E2/E3 对比）
```

## 提示设计

GT 框（主结果）· 扰动框 ±10%/±20%（鲁棒性曲线）· 中心点/3 点/5 点 ·
**自动框级联**（用作业 1 的 U-Net 预测框作提示，把两个作业缝成一条系统）

## 权重下载

优先 `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`；
不可达时改从 ModelScope 或 `hf-mirror`（`facebook/sam-vit-base`）取，并设 `HF_ENDPOINT=https://hf-mirror.com`。

## 硬性约束

- 报告必须说清 **Transformer 落点**：ViT 图像编码器 + mask decoder 的双向交叉注意力
- GT 框属于交互式分割标准设定，报告中**主动写清**并用扰动框实验佐证，不回避
- E3 必须分段 checkpoint，禁止一次性长跑（Air 无风扇 + MPS 偶发崩溃）
