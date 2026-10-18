# 作业 1｜医学图像分割：U-Net / ResUNet / Attention U-Net

**数据集**：ISIC 2018 Task 1（皮肤病变），训练 2,594 + 验证 100 + 测试 1,000  
**完整执行规格**：见 `../documents/作业执行总提示词.md` §4 与 §6–§8  
**实验进度与结果**：打开一站式门户 <http://127.0.0.1:8800/#/hw1>（曲线、评测、权重下载、源码浏览）  
> `dashboard/` 目录已废弃（端口 8801），功能全部并入门户，见 [dashboard/DEPRECATED.md](dashboard/DEPRECATED.md)

## 环境（按用途命名）

| 环境名                   | 用途                          | 目录                                        |
| --------------------- | --------------------------- | ----------------------------------------- |
| `hw1-unet-medseg`     | 作业 1：手写医学图像分割（U-Net 系列）     | `/opt/anaconda3/envs/hw1-unet-medseg`     |
| `hw2-sam-transformer` | 作业 2：SAM(ViT) 提示分割与 LoRA 微调 | `/opt/anaconda3/envs/hw2-sam-transformer` |

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate hw1-unet-medseg
python -c "import torch;print(torch.__version__, torch.backends.mps.is_available())"   # 需打印 True
```

Python 3.11，conda-forge 源，核心依赖：torch / torchvision / numpy / pandas / matplotlib / scikit-learn / opencv-python-headless / seaborn / tqdm / scipy

> 两个环境原来是无名的前缀环境（`hw1_unet/.conda`、`hw2_sam/.conda`），  
> 已按用途重建为具名环境并删除旧的空环境，方便 `conda activate <名字>` 直接激活。

## 目录

| 路径                     | 用途                                                   |
| ---------------------- | ---------------------------------------------------- |
| `data/raw/`            | 官方压缩包 + 解压后的原始数据（**禁止放 `~/.cache`**）                 |
| `data/processed/`      | resize 成 384×384 母版后的 npy：images / masks / ids       |
| `src/`                 | 三个模型的**手写实现** + 训练/评测/出图脚本                           |
| `outputs/checkpoints/` | 按 epoch 存盘（`{tag}_last.pth` / `{tag}_best.pth`）      |
| `outputs/logs/`        | 每实验一个 CSV：epoch / lr / loss / dice / iou / hd95 / 耗时 |
| `outputs/figures/`     | 结构图、训练曲线、消融图、测试样本四联图                                 |
| `outputs/predictions/` | 测试集预测框（给作业 2 的级联实验用）                                 |
| `reports/`             | `data_stats.md`（数据统计）+ `tables.md`（结果表）+ 实验报告        |

## 复现步骤

```bash
conda activate hw1-unet-medseg

python src/download_isic.py                  # 1) 分块并行下载官方 6 个压缩包
python src/prepare_data.py                   # 2) 解压、配对、resize、出 data_stats.md
python src/plot_figures.py --arch            # 3) 三张模型结构图
python src/run_all.py                        # 4) 训练 + 评测 + 汇总（自动分段续跑）
```

单独跑一个实验：

```bash
python src/train.py --model unet --seed 42 --tag unet_s42          # 训练（25 分钟一段）
python src/train.py --model unet --seed 42 --tag unet_s42 --resume # 续跑
python src/evaluate.py --tag unet_s42 --split test                 # 测试集评测 + 四联图
python src/collect_results.py                                      # 汇总成报告表格
```

## 关键配置（默认值，改动需记录）

256×256 / batch 8 / AdamW 1e-3 cosine / 0.5×BCE + 0.5×Dice / **12 epoch / patience 6** / 种子 42  
> 关于 epoch：规格原定 30 epoch，但全量实测约 5.5 分钟/epoch（M5 MPS fp16），
> 30 epoch × 15 组实验 ≈ 45 小时不可行；`run_all.py` 统一压到 12 epoch + patience 6，
> 并在报告中如实说明这一取舍。`train.py` 的默认值仍是 30，由 run_all 显式覆盖。

**必须开 `torch.autocast("mps", dtype=torch.float16)`**，设备自动探测 `cuda → mps → cpu`
**续跑**：`--resume` 会从 `{tag}_last.pth` 继续；CSV 日志以追加模式写入（2026-10-18 修复，此前会清空历史 epoch）。

## 硬性约束

- **禁止用 MONAI / segmentation_models_pytorch 的现成 U-Net**，必须手写
- 单次连续训练不超过 25 分钟（MacBook Air 无风扇，会降频）—— 由 `--max-minutes` 强制，  
  `run_all.py` 自动分段续跑
- 关键对比跑 3 个种子，报均值 ± 标准差
- HD95 计算时空掩膜单独统计，禁止除零污染结果
