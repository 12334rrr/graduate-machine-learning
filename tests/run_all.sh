#!/bin/bash
# 提交/推送前的统一测试入口：任何代码修改后必须先跑本脚本，全绿才允许 commit/push。
# 用法：bash tests/run_all.sh
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source /opt/anaconda3/etc/profile.d/conda.sh

echo "============================================================"
echo "[0/4] 语法检查"
echo "============================================================"
/usr/bin/python3 -m py_compile "$ROOT/dataset_hub/server.py"
conda activate hw1-unet-medseg
python -m py_compile "$ROOT/hw1_unet/src/dataset.py" "$ROOT/hw1_unet/src/train.py" \
  "$ROOT/hw1_unet/src/evaluate.py" "$ROOT/hw1_unet/src/collect_results.py" \
  "$ROOT/hw1_unet/src/run_cv.py"
conda activate hw2-sam-transformer
python -m py_compile "$ROOT/hw2_sam/src/evaluate_sam.py" "$ROOT/hw2_sam/src/plot_cases.py" \
  "$ROOT/hw2_sam/src/download_sam.py"
echo "  语法检查通过"

echo "============================================================"
echo "[1/4] 作业 1 单元测试"
echo "============================================================"
cd "$ROOT"
conda activate hw1-unet-medseg
python tests/test_hw1.py

echo "============================================================"
echo "[2/4] 作业 2 单元测试"
echo "============================================================"
conda activate hw2-sam-transformer
python tests/test_hw2.py

echo "============================================================"
echo "[3/4] 门户单元 + HTTP 冒烟"
echo "============================================================"
/usr/bin/python3 tests/test_hub.py

echo "============================================================"
echo "[4/4] 报告/提交包结构检查"
echo "============================================================"
for f in hw1_unet/reports/实验报告.md hw2_sam/reports/实验报告.md \
         hw1_unet/reports/tables.md submission/提交说明.md; do
  if [ ! -f "$ROOT/$f" ]; then
    echo "  [缺失] $f"
    exit 1
  fi
done
for f in hw1_unet/reports/实验报告.pdf hw2_sam/reports/实验报告.pdf; do
  if [ -f "$ROOT/$f" ] && [ "$(wc -c < "$ROOT/$f")" -lt 100000 ]; then
    echo "  [异常] $f 体积过小"
    exit 1
  fi
done
echo "  结构检查通过"
echo
echo "ALL TESTS PASSED"
