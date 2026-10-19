#!/bin/bash
# 生成课程提交包：两份报告的 PDF/DOCX + 两个作业的源码与结果
# 用法：bash documents/打包提交.sh
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUB="$ROOT/submission"

echo "[1/3] 导出报告 PDF/DOCX"
bash "$ROOT/documents/导出报告.sh"

echo "[2/3] 建立 submission/"
rm -rf "$SUB"
mkdir -p "$SUB"
cp "$ROOT/hw1_unet/reports/实验报告.pdf" "$SUB/作业1_实验报告.pdf"
cp "$ROOT/hw1_unet/reports/实验报告.docx" "$SUB/作业1_实验报告.docx"
cp "$ROOT/hw2_sam/reports/实验报告.pdf" "$SUB/作业2_实验报告.pdf"
cp "$ROOT/hw2_sam/reports/实验报告.docx" "$SUB/作业2_实验报告.docx"
cp "$ROOT/documents/提交说明.md" "$SUB/提交说明.md"

echo "[3/3] 打包源码与结果（排除数据/权重/缓存）"
cd "$ROOT/hw1_unet"
zip -qr "$SUB/作业1_源码与结果.zip" \
  src reports README.md requirements.txt \
  outputs/logs outputs/figures outputs/ablation.json outputs/results_summary.json \
  -x "*/__pycache__/*" "*.pyc"
[ -f outputs/cv_results.json ] && zip -q "$SUB/作业1_源码与结果.zip" outputs/cv_results.json

cd "$ROOT/hw2_sam"
zip -qr "$SUB/作业2_源码与结果.zip" \
  src reports README.md requirements.txt \
  outputs/logs outputs/figures \
  -x "*/__pycache__/*" "*.pyc"

echo
echo "完成：$SUB"
ls -lh "$SUB"
