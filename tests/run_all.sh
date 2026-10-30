#!/bin/bash
# 提交/推送前的统一测试入口：任何代码修改后必须先跑本脚本，全绿才允许 commit/push。
# 用法：bash tests/run_all.sh
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source /opt/anaconda3/etc/profile.d/conda.sh

echo "============================================================"
echo "[0/5] 语法检查"
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
echo "[1/5] 作业 1 单元测试"
echo "============================================================"
cd "$ROOT"
conda activate hw1-unet-medseg
python tests/test_hw1.py

echo "============================================================"
echo "[2/5] 作业 2 单元测试"
echo "============================================================"
conda activate hw2-sam-transformer
python tests/test_hw2.py

echo "============================================================"
echo "[3/5] 门户单元 + HTTP 冒烟"
echo "============================================================"
/usr/bin/python3 tests/test_hub.py

echo "============================================================"
echo "[4/5] 前端渲染冒烟（门户在线时）"
echo "============================================================"
if curl -s -o /dev/null --max-time 2 http://127.0.0.1:8800/; then
  CH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
  if [ -x "$CH" ]; then
    rm -rf /tmp/hub-render-profile
    # --no-sandbox：本机沙箱/CI 容器内 Chrome 必需（正常桌面环境加上也无副作用）
    # --no-proxy-server：避免沙箱注入的 HTTP_PROXY 把 127.0.0.1 请求代理出去（连接被拒）
    "$CH" --headless=new --no-sandbox --no-proxy-server --disable-gpu --no-first-run \
      --user-data-dir=/tmp/hub-render-profile --virtual-time-budget=6000 \
      --enable-logging=stderr --v=0 --dump-dom "http://127.0.0.1:8800/#/hw1" \
      > /tmp/hub-render.html 2>/tmp/hub-render.err || true
    grep -q "实验地图" /tmp/hub-render.html || { echo "  [失败] 前端未渲染出实验地图"; exit 1; }
    if grep -qi "CONSOLE.*\\(error\\|uncaught\\)" /tmp/hub-render.err; then
      echo "  [失败] 前端控制台报错"; grep -i "CONSOLE" /tmp/hub-render.err | head -5; exit 1
    fi
    echo "  前端渲染与控制台检查通过"
  else
    echo "  跳过（未找到 Chrome）"
  fi
else
  echo "  跳过（门户未启动）"
fi

echo "============================================================"
echo "[5/5] 报告/提交包结构检查"
echo "============================================================"
for f in hw1_unet/reports/实验报告.md hw2_sam/reports/实验报告.md \
         hw1_unet/reports/tables.md; do
  if [ ! -f "$ROOT/$f" ]; then
    echo "  [缺失] $f"
    exit 1
  fi
done
# 最终交付物（2026-10-30 清理后，正式报告以最终提交包为准；hw*/reports 的 md 仅为指引存根）
FINAL_DIR="$ROOT/202620000613_王春博_机器学习课程作业_最终提交包 2"
for f in "202620000613_王春博_机器学习课程作业_最终报告.pdf" \
         "202620000613_王春博_机器学习课程作业_最终报告.docx" \
         "RESULTS_RELEASE.json"; do
  if [ ! -s "$FINAL_DIR/$f" ] && [ ! -s "$ROOT/$f" ]; then
    echo "  [缺失] 最终交付物：$f"
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
