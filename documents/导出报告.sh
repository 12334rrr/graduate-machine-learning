#!/bin/bash
# 把两份 Markdown 报告导出为 DOCX / PDF（图片内嵌，可直接提交）
# 依赖：pandoc；Google Chrome（无头打印 PDF）
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS="$(mktemp /tmp/report_css_XXXX.css)"
cat > "$CSS" <<'EOF'
@page { size: A4; margin: 17mm 15mm; }
html { -webkit-print-color-adjust: exact; }
body { font-family: "PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
  font-size: 10.8pt; line-height: 1.65; max-width: 100%; margin: 0; }
h1 { font-size: 19pt; margin: 0 0 10px; }
h2 { font-size: 14.5pt; margin-top: 22px; page-break-after: avoid; }
h3 { font-size: 12.5pt; margin-top: 16px; page-break-after: avoid; }
img { max-width: 100%; max-height: 215mm; object-fit: contain; display: block;
  margin: 8px auto; page-break-inside: avoid; }
table { border-collapse: collapse; width: 100%; font-size: 9.2pt;
  page-break-inside: avoid; margin: 8px 0; }
th, td { border: 1px solid #999; padding: 3px 6px; text-align: left; vertical-align: top; }
code { font-family: Menlo, monospace; font-size: 9pt; }
pre { font-family: Menlo, monospace; font-size: 8.8pt; white-space: pre-wrap;
  word-break: break-word; background: #f6f8fa; border: 1px solid #ddd;
  padding: 8px; page-break-inside: avoid; }
blockquote { border-left: 3px solid #bbb; margin-left: 0; padding-left: 12px; color: #444; }
hr { border: none; border-top: 1px solid #ccc; margin: 18px 0; }
EOF

for proj in hw1_unet hw2_sam; do
  echo "[导出] $proj"
  cd "$ROOT/$proj"
  pandoc reports/实验报告.md -o reports/实验报告.docx --resource-path=reports:.
  pandoc reports/实验报告.md -o "/tmp/${proj}_report.html" \
    --standalone --embed-resources --resource-path=reports:. --css="$CSS"
  rm -rf "/tmp/chrome_report_$proj"
  "$CH" --headless=new --disable-gpu --no-first-run \
    --user-data-dir="/tmp/chrome_report_$proj" --no-pdf-header-footer \
    --print-to-pdf="reports/实验报告.pdf" "file:///tmp/${proj}_report.html" >/dev/null 2>&1
  echo "  → $proj/reports/实验报告.{docx,pdf}"
done
rm -f "$CSS"
echo "完成。"
