#!/usr/bin/env bash
set -euo pipefail

ROOT="https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/"
OUT="/home/sun/wd_14/OISST/avhrr"   # 根据需要改
mkdir -p "$OUT"
cd "$OUT"

echo "[1/3] 列出子目录..."
# 提取所有以 / 结尾的链接（目录），再过滤出 6 位数字的“年月目录”
curl -s "$ROOT" \
| grep -oE 'href="[^"]+/"' \
| sed -E 's/^href="([^"]+)\/"$/\1\//' \
| grep -E '^[0-9]{6}/$' \
| sort -u > subdirs.txt

echo "[2/3] 将下载 $(wc -l < subdirs.txt) 个目录中的 .nc/.nc.gz 文件"
i=0; total=$(wc -l < subdirs.txt)
while read -r d; do
  ((i++))
  echo "[${i}/${total}] $d"
  wget -r -np -nH -c -N \
    --cut-dirs=6 \
    -e robots=off \
    --level=1 \
    -A .nc,.nc.gz \
    --no-verbose \
    "${ROOT}${d}"
done < subdirs.txt

echo "[3/3] 完成：$OUT"
