#!/bin/bash
# 数据下载脚本（全部数据集一键获取 + 哈希校验）
# 用法: bash scripts/download_all_data.sh
set -e
cd "$(dirname "$0")/.."
mkdir -p data/raw

echo "=== 1. SCAN（已在仓库 data/../SCAN 与 reference/SCAN-official，无需下载） ==="
ls SCAN/simple_split/tasks_train_simple.txt >/dev/null && echo "  SCAN OK (in-repo)"

echo "=== 2. COGS（已在仓库 data/cogs/，或从官方重新下载） ==="
if [ ! -f data/cogs/train.tsv ]; then
  mkdir -p data/cogs
  for f in train dev test gen gen_cases_to_paper_sections; do
    curl -sL "https://raw.githubusercontent.com/najoungkim/COGS/main/data/$f.tsv" -o "data/cogs/$f.tsv" ||
    curl -sL "https://raw.githubusercontent.com/najoungkim/COGS/main/data/$f.json" -o "data/cogs/$f.json"
  done
fi
python3 -c "
import hashlib, json
man = json.load(open('configs/cogs_manifest.json'))
for n in ('train','dev','test','gen'):
    h = hashlib.sha256(open(f'data/cogs/{n}.tsv','rb').read()).hexdigest()
    assert h == man[n]['sha256'], f'{n} SHA mismatch'
    print(f'  COGS {n}.tsv SHA OK')"

echo "=== 3. CFQ splits（已在仓库 data/raw/CFQ/cfq/splits/） ==="
ls data/raw/CFQ/cfq/splits/mcd1.json >/dev/null && echo "  CFQ splits OK (in-repo)"

echo "=== 4. CFQ dataset.json（4.5GB，超 GitHub 上限，从官方下载） ==="
if [ ! -f data/raw/CFQ/cfq/dataset.json ]; then
  echo "  下载 cfq1.1.tar.gz（267MB 压缩，解压后 dataset.json 4.5GB）..."
  curl -sL "https://storage.googleapis.com/cfq_dataset/cfq1.1.tar.gz" -o /tmp/cfq1.1.tar.gz
  tar xzf /tmp/cfq1.1.tar.gz -C data/raw/CFQ
fi
echo "  CFQ dataset.json: $(du -h data/raw/CFQ/cfq/dataset.json | cut -f1)（如需校验 SHA 见 configs/cfq_manifest.json）"

echo "=== 5. 参考仓库（已在仓库 reference/，或按需重新克隆） ==="
echo "  reference/: SCAN-official, COGS, TinyRecursiveModels, coconut, TaH, pycma"
echo "全部数据就绪。"
