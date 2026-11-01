#!/usr/bin/env bash
# One command for the whole dataset pipeline. Run from this folder:  bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1/5 Python environment"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt

echo "== 2/5 Download CONPACT from Hugging Face"
python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
snapshot_download("ZhiqiEliWang/conpact", repo_type="dataset", local_dir="data/conpact",
                  allow_patterns=["wildchat/*", "hermes/*", "swe_natural/*"])
for cfg in ("wildchat", "hermes", "swe_natural"):
    n = len(list(Path("data/conpact", cfg).glob("*.parquet")))
    if n == 0:
        raise SystemExit(f"Download incomplete: no parquet files in data/conpact/{cfg}")
    print(f"   {cfg}: {n} parquet file(s)")
PY

echo "== 3/5 Build the dataset"
(cd scripts && python build_dataset.py --local_dir ../data/conpact --out_dir ../data/build > ../data/build_summary.txt)
echo "   summary written to data/build_summary.txt"

echo "== 4/5 Apply the labels"
(cd scripts && python apply_labels.py)

echo "== 5/5 Finalize"
(cd scripts && python finalize_labels.py --build_dir ../data/build --out_dir ../data/final > ../data/final_summary.txt)

echo
echo "Done. Final dataset in data/final/:"
ls data/final