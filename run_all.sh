#!/usr/bin/env bash
# Chạy toàn bộ E1–E3 rồi sinh bảng/hình cho báo cáo.
#   bash run_all.sh            # chạy đầy đủ, ghi vào results/
#   QUICK=1 bash run_all.sh    # 1 epoch mỗi mô hình để thử nhanh, ghi vào results_quick/
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${QUICK:-0}" == "1" ]]; then
  R=results_quick; EP1=1; EP2=1; EP3=1
else
  R=results; EP1=15; EP2=20; EP3=15
fi

python -m tests.test_all

python -m E1.train --model all --epochs "$EP1" --out "$R/E1"

python -m E2.check_msa --out "$R/E2/msa_check.json"
python -m E2.bench_msa --out "$R/E2/msa_bench.json"
python -m E2.train --epochs "$EP2" --out "$R/E2"

python -m E3.train --epochs "$EP3" --out "$R/E3"
python -m E3.temporal --e1 "$R/E1" --e3 "$R/E3"

python scripts/make_report_assets.py --results "$R" --out reports/generated
echo "Xong. Biên dịch báo cáo: cd reports && xelatex exercises.tex && xelatex exercises.tex"
