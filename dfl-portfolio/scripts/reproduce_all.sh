#!/usr/bin/env bash
# Re-run every experiment in the report.  About 4 hours on a 2-core laptop CPU.
#   bash scripts/reproduce_all.sh          # everything
#   QUICK=1 bash scripts/reproduce_all.sh  # 1-seed, 3-fold smoke test (5-10 minutes)
set -euo pipefail
cd "$(dirname "$0")/.."
P=${PROCESSES:-2}
Q=${QUICK:+--quick}
mkdir -p results/logs

python scripts/download_data.py                                   | tee results/logs/download.log
python -m pytest -q tests                                          | tee results/logs/tests.log
python scripts/run_walkforward.py configs/sector9.yaml        --processes $P $Q | tee results/logs/sector9.log
python scripts/run_synthetic.py   configs/synthetic.yaml      --processes $P $Q | tee results/logs/synthetic.log
python scripts/run_synthetic.py   configs/synthetic_beta.yaml --processes $P $Q | tee results/logs/synthetic_beta.log
python scripts/run_walkforward.py configs/megacap10.yaml      --processes $P $Q | tee results/logs/megacap10.log
python scripts/run_walkforward.py configs/megacap10_notebook_replication.yaml --processes $P $Q | tee results/logs/notebook_replication.log
python scripts/run_walkforward.py configs/sector9_notebook_replication.yaml --processes $P $Q | tee results/logs/sector9_notebook_replication.log
python scripts/run_walkforward.py configs/ff12.yaml           --processes $P $Q | tee results/logs/ff12.log
python scripts/run_walkforward.py configs/ff12_eta.yaml       --processes $P $Q | tee results/logs/ff12_eta.log
python scripts/run_walkforward.py configs/robust_sector9.yaml --processes $P $Q | tee results/logs/robust.log
python scripts/run_walkforward.py configs/robust_sector9.yaml --processes $P $Q --frozen | tee results/logs/robust_frozen.log
if [ -z "${QUICK:-}" ]; then
  python scripts/run_synthetic.py configs/synthetic_lambda.yaml --processes $P | tee results/logs/synthetic_lambda.log
  python scripts/run_ablation.py  configs/ablation_sector9.yaml --processes $P | tee results/logs/ablation.log
fi
python scripts/make_figures.py $Q | tee results/logs/figures.log
