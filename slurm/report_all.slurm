#!/usr/bin/env bash
#SBATCH --job-name=canto-report
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

"$PYTHON" scripts/plot_experiments.py \
  --outputs-root outputs \
  --data-report artifacts/manifests/data_report.json \
  --ood-data-report artifacts/manifests/ood/v1/data_report.json \
  --report-dir artifacts/reports/experiments/all
