#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/bolin/cantonese-asr}
ENV_DIR=${ENV_DIR:-/home/bolin/envs/cantonese-asr-whisper}
PYTHON=${PYTHON:-$ENV_DIR/bin/python}

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment not found: $PYTHON" >&2
  exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_DIR"
mkdir -p logs outputs artifacts/reports/experiments

plot_run() {
  local run_dir=$1
  local report_name=$2
  "$PYTHON" scripts/plot_experiments.py \
    --outputs-root "$run_dir" \
    --data-report artifacts/manifests/data_report.json \
    --report-dir "artifacts/reports/experiments/by_trial/$report_name"
}

