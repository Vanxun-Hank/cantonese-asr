#!/usr/bin/env bash
#SBATCH --job-name=canto-p2-eval
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/canto-raw-winner-p2-eval-%A.out

set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-/home/bolin/cantonese-asr}
source "$PROJECT_DIR/slurm/common.sh"
cd "$PROJECT_DIR"

ARM=${P2_ARM:?P2_ARM is required}
MODEL_DIR=${P2_MODEL_DIR:?P2_MODEL_DIR is required}
PROCESSOR_DIR=${P2_PROCESSOR_DIR:?P2_PROCESSOR_DIR is required}
SURFACES=${P2_SURFACES:-validation}
OUTPUT_DIR=${P2_EVAL_OUTPUT_DIR:-outputs/raw-winner-p2-structural-probe-v1/evaluation/$ARM}
MAX_SAMPLES=${P2_MAX_SAMPLES:-}
test -d "$MODEL_DIR"
test -d "$PROCESSOR_DIR"
test ! -e "$OUTPUT_DIR"

read -r -a SURFACE_ARGS <<< "$SURFACES"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
EXTRA=()
if [[ -n "$MAX_SAMPLES" ]]; then EXTRA+=(--max-samples "$MAX_SAMPLES"); fi
"$PYTHON" scripts/evaluate_raw_winner_p2_capacity.py \
  --model-dir "$MODEL_DIR" \
  --processor-dir "$PROCESSOR_DIR" \
  --project-root "$PROJECT_DIR" \
  --validation-manifest artifacts/manifests/validation.jsonl \
  --public-manifest artifacts/manifests/public_excluded.jsonl \
  --ood-manifest artifacts/manifests/ood/v1/ood_panel.jsonl \
  --output-dir "$OUTPUT_DIR" \
  --batch-size 1 \
  --surfaces "${SURFACE_ARGS[@]}" \
  "${EXTRA[@]}"
