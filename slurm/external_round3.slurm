#!/usr/bin/env bash
#SBATCH --job-name=canto-ext-r3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/home/bolin/cantonese-asr/logs/%x-%A_%a.out

set -euo pipefail
source /home/bolin/cantonese-asr/slurm/common.sh

INDEX=${SLURM_ARRAY_TASK_ID:?Missing array task ID}
MIX_ROOT=artifacts/manifests/external/round3

case "$INDEX" in
  0)
    TRIAL=external73
    MANIFEST="$MIX_ROOT/external73.jsonl"
    ;;
  1)
    TRIAL=external80
    MANIFEST="$MIX_ROOT/external80.jsonl"
    ;;
  2)
    TRIAL=external85
    MANIFEST="$MIX_ROOT/external85.jsonl"
    ;;
  3)
    TRIAL=all_train
    MANIFEST="$MIX_ROOT/all_train.jsonl"
    ;;
  *)
    echo "Unexpected task id: $INDEX (expected 0, 1, 2, or 3)" >&2
    exit 2
    ;;
esac

if [[ ! -f "$MANIFEST" ]]; then
  echo "Round 3 training manifest not found: $MANIFEST" >&2
  exit 3
fi

RUN_DIR="outputs/external-round3/$TRIAL"

# train.py performs Full SFT with AdamW, zh/transcribe generation, gradient
# checkpointing, gradient clipping at 1.0, and eval/save once per epoch.
"$PYTHON" train.py \
  --model artifacts/models/whisper-small \
  --train-manifest "$MANIFEST" \
  --validation-manifest artifacts/manifests/validation.jsonl \
  --train-probe-manifest artifacts/manifests/train_probe.jsonl \
  --project-root "$PROJECT_DIR" \
  --output-dir "$RUN_DIR" \
  --trial-name "$TRIAL" \
  --learning-rate 2e-5 \
  --scheduler cosine \
  --epochs 3 \
  --batch-size 8 \
  --eval-batch-size 4 \
  --gradient-accumulation-steps 2 \
  --warmup-ratio 0.05 \
  --weight-decay 0.01 \
  --logging-steps 25 \
  --num-workers 4 \
  --save-total-limit 3 \
  --seed 42 \
  --generation-max-length 225 \
  --bf16 \
  --tf32

"$PYTHON" scripts/evaluate_checkpoints.py \
  --run-dir "$RUN_DIR" \
  --audio-dir "$PROJECT_DIR" \
  --validation-manifest artifacts/manifests/validation.jsonl \
  --train-probe-manifest artifacts/manifests/train_probe.jsonl \
  --ood-panel-manifest artifacts/manifests/ood/v1/ood_panel.jsonl \
  --batch-size 8 \
  --num-workers 4 \
  --generation-max-length 225

SELECTION_STATUS=0
"$PYTHON" scripts/select_best_checkpoint.py \
  --run-dir "$RUN_DIR" \
  --baseline-metrics outputs/zero-shot/validation/metrics.json \
  --min-sentence-accuracy 0.8219 \
  --max-cer 0.1163 \
  --output "$RUN_DIR/checkpoint_selection.json" || SELECTION_STATUS=$?

"$PYTHON" scripts/plot_experiments.py \
  --outputs-root "$RUN_DIR" \
  --data-report artifacts/manifests/data_report.json \
  --ood-data-report artifacts/manifests/ood/v1/data_report.json \
  --report-dir "artifacts/reports/experiments/by_trial/$TRIAL"

if [[ "$SELECTION_STATUS" -ne 0 ]]; then
  echo "No checkpoint passed the fixed validation guardrails for $TRIAL; diagnostics were preserved." >&2
  exit "$SELECTION_STATUS"
fi
