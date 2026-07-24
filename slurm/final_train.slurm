#!/usr/bin/env bash
#SBATCH --job-name=canto-final
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

FINAL_LR=${FINAL_LR:?Submit with FINAL_LR, FINAL_SCHEDULER, BATCH_SIZE and GRAD_ACCUM}
FINAL_SCHEDULER=${FINAL_SCHEDULER:?}
BATCH_SIZE=${BATCH_SIZE:?}
GRAD_ACCUM=${GRAD_ACCUM:?}
TRIAL="final-lr${FINAL_LR}-${FINAL_SCHEDULER}-b${BATCH_SIZE}-ga${GRAD_ACCUM}"
RUN_DIR="outputs/final/$TRIAL"

"$PYTHON" train.py \
  --model artifacts/models/whisper-small \
  --train-manifest artifacts/manifests/train.jsonl \
  --validation-manifest artifacts/manifests/validation.jsonl \
  --train-probe-manifest artifacts/manifests/train_probe.jsonl \
  --project-root "$PROJECT_DIR" \
  --output-dir "$RUN_DIR" \
  --trial-name "$TRIAL" \
  --learning-rate "$FINAL_LR" \
  --scheduler "$FINAL_SCHEDULER" \
  --epochs 8 \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size 4 \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --warmup-ratio 0.05 \
  --early-stopping-patience 2 \
  --logging-steps 25 \
  --num-workers 4 \
  --bf16 \
  --tf32

"$PYTHON" scripts/evaluate_checkpoints.py \
  --run-dir "$RUN_DIR" \
  --audio-dir "$PROJECT_DIR" \
  --validation-manifest artifacts/manifests/validation.jsonl \
  --train-probe-manifest artifacts/manifests/train_probe.jsonl \
  --batch-size 8
plot_run "$RUN_DIR" "$TRIAL"
