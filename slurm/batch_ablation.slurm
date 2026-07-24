#!/usr/bin/env bash
#SBATCH --job-name=canto-batch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/%x-%A_%a.out

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

BEST_LR=${BEST_LR:?Submit with --export=ALL,BEST_LR=...,BEST_SCHEDULER=...,BATCH_SIZE=...}
BEST_SCHEDULER=${BEST_SCHEDULER:?}
BATCH_SIZE=${BATCH_SIZE:?}
TARGETS=(16 32)
ARRAY_INDEX=${SLURM_ARRAY_TASK_ID:?Missing SLURM_ARRAY_TASK_ID}
TARGET=${TARGETS[$ARRAY_INDEX]}
GRAD_ACCUM=$(( (TARGET + BATCH_SIZE - 1) / BATCH_SIZE ))
EFFECTIVE=$(( BATCH_SIZE * GRAD_ACCUM ))
TRIAL="batch-target${TARGET}-actual${EFFECTIVE}-b${BATCH_SIZE}-ga${GRAD_ACCUM}"
RUN_DIR="outputs/batch-ablation/$TRIAL"

"$PYTHON" train.py \
  --model artifacts/models/whisper-small \
  --train-manifest artifacts/manifests/train.jsonl \
  --validation-manifest artifacts/manifests/validation.jsonl \
  --train-probe-manifest artifacts/manifests/train_probe.jsonl \
  --project-root "$PROJECT_DIR" \
  --output-dir "$RUN_DIR" \
  --trial-name "$TRIAL" \
  --learning-rate "$BEST_LR" \
  --scheduler "$BEST_SCHEDULER" \
  --epochs 3 \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size 4 \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --warmup-ratio 0.05 \
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
