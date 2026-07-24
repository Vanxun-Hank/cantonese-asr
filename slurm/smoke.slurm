#!/usr/bin/env bash
#SBATCH --job-name=canto-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

RUN_DIR=outputs/smoke-overfit
"$PYTHON" train.py \
  --model artifacts/models/whisper-small \
  --train-manifest artifacts/manifests/train_probe.jsonl \
  --validation-manifest artifacts/manifests/train_probe.jsonl \
  --train-probe-manifest artifacts/manifests/train_probe.jsonl \
  --project-root "$PROJECT_DIR" \
  --output-dir "$RUN_DIR" \
  --trial-name smoke-overfit \
  --learning-rate 1e-4 \
  --scheduler linear \
  --epochs 10 \
  --batch-size 2 \
  --eval-batch-size 4 \
  --gradient-accumulation-steps 1 \
  --warmup-ratio 0.05 \
  --max-train-samples 32 \
  --max-validation-samples 32 \
  --max-train-probe-samples 32 \
  --logging-steps 1 \
  --num-workers 2 \
  --bf16 \
  --tf32

"$PYTHON" scripts/evaluate_checkpoints.py \
  --run-dir "$RUN_DIR" \
  --audio-dir "$PROJECT_DIR" \
  --validation-manifest artifacts/manifests/train_probe.jsonl \
  --train-probe-manifest artifacts/manifests/train_probe.jsonl \
  --batch-size 8
plot_run "$RUN_DIR" smoke-overfit

