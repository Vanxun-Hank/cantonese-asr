#!/usr/bin/env bash
#SBATCH --job-name=canto-baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/bolin/cantonese-asr/logs/%x-%j.out

# Single-trial reference run. The maintained four-way comparison remains
# slurm/grid_search.slurm; this entry never submits a nested Slurm job.
source /home/bolin/cantonese-asr/slurm/common.sh

TRIAL="baseline-lr1e-5-linear"
RUN_DIR="outputs/baseline/$TRIAL"

"$PYTHON" train.py \
  --model artifacts/models/whisper-small \
  --train-manifest artifacts/manifests/train.jsonl \
  --validation-manifest artifacts/manifests/validation.jsonl \
  --train-probe-manifest artifacts/manifests/train_probe.jsonl \
  --project-root "$PROJECT_DIR" \
  --output-dir "$RUN_DIR" \
  --trial-name "$TRIAL" \
  --learning-rate 1e-5 \
  --scheduler linear \
  --epochs 3 \
  --batch-size "${BATCH_SIZE:-4}" \
  --eval-batch-size 4 \
  --gradient-accumulation-steps "${GRAD_ACCUM:-4}" \
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
