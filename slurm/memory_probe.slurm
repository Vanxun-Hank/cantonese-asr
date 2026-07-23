#!/usr/bin/env bash
#SBATCH --job-name=canto-mem
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/bolin/cantonese-asr/logs/%x-%A_%a.out

source /home/bolin/cantonese-asr/slurm/common.sh

BATCHES=(4 6 8)
ACCUMULATIONS=(4 3 2)
INDEX=${SLURM_ARRAY_TASK_ID:?}
BATCH_SIZE=${BATCHES[$INDEX]}
GRAD_ACCUM=${ACCUMULATIONS[$INDEX]}
TRIAL="mem-b${BATCH_SIZE}-ga${GRAD_ACCUM}"
RUN_DIR="outputs/memory-probe/$TRIAL"

"$PYTHON" train.py \
  --model artifacts/models/whisper-small \
  --train-manifest artifacts/manifests/train.jsonl \
  --validation-manifest artifacts/manifests/validation.jsonl \
  --train-probe-manifest artifacts/manifests/train_probe.jsonl \
  --project-root "$PROJECT_DIR" \
  --output-dir "$RUN_DIR" \
  --trial-name "$TRIAL" \
  --learning-rate 1e-5 \
  --epochs 1 \
  --max-steps 20 \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size 4 \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --max-train-samples 128 \
  --max-validation-samples 32 \
  --max-train-probe-samples 32 \
  --logging-steps 1 \
  --num-workers 2 \
  --bf16 \
  --tf32
plot_run "$RUN_DIR" "$TRIAL"

