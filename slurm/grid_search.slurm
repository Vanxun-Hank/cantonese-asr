#!/usr/bin/env bash
#SBATCH --job-name=canto-grid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/home/bolin/cantonese-asr/logs/%x-%A_%a.out

source /home/bolin/cantonese-asr/slurm/common.sh

LEARNING_RATES=(5e-6 1e-5 2e-5 1e-5)
SCHEDULERS=(linear linear linear cosine)
INDEX=${SLURM_ARRAY_TASK_ID:?}
LR=${LEARNING_RATES[$INDEX]}
SCHEDULER=${SCHEDULERS[$INDEX]}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
TRIAL="grid-${INDEX}-lr${LR}-${SCHEDULER}-b${BATCH_SIZE}-ga${GRAD_ACCUM}"
RUN_DIR="outputs/grid-round1/$TRIAL"

"$PYTHON" train.py \
  --model artifacts/models/whisper-small \
  --train-manifest artifacts/manifests/train.jsonl \
  --validation-manifest artifacts/manifests/validation.jsonl \
  --train-probe-manifest artifacts/manifests/train_probe.jsonl \
  --project-root "$PROJECT_DIR" \
  --output-dir "$RUN_DIR" \
  --trial-name "$TRIAL" \
  --learning-rate "$LR" \
  --scheduler "$SCHEDULER" \
  --epochs 3 \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size 4 \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --warmup-ratio 0.05 \
  --weight-decay 0.01 \
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
