#!/usr/bin/env bash
#SBATCH --job-name=canto-ext-r1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/bolin/cantonese-asr/logs/%x-%A_%a.out

source /home/bolin/cantonese-asr/slurm/common.sh

LR=2e-5
SCHEDULER=linear
BATCH_SIZE=8
GRAD_ACCUM=2
MIX_ROOT=artifacts/manifests/external/round1
OUTPUT_ROOT=outputs/external-round1

train_one() {
  local model=$1
  local manifest=$2
  local output_dir=$3
  local trial=$4
  local epochs=$5

  "$PYTHON" train.py \
    --model "$model" \
    --train-manifest "$manifest" \
    --validation-manifest artifacts/manifests/validation.jsonl \
    --train-probe-manifest artifacts/manifests/train_probe.jsonl \
    --project-root "$PROJECT_DIR" \
    --output-dir "$output_dir" \
    --trial-name "$trial" \
    --learning-rate "$LR" \
    --scheduler "$SCHEDULER" \
    --epochs "$epochs" \
    --batch-size "$BATCH_SIZE" \
    --eval-batch-size 4 \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --warmup-ratio 0.05 \
    --weight-decay 0.01 \
    --logging-steps 25 \
    --num-workers 4 \
    --seed 42 \
    --bf16 \
    --tf32
}

case "$SLURM_ARRAY_TASK_ID" in
  0)
    TRIAL=ext-r1-control-official
    train_one artifacts/models/whisper-small "$MIX_ROOT/control_official.jsonl" \
      "$OUTPUT_ROOT/$TRIAL" "$TRIAL" 3
    plot_run "$OUTPUT_ROOT/$TRIAL" "$TRIAL"
    ;;
  1)
    TRIAL=ext-r1-external25-balanced
    train_one artifacts/models/whisper-small "$MIX_ROOT/external25.jsonl" \
      "$OUTPUT_ROOT/$TRIAL" "$TRIAL" 3
    plot_run "$OUTPUT_ROOT/$TRIAL" "$TRIAL"
    ;;
  2)
    TRIAL=ext-r1-external50-balanced
    train_one artifacts/models/whisper-small "$MIX_ROOT/external50.jsonl" \
      "$OUTPUT_ROOT/$TRIAL" "$TRIAL" 3
    plot_run "$OUTPUT_ROOT/$TRIAL" "$TRIAL"
    ;;
  3)
    PRETRIAL=ext-r1-preadapt-balanced
    FINETRIAL=ext-r1-preadapt-then-official
    train_one artifacts/models/whisper-small "$MIX_ROOT/preadapt_external.jsonl" \
      "$OUTPUT_ROOT/$PRETRIAL" "$PRETRIAL" 1
    train_one "$OUTPUT_ROOT/$PRETRIAL/best_model" artifacts/manifests/train.jsonl \
      "$OUTPUT_ROOT/$FINETRIAL" "$FINETRIAL" 3
    plot_run "$OUTPUT_ROOT/$PRETRIAL" "$PRETRIAL"
    plot_run "$OUTPUT_ROOT/$FINETRIAL" "$FINETRIAL"
    ;;
  *)
    echo "Unexpected task id: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac
