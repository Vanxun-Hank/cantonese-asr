#!/usr/bin/env bash
#SBATCH --job-name=canto-zero
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

RUN_DIR=outputs/zero-shot
MODEL_DIR=artifacts/models/whisper-small
mkdir -p "$RUN_DIR/validation" "$RUN_DIR/template_pre"

"$PYTHON" predict.py \
  --model_dir "$MODEL_DIR" \
  --audio_dir "$PROJECT_DIR" \
  --test_list artifacts/manifests/validation.jsonl \
  --output_jsonl "$RUN_DIR/validation/predictions.jsonl" \
  --batch_size 8
"$PYTHON" scripts/evaluate_predictions.py \
  --pred-jsonl "$RUN_DIR/validation/predictions.jsonl" \
  --reference artifacts/manifests/validation.jsonl \
  --reference-field text \
  --report-dir "$RUN_DIR/validation" \
  --metrics-jsonl "$RUN_DIR/metrics.jsonl" \
  --trial zero-shot \
  --split validation

"$PYTHON" predict.py \
  --model_dir "$MODEL_DIR" \
  --audio_dir artifacts/data/train_raw/4856-10393 \
  --test_list artifacts/datasets/official/template_pre.jsonl \
  --output_jsonl "$RUN_DIR/template_pre/predictions.jsonl" \
  --batch_size 8
"$PYTHON" scripts/evaluate_predictions.py \
  --pred-jsonl "$RUN_DIR/template_pre/predictions.jsonl" \
  --reference artifacts/datasets/official/template_pre.jsonl \
  --reference-field ref_text \
  --report-dir "$RUN_DIR/template_pre" \
  --metrics-jsonl "$RUN_DIR/metrics.jsonl" \
  --trial zero-shot \
  --split template_pre

plot_run "$RUN_DIR" zero-shot
