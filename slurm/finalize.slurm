#!/usr/bin/env bash
#SBATCH --job-name=canto-finalize
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

RUN_DIR=outputs/final/final-lr2e-5-linear-b8-ga2
SELECTION="$RUN_DIR/checkpoint_selection.json"
SUBMISSION_DIR=outputs/submission
SUBMISSION_ZIP="$SUBMISSION_DIR/submission.zip"
mkdir -p "$SUBMISSION_DIR" artifacts/reports/final

"$PYTHON" scripts/select_best_checkpoint.py \
  --run-dir "$RUN_DIR" \
  --baseline-metrics outputs/zero-shot/validation/metrics.json \
  --output "$SELECTION"
MODEL_DIR=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["model_dir"])' "$SELECTION")

"$PYTHON" predict.py \
  --model_dir "$MODEL_DIR" \
  --processor_dir "$RUN_DIR" \
  --audio_dir artifacts/data/train_raw/4856-10393 \
  --test_list artifacts/datasets/official/template_pre.jsonl \
  --output_jsonl "$RUN_DIR/public_review/predictions.jsonl" \
  --batch_size 8 \
  --generation-max-length 225
"$PYTHON" scripts/evaluate_predictions.py \
  --pred-jsonl "$RUN_DIR/public_review/predictions.jsonl" \
  --reference artifacts/datasets/official/template_pre.jsonl \
  --reference-field ref_text \
  --report-dir "$RUN_DIR/public_review"
"$PYTHON" scripts/validate_prediction_order.py \
  --test-list artifacts/datasets/official/template_pre.jsonl \
  --predictions "$RUN_DIR/public_review/predictions.jsonl" \
  --output-report artifacts/reports/final/public_review_order.json

"$PYTHON" predict.py \
  --model_dir "$MODEL_DIR" \
  --processor_dir "$RUN_DIR" \
  --audio_dir artifacts/data/test_audio \
  --test_list artifacts/datasets/official/template.jsonl \
  --output_jsonl "$RUN_DIR/formal_protocol/predictions.jsonl" \
  --batch_size 8 \
  --generation-max-length 225
"$PYTHON" scripts/validate_prediction_order.py \
  --test-list artifacts/datasets/official/template.jsonl \
  --predictions "$RUN_DIR/formal_protocol/predictions.jsonl" \
  --output-report artifacts/reports/final/formal_protocol_order.json

"$PYTHON" scripts/package_submission.py \
  --model-dir "$MODEL_DIR" \
  --predict-py predict.py \
  --requirements requirements-submission.txt \
  --output-zip "$SUBMISSION_ZIP"
"$PYTHON" scripts/verify_submission.py \
  --submission-zip "$SUBMISSION_ZIP" \
  --audio-dir artifacts/data/test_audio \
  --test-list artifacts/datasets/official/template.jsonl \
  --max-samples 32 \
  --output-report artifacts/reports/final/offline_submission_verification.json

"$PYTHON" scripts/plot_experiments.py \
  --outputs-root outputs \
  --data-report artifacts/manifests/data_report.json \
  --ood-data-report artifacts/manifests/ood/v1/data_report.json \
  --report-dir artifacts/reports/experiments/all
"$PYTHON" scripts/check_submission.py \
  --submission-zip "$SUBMISSION_ZIP" \
  --output-report artifacts/reports/submission_acceptance.json
