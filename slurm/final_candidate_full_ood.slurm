#!/usr/bin/env bash
#SBATCH --job-name=canto-final-ood
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/bolin/cantonese-asr/logs/%x-%j.out

source /home/bolin/cantonese-asr/slurm/common.sh

FINAL_DIR=outputs/final-candidate
SELECTION="$FINAL_DIR/global_selection.json"
REPORT_DIR="$FINAL_DIR/full_ood"
PREDICTIONS="$REPORT_DIR/predictions.jsonl"
mkdir -p "$REPORT_DIR"

"$PYTHON" scripts/select_global_candidate.py \
  --selection outputs/external-round1/ext-r1-control-official/checkpoint_selection.json \
  --selection outputs/external-round1/ext-r1-external25-balanced/checkpoint_selection.json \
  --selection outputs/external-round1/ext-r1-external50-balanced/checkpoint_selection.json \
  --selection outputs/external-round1/ext-r1-preadapt-then-official/checkpoint_selection.json \
  --selection outputs/final/final-lr2e-5-linear-b8-ga2/checkpoint_selection.json \
  --output "$SELECTION"

MODEL_DIR=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["model_dir"])' "$SELECTION")
RUN_DIR=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["run_dir"])' "$SELECTION")

"$PYTHON" predict.py \
  --model_dir "$MODEL_DIR" \
  --processor_dir "$RUN_DIR" \
  --audio_dir "$PROJECT_DIR" \
  --test_list artifacts/manifests/ood/v1/ood_all.jsonl \
  --output_jsonl "$PREDICTIONS" \
  --batch_size 8 \
  --generation-max-length 225

"$PYTHON" scripts/evaluate_predictions.py \
  --pred-jsonl "$PREDICTIONS" \
  --reference artifacts/manifests/ood/v1/ood_all.jsonl \
  --reference-field text \
  --report-dir "$REPORT_DIR" \
  --write-symmetric-t2s

"$PYTHON" scripts/evaluate_manifest_loss.py \
  --model-dir "$MODEL_DIR" \
  --processor-dir "$RUN_DIR" \
  --manifest artifacts/manifests/ood/v1/ood_all.jsonl \
  --project-root "$PROJECT_DIR" \
  --output "$REPORT_DIR/teacher_forced_loss.json" \
  --batch-size 8 \
  --num-workers 4

cp "$SELECTION" "$REPORT_DIR/global_selection.json"

# Public self-test and formal protocol checks are reviews only; neither selects a model.
"$PYTHON" predict.py \
  --model_dir "$MODEL_DIR" \
  --processor_dir "$RUN_DIR" \
  --audio_dir artifacts/data/train_raw/4856-10393 \
  --test_list artifacts/datasets/official/template_pre.jsonl \
  --output_jsonl "$FINAL_DIR/public_review_predictions.jsonl" \
  --batch_size 8 \
  --generation-max-length 225
"$PYTHON" scripts/evaluate_predictions.py \
  --pred-jsonl "$FINAL_DIR/public_review_predictions.jsonl" \
  --reference artifacts/datasets/official/template_pre.jsonl \
  --reference-field ref_text \
  --report-dir "$FINAL_DIR/public_review"

"$PYTHON" predict.py \
  --model_dir "$MODEL_DIR" \
  --processor_dir "$RUN_DIR" \
  --audio_dir artifacts/data/test_audio \
  --test_list artifacts/datasets/official/template.jsonl \
  --output_jsonl "$FINAL_DIR/formal_predictions.jsonl" \
  --batch_size 8 \
  --generation-max-length 225
"$PYTHON" scripts/validate_prediction_order.py \
  --test-list artifacts/datasets/official/template.jsonl \
  --predictions "$FINAL_DIR/formal_predictions.jsonl" \
  --output-report "$FINAL_DIR/formal_order.json"

"$PYTHON" scripts/package_submission.py \
  --model-dir "$MODEL_DIR" \
  --predict-py predict.py \
  --requirements requirements-submission.txt \
  --output-zip "$FINAL_DIR/submission.zip"
"$PYTHON" scripts/verify_submission.py \
  --submission-zip "$FINAL_DIR/submission.zip" \
  --audio-dir artifacts/data/test_audio \
  --test-list artifacts/datasets/official/template.jsonl \
  --max-samples 32 \
  --output-report "$FINAL_DIR/offline_submission_verification.json"

"$PYTHON" scripts/check_goal1.py \
  --submission-zip "$FINAL_DIR/submission.zip" \
  --output-report artifacts/reports/goal1_acceptance.json
