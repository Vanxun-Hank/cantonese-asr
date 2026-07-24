#!/usr/bin/env bash
#SBATCH --job-name=canto-final-ood
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

FINAL_DIR=outputs/final-candidate
SELECTION="$FINAL_DIR/global_selection.json"
REPORT_DIR="$FINAL_DIR/full_ood"
PREDICTIONS="$REPORT_DIR/predictions.jsonl"
mkdir -p "$REPORT_DIR"

ROUND3_SELECTIONS=()
for report in \
  outputs/external-round3/external73/checkpoint_selection.json \
  outputs/external-round3/external80/checkpoint_selection.json \
  outputs/external-round3/external85/checkpoint_selection.json \
  outputs/external-round3/all_train/checkpoint_selection.json
do
  if [[ ! -f "$report" ]]; then
    echo "Skipping Round 3 selection report (missing): $report" >&2
    continue
  fi

  if "$PYTHON" - "$report" <<'PY'
import json
import math
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        report = json.load(handle)
except (OSError, json.JSONDecodeError) as error:
    print(f"Invalid Round 3 selection report: {error}", file=sys.stderr)
    raise SystemExit(1)

if not isinstance(report, dict):
    print("Round 3 selection report must be a JSON object", file=sys.stderr)
    raise SystemExit(1)
surface_is_valid = (
    report.get("selection_surface") == "internal_validation_only"
)
if not surface_is_valid:
    print("Round 3 selection report is not validation-only", file=sys.stderr)
    raise SystemExit(1)

selected = report.get("selected")
eligible_is_valid = (
    isinstance(selected, dict) and selected.get("eligible") is True
)
if not eligible_is_valid:
    print("Round 3 selection report has no eligible checkpoint", file=sys.stderr)
    raise SystemExit(1)

model_dir = selected.get("model_dir")
if not isinstance(model_dir, str) or not model_dir.strip():
    print("Round 3 selected checkpoint has no valid model_dir", file=sys.stderr)
    raise SystemExit(1)

try:
    metrics = [
        float(selected[field])
        for field in ("sentence_accuracy_tol2", "cer", "validation_loss")
    ]
    int(selected["checkpoint_step"])
except (KeyError, TypeError, ValueError) as error:
    print(f"Round 3 selected checkpoint is incomplete: {error}", file=sys.stderr)
    raise SystemExit(1)
if not all(math.isfinite(value) for value in metrics):
    print("Round 3 selected checkpoint has non-finite metrics", file=sys.stderr)
    raise SystemExit(1)
PY
  then
    ROUND3_SELECTIONS+=(--selection "$report")
  else
    echo "Skipping Round 3 selection report (invalid or ineligible): $report" >&2
  fi
done

"$PYTHON" scripts/select_global_candidate.py \
  --selection outputs/external-round1/ext-r1-control-official/checkpoint_selection.json \
  --selection outputs/external-round1/ext-r1-external25-balanced/checkpoint_selection.json \
  --selection outputs/external-round1/ext-r1-external50-balanced/checkpoint_selection.json \
  --selection outputs/external-round1/ext-r1-preadapt-then-official/checkpoint_selection.json \
  --selection outputs/final/final-lr2e-5-linear-b8-ga2/checkpoint_selection.json \
  --selection outputs/external-round2/ext-r2-external50-e5-linear/checkpoint_selection.json \
  --selection outputs/external-round2/ext-r2-external73-e3-linear/checkpoint_selection.json \
  --selection outputs/external-round2/ext-r2-external73-e5-linear/checkpoint_selection.json \
  --selection outputs/external-round2/ext-r2-external73-e3-cosine/checkpoint_selection.json \
  "${ROUND3_SELECTIONS[@]}" \
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

"$PYTHON" scripts/check_submission.py \
  --submission-zip "$FINAL_DIR/submission.zip" \
  --output-report artifacts/reports/submission_acceptance.json
