#!/usr/bin/env bash
#SBATCH --job-name=canto-r8-mdcc73
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MANIFEST=artifacts/manifests/round8/official-mdcc73/train.jsonl
DATA_REPORT=artifacts/manifests/round8/official-mdcc73/data_report.json
RUN_DIR=outputs/round8-mdcc-source-isolation/official-mdcc73-fresh
if [[ -e "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty Round 8 run directory: $RUN_DIR" >&2
  exit 3
fi

"$PYTHON" - "$MANIFEST" "$DATA_REPORT" <<'PY'
import json
import sys
from pathlib import Path

from cantonese_asr.io import sha256_file

manifest = Path(sys.argv[1])
report = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if report.get("passed") is not True:
    raise SystemExit("Round 8 data report is not marked passed")
if report["output"]["rows"] != 23_304:
    raise SystemExit(f"Unexpected training rows: {report['output']}")
if report["output"]["sources"] != {"mdcc": 17_012, "official": 6_292}:
    raise SystemExit(f"Unexpected training sources: {report['output']}")
if report["output"]["sha256"] != sha256_file(manifest):
    raise SystemExit("Round 8 training Manifest SHA mismatch")
if report["sampling"]["without_replacement"] is not True:
    raise SystemExit("Round 8 MDCC sampling was not without replacement")
PY

"$PYTHON" train.py \
  --model artifacts/models/whisper-small \
  --train-manifest "$MANIFEST" \
  --validation-manifest artifacts/manifests/validation.jsonl \
  --train-probe-manifest artifacts/manifests/train_probe.jsonl \
  --project-root "$PROJECT_DIR" \
  --output-dir "$RUN_DIR" \
  --trial-name official-mdcc73-fresh \
  --freeze-encoder-layers 0 \
  --learning-rate 2e-5 \
  --scheduler cosine \
  --epochs 3 \
  --batch-size 8 \
  --eval-batch-size 4 \
  --gradient-accumulation-steps 2 \
  --warmup-ratio 0.05 \
  --weight-decay 0.01 \
  --logging-steps 25 \
  --num-workers 4 \
  --save-total-limit 3 \
  --generation-max-length 225 \
  --no-apply-spec-augment \
  --seed 42 \
  --bf16 \
  --tf32

"$PYTHON" scripts/evaluate_checkpoints.py \
  --run-dir "$RUN_DIR" \
  --audio-dir "$PROJECT_DIR" \
  --validation-manifest artifacts/manifests/validation.jsonl \
  --train-probe-manifest artifacts/manifests/train_probe.jsonl \
  --ood-panel-manifest artifacts/manifests/ood/v1/ood_panel.jsonl \
  --diagnostic-manifest ood_cv=artifacts/manifests/round8/official-mdcc73/ood_cv.jsonl \
  --diagnostic-manifest ood_mdcc=artifacts/manifests/round8/official-mdcc73/ood_mdcc.jsonl \
  --batch-size 8 \
  --num-workers 4 \
  --generation-max-length 225

SELECTION_STATUS=0
"$PYTHON" scripts/select_best_checkpoint.py \
  --run-dir "$RUN_DIR" \
  --baseline-metrics outputs/zero-shot/validation/metrics.json \
  --min-sentence-accuracy 0.8219 \
  --max-cer 0.1163 \
  --output "$RUN_DIR/checkpoint_selection.json" || SELECTION_STATUS=$?
if [[ "$SELECTION_STATUS" -ne 0 ]]; then
  echo "No Round 8 checkpoint passed fixed validation guardrails." >&2
  exit "$SELECTION_STATUS"
fi
