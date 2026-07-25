#!/usr/bin/env bash
#SBATCH --job-name=canto-r8-ablate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/bolin/cantonese-asr/logs/%x-%A_%a.out

set -euo pipefail
source /home/bolin/cantonese-asr/slurm/common.sh

INDEX=${SLURM_ARRAY_TASK_ID:?Missing array task ID}
MANIFESTS=(
  artifacts/manifests/round8/source-ablation/official_mdcc57.jsonl
  artifacts/manifests/round8/source-ablation/official_cv57.jsonl
  artifacts/manifests/external/round2/external73.jsonl
)
NAMES=(
  official-mdcc57-fresh
  official-cv57-fresh
  round2-exact-fresh
)
EXPECTED_ROWS=(14743 14743 23304)

MANIFEST=${MANIFESTS[$INDEX]}
TRIAL=${NAMES[$INDEX]}
EXPECTED=${EXPECTED_ROWS[$INDEX]}
RUN_DIR="outputs/round8-source-ablation/$TRIAL"
if [[ -e "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty Round 8 ablation directory: $RUN_DIR" >&2
  exit 3
fi

"$PYTHON" - "$MANIFEST" "$EXPECTED" "$INDEX" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

from cantonese_asr.io import read_jsonl, sha256_file

manifest = Path(sys.argv[1])
expected = int(sys.argv[2])
index = int(sys.argv[3])
rows = read_jsonl(manifest)
if len(rows) != expected or len({str(row["id"]) for row in rows}) != expected:
    raise SystemExit(f"Unexpected manifest row/ID count: {manifest} {len(rows)}")
counts = Counter(str(row.get("source")) for row in rows)
expected_sources = (
    {"official": 6_292, "mdcc": 8_451},
    {"official": 6_292, "common_voice_26_zh_HK": 8_451},
    {"official": 6_292, "common_voice_26_zh_HK": 8_451, "mdcc": 8_561},
)[index]
if counts != expected_sources:
    raise SystemExit(f"Unexpected source composition: {counts}")
if index == 2 and sha256_file(manifest) != (
    "2ba7edef6811b0d3e96b3347aa70f4237ec27735652d3d313bce62525f1cdea6"
):
    raise SystemExit("Round 2 exact control Manifest SHA mismatch")
PY

"$PYTHON" train.py \
  --model artifacts/models/whisper-small \
  --train-manifest "$MANIFEST" \
  --validation-manifest artifacts/manifests/validation.jsonl \
  --train-probe-manifest artifacts/manifests/train_probe.jsonl \
  --project-root "$PROJECT_DIR" \
  --output-dir "$RUN_DIR" \
  --trial-name "$TRIAL" \
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
  echo "No checkpoint passed fixed validation guardrails for $TRIAL." >&2
  exit "$SELECTION_STATUS"
fi
