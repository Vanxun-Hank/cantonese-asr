#!/usr/bin/env bash
#SBATCH --job-name=canto-r8-md57-fix
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MANIFEST=artifacts/manifests/round8/source-ablation/official_mdcc57.jsonl
RUN_DIR=outputs/round8-source-ablation/official-mdcc57-fresh
CHECKPOINT="$RUN_DIR/checkpoint-1844"
if [[ ! -f "$CHECKPOINT/model.safetensors" ]]; then
  echo "Recovery checkpoint is missing: $CHECKPOINT" >&2
  exit 3
fi
if [[ -d "$RUN_DIR/checkpoint-2766" ]]; then
  echo "Refusing recovery because final checkpoint already exists." >&2
  exit 4
fi

"$PYTHON" train.py \
  --model artifacts/models/whisper-small \
  --train-manifest "$MANIFEST" \
  --validation-manifest artifacts/manifests/validation.jsonl \
  --train-probe-manifest artifacts/manifests/train_probe.jsonl \
  --project-root "$PROJECT_DIR" \
  --output-dir "$RUN_DIR" \
  --trial-name official-mdcc57-fresh \
  --freeze-encoder-layers 0 \
  --learning-rate 2e-5 \
  --scheduler cosine \
  --epochs 3 \
  --batch-size 8 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 2 \
  --warmup-ratio 0.05 \
  --weight-decay 0.01 \
  --logging-steps 25 \
  --num-workers 4 \
  --save-total-limit 3 \
  --generation-max-length 225 \
  --no-apply-spec-augment \
  --resume-from-checkpoint "$CHECKPOINT" \
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
  --generation-max-length 225 \
  --skip-existing

SELECTION_STATUS=0
"$PYTHON" scripts/select_best_checkpoint.py \
  --run-dir "$RUN_DIR" \
  --baseline-metrics outputs/zero-shot/validation/metrics.json \
  --min-sentence-accuracy 0.8219 \
  --max-cer 0.1163 \
  --output "$RUN_DIR/checkpoint_selection.json" || SELECTION_STATUS=$?
if [[ "$SELECTION_STATUS" -ne 0 ]]; then
  echo "Recovered MDCC57 run still has no eligible checkpoint." >&2
  exit "$SELECTION_STATUS"
fi
