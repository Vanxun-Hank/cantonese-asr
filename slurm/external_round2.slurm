#!/usr/bin/env bash
#SBATCH --job-name=canto-ext-r2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/home/bolin/cantonese-asr/logs/%x-%A_%a.out

source /home/bolin/cantonese-asr/slurm/common.sh

INDEX=${SLURM_ARRAY_TASK_ID:?Missing array task ID}
MANIFESTS=(
  artifacts/manifests/external/round1/external50.jsonl
  artifacts/manifests/external/round2/external73.jsonl
  artifacts/manifests/external/round2/external73.jsonl
  artifacts/manifests/external/round2/external73.jsonl
)
EPOCHS=(5 3 5 3)
SCHEDULERS=(linear linear linear cosine)
NAMES=(
  ext-r2-external50-e5-linear
  ext-r2-external73-e3-linear
  ext-r2-external73-e5-linear
  ext-r2-external73-e3-cosine
)

MANIFEST=${MANIFESTS[$INDEX]}
NUM_EPOCHS=${EPOCHS[$INDEX]}
SCHEDULER=${SCHEDULERS[$INDEX]}
TRIAL=${NAMES[$INDEX]}
RUN_DIR="outputs/external-round2/$TRIAL"

"$PYTHON" train.py \
  --model artifacts/models/whisper-small \
  --train-manifest "$MANIFEST" \
  --validation-manifest artifacts/manifests/validation.jsonl \
  --train-probe-manifest artifacts/manifests/train_probe.jsonl \
  --project-root "$PROJECT_DIR" \
  --output-dir "$RUN_DIR" \
  --trial-name "$TRIAL" \
  --learning-rate 2e-5 \
  --scheduler "$SCHEDULER" \
  --epochs "$NUM_EPOCHS" \
  --batch-size 8 \
  --eval-batch-size 4 \
  --gradient-accumulation-steps 2 \
  --warmup-ratio 0.05 \
  --weight-decay 0.01 \
  --logging-steps 25 \
  --num-workers 4 \
  --save-total-limit "$NUM_EPOCHS" \
  --seed 42 \
  --bf16 \
  --tf32

"$PYTHON" scripts/evaluate_checkpoints.py \
  --run-dir "$RUN_DIR" \
  --audio-dir "$PROJECT_DIR" \
  --validation-manifest artifacts/manifests/validation.jsonl \
  --train-probe-manifest artifacts/manifests/train_probe.jsonl \
  --ood-panel-manifest artifacts/manifests/ood/v1/ood_panel.jsonl \
  --batch-size 8 \
  --num-workers 4 \
  --generation-max-length 225

SELECTION_STATUS=0
"$PYTHON" scripts/select_best_checkpoint.py \
  --run-dir "$RUN_DIR" \
  --baseline-metrics outputs/zero-shot/validation/metrics.json \
  --max-cer 0.1163 \
  --min-sentence-accuracy 0.8219 \
  --output "$RUN_DIR/checkpoint_selection.json" || SELECTION_STATUS=$?

"$PYTHON" scripts/plot_experiments.py \
  --outputs-root "$RUN_DIR" \
  --data-report artifacts/manifests/data_report.json \
  --ood-data-report artifacts/manifests/ood/v1/data_report.json \
  --report-dir "artifacts/reports/experiments/by_trial/$TRIAL"

if [[ "$SELECTION_STATUS" -ne 0 ]]; then
  echo "No checkpoint passed the fixed validation guardrails; diagnostics were preserved." >&2
  exit "$SELECTION_STATUS"
fi
