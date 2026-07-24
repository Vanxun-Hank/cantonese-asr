#!/usr/bin/env bash
#SBATCH --job-name=canto-reeval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-4%4
#SBATCH --output=logs/%x-%A_%a.out

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

RUNS=(
  outputs/external-round1/ext-r1-control-official
  outputs/external-round1/ext-r1-external25-balanced
  outputs/external-round1/ext-r1-external50-balanced
  outputs/external-round1/ext-r1-preadapt-then-official
  outputs/final/final-lr2e-5-linear-b8-ga2
)
INDEX=${SLURM_ARRAY_TASK_ID:?Missing array task ID}
RUN_DIR=${RUNS[$INDEX]}

"$PYTHON" scripts/evaluate_checkpoints.py \
  --run-dir "$RUN_DIR" \
  --audio-dir "$PROJECT_DIR" \
  --validation-manifest artifacts/manifests/validation.jsonl \
  --train-probe-manifest artifacts/manifests/train_probe.jsonl \
  --ood-panel-manifest artifacts/manifests/ood/v1/ood_panel.jsonl \
  --batch-size 8 \
  --num-workers 4 \
  --generation-max-length 225

"$PYTHON" scripts/select_best_checkpoint.py \
  --run-dir "$RUN_DIR" \
  --baseline-metrics outputs/zero-shot/validation/metrics.json \
  --max-cer 0.1163 \
  --min-sentence-accuracy 0.8219 \
  --output "$RUN_DIR/checkpoint_selection.json"

TRIAL=$(basename "$RUN_DIR")
"$PYTHON" scripts/plot_experiments.py \
  --outputs-root "$RUN_DIR" \
  --data-report artifacts/manifests/data_report.json \
  --ood-data-report artifacts/manifests/ood/v1/data_report.json \
  --report-dir "artifacts/reports/experiments/by_trial/$TRIAL"
