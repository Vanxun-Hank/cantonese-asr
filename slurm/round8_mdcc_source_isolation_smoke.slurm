#!/usr/bin/env bash
#SBATCH --job-name=canto-r8-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/bolin/cantonese-asr/logs/%x-%j.out

set -euo pipefail
source /home/bolin/cantonese-asr/slurm/common.sh

MANIFEST=artifacts/manifests/round8/official-mdcc73/smoke32.jsonl
RUN_DIR=outputs/round8-mdcc-source-isolation/smoke32
if [[ -e "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty Round 8 smoke directory: $RUN_DIR" >&2
  exit 3
fi

"$PYTHON" train.py \
  --model artifacts/models/whisper-small \
  --train-manifest "$MANIFEST" \
  --validation-manifest "$MANIFEST" \
  --train-probe-manifest "$MANIFEST" \
  --project-root "$PROJECT_DIR" \
  --output-dir "$RUN_DIR" \
  --trial-name round8-official-mdcc73-smoke32 \
  --freeze-encoder-layers 0 \
  --learning-rate 1e-4 \
  --scheduler linear \
  --epochs 10 \
  --batch-size 2 \
  --eval-batch-size 4 \
  --gradient-accumulation-steps 1 \
  --warmup-ratio 0.05 \
  --weight-decay 0.01 \
  --logging-steps 1 \
  --num-workers 2 \
  --save-total-limit 2 \
  --max-train-samples 32 \
  --max-validation-samples 32 \
  --max-train-probe-samples 32 \
  --generation-max-length 225 \
  --no-apply-spec-augment \
  --seed 42 \
  --bf16 \
  --tf32

"$PYTHON" scripts/evaluate_checkpoints.py \
  --run-dir "$RUN_DIR" \
  --audio-dir "$PROJECT_DIR" \
  --validation-manifest "$MANIFEST" \
  --train-probe-manifest "$MANIFEST" \
  --batch-size 8 \
  --num-workers 2 \
  --generation-max-length 225

"$PYTHON" - "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
if config["model"]["full_sft"] is not True:
    raise SystemExit("Round 8 smoke was not Full SFT")
if config["model"]["spec_augment"]["enabled"] is not False:
    raise SystemExit("Round 8 smoke unexpectedly enabled SpecAugment")
state = json.loads((run_dir / "trainer_state.json").read_text(encoding="utf-8"))
best = Path(state["best_model_checkpoint"]).name
metrics = json.loads(
    (run_dir / "diagnostics" / best / "validation" / "metrics.json").read_text(
        encoding="utf-8"
    )
)
if metrics["sentence_accuracy_tol2"] < 0.75:
    raise SystemExit(f"32-sample smoke overfit failed: {metrics}")
if not (run_dir / "best_model" / "model.safetensors").is_file():
    raise SystemExit("Round 8 smoke best_model is not reloadable")
print(json.dumps({"best_checkpoint": best, "metrics": metrics}, ensure_ascii=False, indent=2))
PY
