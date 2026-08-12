#!/usr/bin/env bash
#SBATCH --job-name=canto-p2-cap
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=logs/canto-raw-winner-p2-cap-%A.out

set -euo pipefail
source "$(dirname "$0")/common.sh"

CONFIG=${P2_CONFIG:-configs/rounds/raw_winner_p2_structural_probe.json}
ARM=${P2_ARM:?P2_ARM is required}
EXPECTED_NODE=${P2_EXPECTED_NODE:?P2_EXPECTED_NODE is required}
if [[ $(hostname -s) != "$EXPECTED_NODE" ]]; then
  echo "Wrong node: $(hostname -s), expected $EXPECTED_NODE" >&2
  exit 31
fi

readarray -t VALUES < <("$PYTHON" - "$CONFIG" "$ARM" <<'PY'
import json,sys
c=json.load(open(sys.argv[1])); a=c["arms"][sys.argv[2]]
print(c["models"][a["model"]]); print(a["world_size"]); print(a["batch"])
print(a["accum"]); print(a["lr"]); print(a["method"])
PY
)
MODEL=${VALUES[0]}; WORLD=${VALUES[1]}; BATCH=${VALUES[2]}; ACCUM=${VALUES[3]}
LR=${VALUES[4]}; METHOD=${VALUES[5]}
VISIBLE_COUNT=$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')
if [[ "$VISIBLE_COUNT" -ne "$WORLD" ]]; then
  echo "Visible GPU count $VISIBLE_COUNT != registered world size $WORLD" >&2
  exit 32
fi

EXPOSURE=${P2_EXPOSURE:-artifacts/raw_winner_p2/preparation/external73_fixed_exposure_2400.jsonl}
STEPS=${P2_MAX_STEPS:-150}
OUT_ROOT=${P2_OUT_ROOT:-outputs/raw-winner-p2-structural-probe-v1/capacity}
OUT=$OUT_ROOT/$ARM
test -s "$EXPOSURE"
test -d "$MODEL"
test ! -e "$OUT"
ACTUAL_ROWS=$(wc -l < "$EXPOSURE")
EXPECTED_ROWS=$((STEPS * WORLD * BATCH * ACCUM))
if [[ "$ACTUAL_ROWS" -ne "$EXPECTED_ROWS" ]]; then
  echo "Exposure rows $ACTUAL_ROWS != required $EXPECTED_ROWS" >&2
  exit 33
fi

COMMON=(
  train.py --model "$MODEL" --train-manifest "$EXPOSURE"
  --validation-manifest artifacts/manifests/validation.jsonl
  --project-root "$PROJECT_DIR" --output-dir "$OUT" --trial-name "$ARM"
  --max-steps "$STEPS" --batch-size "$BATCH" --eval-batch-size 1
  --gradient-accumulation-steps "$ACCUM" --learning-rate "$LR"
  --weight-decay 0.01 --scheduler cosine --warmup-ratio 0 --warmup-steps 8
  --evaluation-strategy steps --checkpoint-steps "$STEPS" --save-total-limit 2
  --logging-steps 5 --seed 42 --data-seed 42 --sampling-seed 42
  --sampling-policy fixed_exposure --generation-num-beams 2
  --generation-no-repeat-ngram-size 4 --generation-repetition-penalty 1.05
  --generation-max-length 225 --bf16 --tf32 --gradient-checkpointing
  --no-apply-spec-augment --mask-time-prob 0 --mask-feature-prob 0
)
if [[ ${P2_SMOKE:-0} == 1 ]]; then
  COMMON+=(--max-validation-samples 32 --max-train-samples "$ACTUAL_ROWS" --skip-final-evaluation)
fi
if [[ "$METHOD" == lora ]]; then
  COMMON+=(--lora --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --lora-target-scope all_attention_qv)
fi
if [[ "$METHOD" == full_fsdp ]]; then
  COMMON+=(--fsdp-full-shard)
fi

mkdir -p "$(dirname "$OUT")" artifacts/raw_winner_p2/launch
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv,noheader \
  > "artifacts/raw_winner_p2/launch/${ARM}_gpu_start.csv"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
if [[ "$WORLD" -gt 1 ]]; then
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$WORLD" "${COMMON[@]}"
else
  "$PYTHON" "${COMMON[@]}"
fi
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv,noheader \
  > "artifacts/raw_winner_p2/launch/${ARM}_gpu_end.csv"
