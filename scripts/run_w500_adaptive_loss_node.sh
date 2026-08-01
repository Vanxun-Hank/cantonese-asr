#!/usr/bin/env bash
# Node-local worker launcher for the two-node adaptive preparation recovery.
# Required paths and the eligible-manifest SHA are supplied by the parent
# Slurm job through exported environment variables.

set -euo pipefail
source /home/bolin/cantonese-asr/slurm/common.sh

required=(
  PROJECT_DIR PYTHON START PROCESSOR DERIVED_SHARDS RECOVERY
  EXPECTED_ELIGIBLE_SHA
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required recovery environment variable: $name" >&2
    exit 2
  fi
done

node=$(hostname -s)
case "$node" in
  gpu001) worker_offset=0 ;;
  gpu002) worker_offset=4 ;;
  *) echo "Unexpected recovery node: $node" >&2; exit 42 ;;
esac
visible_count="$($PYTHON - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if [[ "$visible_count" != 4 ]]; then
  echo "$node sees $visible_count CUDA devices; expected 4" >&2
  exit 42
fi
mapfile -t physical_indices < <(
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits | sed 's/[[:space:]]//g'
)
if [[ "${physical_indices[*]}" != "0 1 2 3" ]]; then
  echo "$node physical GPU indices are ${physical_indices[*]}; expected 0 1 2 3" >&2
  exit 42
fi
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader \
  >"$RECOVERY/${node}_gpu_allocation.txt"

pids=()
worker_indices=()
for local_gpu in 0 1 2 3; do
  worker_index=$((worker_offset + local_gpu))
  worker_indices+=("$worker_index")
  (
    export CUDA_VISIBLE_DEVICES=$local_gpu
    "$PYTHON" scripts/evaluate_w500_candidate_losses.py \
      --model-dir "$START" \
      --processor-dir "$PROCESSOR" \
      --manifest "$DERIVED_SHARDS/shard_${worker_index}.jsonl" \
      --eligible-manifest-sha256 "$EXPECTED_ELIGIBLE_SHA" \
      --project-root "$PROJECT_DIR" \
      --output "$RECOVERY/losses_${worker_index}.jsonl" \
      --receipt "$RECOVERY/losses_${worker_index}.receipt.json" \
      --batch-size 8 \
      --num-workers 6 \
      --dtype bfloat16
  ) >"$RECOVERY/worker_${worker_index}_${node}.log" 2>&1 &
  pids+=("$!")
done

failures=0
for position in 0 1 2 3; do
  worker_index=${worker_indices[$position]}
  if wait "${pids[$position]}"; then
    printf 'COMPLETED\n' >"$RECOVERY/worker_${worker_index}.status"
  else
    status=$?
    printf 'FAILED exit=%s\n' "$status" >"$RECOVERY/worker_${worker_index}.status"
    failures=$((failures + 1))
  fi
done
if (( failures > 0 )); then
  echo "$node had $failures candidate-loss worker failure(s)" >&2
  exit 10
fi
