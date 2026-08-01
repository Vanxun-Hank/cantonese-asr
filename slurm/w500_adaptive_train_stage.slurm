#!/usr/bin/env bash
# Generic four-arm adaptive continuation stage.  Submit with:
#   sbatch --export=ALL,ADAPTIVE_STAGE=40,ADAPTIVE_ARM_FILE=... this.slurm
#SBATCH --job-name=canto-w500-adapt-train
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu001
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=20:00:00
#SBATCH --output=/home/bolin/cantonese-asr/logs/%x-%j.out

set -euo pipefail
source /home/bolin/cantonese-asr/slurm/common.sh

export CANTO_ALLOWED_GPU_INDICES=0,1,2,3
require_allowed_gpu_allocation

: "${ADAPTIVE_STAGE:?ADAPTIVE_STAGE must be 40, 45, or 50}"
: "${ADAPTIVE_ARM_FILE:?ADAPTIVE_ARM_FILE is required}"
case "$ADAPTIVE_STAGE" in 40|45|50) ;; *) echo "Invalid ADAPTIVE_STAGE=$ADAPTIVE_STAGE" >&2; exit 2;; esac

CONFIG=configs/rounds/w500_adaptive_continuation.json
ARTIFACT=artifacts/w500_adaptive_continuation_v1
PREPARATION=$ARTIFACT/preparation.json
SMOKE=$ARTIFACT/smoke/v1/smoke_summary.json
ARMS=$ADAPTIVE_ARM_FILE
ROOT=${ADAPTIVE_OUTPUT_ROOT:-outputs/w500-adaptive-continuation-v1/stage${ADAPTIVE_STAGE}}
LAUNCH=$ARTIFACT/stage${ADAPTIVE_STAGE}/launch/training

physical_gpus="$(scontrol show job -dd "$SLURM_JOB_ID" | sed -n 's/.*GRES=gpu:[^ ]*(IDX:\([^)]*\)).*/\1/p' | head -n 1)"
if [[ "$physical_gpus" != "0,1,2,3" && "$physical_gpus" != "0-3" ]]; then
  echo "Refusing allocation ${physical_gpus:-unknown}; expected physical GPUs 0-3" >&2
  exit 42
fi
"$PYTHON" - "$PREPARATION" "$SMOKE" <<'PY'
import json,sys
for path in sys.argv[1:]:
 value=json.load(open(path))
 if not value.get('passed'): raise SystemExit(f'Failed prerequisite: {path}')
PY
if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite training root: $ROOT" >&2
  exit 3
fi
mkdir -p "$ROOT" "$LAUNCH"
scontrol show job -dd "$SLURM_JOB_ID" >"$LAUNCH/slurm_job_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader >"$LAUNCH/gpu_allocation_${SLURM_JOB_ID}.txt"
(
  while true; do
    printf '%s,' "$(date --iso-8601=seconds)"
    nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits | tr '\n' ';'
    printf '\n'
    sleep 20
  done
) >"$LAUNCH/gpu_memory_${SLURM_JOB_ID}.csv" &
monitor_pid=$!
cleanup() { kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true; }
trap cleanup EXIT

mapfile -t ARM_NAMES < <("$PYTHON" - "$ARMS" <<'PY'
import json,sys
arms=json.load(open(sys.argv[1]))['arms']
if len(arms)!=4: raise SystemExit(f'Expected four stage arms, got {len(arms)}')
for arm in arms:
 name=str(arm['name'])
 if '/' in name or name in {'.','..'}: raise SystemExit(f'Unsafe arm name: {name}')
 print(name)
PY
)
pids=()
for gpu in 0 1 2 3; do
  arm=${ARM_NAMES[$gpu]}
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    "$PYTHON" scripts/train_w500_adaptive_continuation.py \
      --project-root "$PROJECT_DIR" \
      --config "$CONFIG" \
      --preparation "$PREPARATION" \
      --arm-file "$ARMS" \
      --arm-index "$gpu" \
      --output-dir "$ROOT/$arm" \
      --num-workers 4
  ) >"$LAUNCH/${arm}_${SLURM_JOB_ID}.log" 2>&1 &
  pids+=("$!")
done
failures=0
for gpu in 0 1 2 3; do
  arm=${ARM_NAMES[$gpu]}
  if wait "${pids[$gpu]}"; then
    printf 'COMPLETED\n' >"$LAUNCH/${arm}_${SLURM_JOB_ID}.status"
  else
    status=$?
    printf 'FAILED exit=%s\n' "$status" >"$LAUNCH/${arm}_${SLURM_JOB_ID}.status"
    failures=$((failures + 1))
  fi
done

"$PYTHON" - "$ROOT" "${ARM_NAMES[@]}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); rows={}
for arm in sys.argv[2:]:
 path=root/arm/'training_completion.json'
 value=json.load(open(path)) if path.is_file() else None
 rows[arm]={'passed':bool(value and value.get('passed')),'completion':str(path.resolve()),'value':value}
out={'passed':all(row['passed'] for row in rows.values()),'arms':rows}
(root/'training_completion.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
if not out['passed']: raise SystemExit(11)
PY
if (( failures > 0 )); then
  echo "$failures adaptive training arm(s) failed" >&2
  exit 10
fi
