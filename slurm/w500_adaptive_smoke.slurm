#!/usr/bin/env bash
# Four independent single-GPU two-cycle topology/resume smokes.
#SBATCH --job-name=canto-w500-adapt-smoke
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu001
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --output=/home/bolin/cantonese-asr/logs/%x-%j.out

set -euo pipefail
source /home/bolin/cantonese-asr/slurm/common.sh

export CANTO_ALLOWED_GPU_INDICES=0,1,2,3
require_allowed_gpu_allocation

CONFIG=configs/rounds/w500_adaptive_continuation.json
ARTIFACT=artifacts/w500_adaptive_continuation_v1
PREPARATION=$ARTIFACT/preparation.json
ARMS=$ARTIFACT/stage40_arms.json
SMOKE=$ARTIFACT/smoke/v1
LAUNCH=$ARTIFACT/launch/smoke

physical_gpus="$(scontrol show job -dd "$SLURM_JOB_ID" | sed -n 's/.*GRES=gpu:[^ ]*(IDX:\([^)]*\)).*/\1/p' | head -n 1)"
if [[ "$physical_gpus" != "0,1,2,3" && "$physical_gpus" != "0-3" ]]; then
  echo "Refusing allocation ${physical_gpus:-unknown}; expected physical GPUs 0-3" >&2
  exit 42
fi
if [[ ! -f "$ARTIFACT/MANIFESTS_FROZEN" || "$(cat "$ARTIFACT/MANIFESTS_FROZEN")" != PASS ]]; then
  echo "Frozen preparation receipt is missing" >&2
  exit 2
fi
if [[ -e "$SMOKE" ]]; then
  echo "Refusing to overwrite smoke output: $SMOKE" >&2
  exit 3
fi
mkdir -p "$SMOKE" "$LAUNCH"
scontrol show job -dd "$SLURM_JOB_ID" >"$LAUNCH/slurm_job_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader >"$LAUNCH/gpu_allocation_${SLURM_JOB_ID}.txt"

mapfile -t ARM_NAMES < <("$PYTHON" - "$ARMS" <<'PY'
import json,sys
arms=json.load(open(sys.argv[1]))['arms']
if len(arms)!=4: raise SystemExit(f'Expected four smoke arms, got {len(arms)}')
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
      --output-dir "$SMOKE/$arm" \
      --smoke-cycles 2 \
      --num-workers 2
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
if (( failures > 0 )); then
  echo "$failures adaptive smoke arm(s) failed" >&2
  exit 10
fi

"$PYTHON" - "$SMOKE" "${ARM_NAMES[@]}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); rows={}
for arm in sys.argv[2:]:
    path=root/arm/'training_completion.json'
    value=json.load(open(path)) if path.is_file() else {}
    audits=value.get('gradient_audits',{})
    rows[arm]={
        'passed': bool(value.get('passed'))
            and bool(audits.get('wenet_encoder_only'))
            and bool(audits.get('post_official_wenet_decoder_and_moments_unchanged'))
            and bool(audits.get('official_full_sft')),
        'completion':str(path.resolve()),
        'gradient_audits':audits,
    }
out={'passed':all(row['passed'] for row in rows.values()),'arms':rows}
(root/'smoke_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
raise SystemExit(0 if out['passed'] else 11)
PY
