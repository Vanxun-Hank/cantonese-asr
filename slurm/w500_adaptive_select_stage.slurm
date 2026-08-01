#!/usr/bin/env bash
# CPU-only fixed-validation selector.  Public/OOD are catastrophic vetoes only.
#SBATCH --job-name=canto-w500-adapt-select
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu001
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/bolin/cantonese-asr/logs/%x-%j.out

set -euo pipefail
source /home/bolin/cantonese-asr/slurm/common.sh
: "${ADAPTIVE_STAGE:?ADAPTIVE_STAGE must be 40, 45, or 50}"
case "$ADAPTIVE_STAGE" in 40|45|50) ;; *) echo "Invalid ADAPTIVE_STAGE=$ADAPTIVE_STAGE" >&2; exit 2;; esac

CONFIG=configs/rounds/w500_adaptive_continuation.json
ARTIFACT=artifacts/w500_adaptive_continuation_v1
if [[ -n "${ADAPTIVE_SELECTION_EVAL_ROOT:-}" ]]; then
  EVAL=$ADAPTIVE_SELECTION_EVAL_ROOT
elif [[ "$ADAPTIVE_STAGE" == 50 ]]; then
  # The final acoustic winner may be any eligible 36.25--50h checkpoint.
  EVAL=outputs/w500-adaptive-continuation-v1/evaluation
else
  EVAL=outputs/w500-adaptive-continuation-v1/evaluation/stage${ADAPTIVE_STAGE}
fi
SELECTION=$ARTIFACT/stage${ADAPTIVE_STAGE}/selection.json
LAUNCH=$ARTIFACT/stage${ADAPTIVE_STAGE}/launch/selection
mkdir -p "$LAUNCH"
scontrol show job -dd "$SLURM_JOB_ID" >"$LAUNCH/slurm_job_${SLURM_JOB_ID}.txt"

TRAIN_ROOT=${ADAPTIVE_TRAIN_ROOT:-outputs/w500-adaptive-continuation-v1/stage${ADAPTIVE_STAGE}}
STAGE_EVAL_ROOT=${ADAPTIVE_STAGE_EVAL_ROOT:-outputs/w500-adaptive-continuation-v1/evaluation/stage${ADAPTIVE_STAGE}}
training_completion=$TRAIN_ROOT/training_completion.json
evaluation_completion=$STAGE_EVAL_ROOT/evaluation_completion.json
"$PYTHON" - "$training_completion" "$evaluation_completion" <<'PY'
import json,sys
for role,path in zip(('training','evaluation'),sys.argv[1:]):
 value=json.load(open(path))
 if not value.get('passed'): raise SystemExit(f'Failed {role} completion: {path}')
PY

args=(
  --config "$CONFIG"
  --evaluation-root "$EVAL"
  --target-hours "$ADAPTIVE_STAGE"
  --selection-output "$SELECTION"
)
case "$ADAPTIVE_STAGE" in
  40)
    args+=(
      --balanced-next-manifest "$ARTIFACT/manifests/wenet_balanced_after35_to_45p0h.jsonl"
      --next-arms-output "$ARTIFACT/stage45_arms.json"
    )
    ;;
  45)
    args+=(
      --balanced-next-manifest "$ARTIFACT/manifests/wenet_balanced_after35_to_50p0h.jsonl"
      --next-arms-output "$ARTIFACT/stage50_arms.json"
    )
    ;;
esac
"$PYTHON" scripts/select_w500_adaptive_stage.py "${args[@]}" \
  >"$LAUNCH/select_${SLURM_JOB_ID}.log" 2>&1

"$PYTHON" - "$SELECTION" "$ADAPTIVE_STAGE" <<'PY'
import json,sys
value=json.load(open(sys.argv[1])); stage=float(sys.argv[2])
if value.get('ranking_surface')!='fixed_validation_only' or value.get('public_ood_used_for_ranking') is not False:
 raise SystemExit('Selection surface receipt failed')
if stage < 50 and len(value.get('branches',[])) != 4:
 raise SystemExit('Next stage did not produce four branches')
if stage == 50 and not value.get('selected',{}).get('winner'):
 raise SystemExit('Stage 50 did not select the global acoustic winner')
print(json.dumps({'passed':True,'stage':stage,'ranking':value.get('ranking'),'branches':len(value.get('branches',[]))},indent=2))
PY
printf 'COMPLETED\n' >"$LAUNCH/status_${SLURM_JOB_ID}.txt"
