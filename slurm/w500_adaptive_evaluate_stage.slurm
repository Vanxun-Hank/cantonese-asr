#!/usr/bin/env bash
# Live evaluation half of the two-node adaptive stage pipeline.
# gpu001 runs four independent single-GPU training arms; this job binds the
# matching four arms to gpu002 GPUs 0-3 and evaluates each atomic checkpoint as
# soon as it appears. Submit with ADAPTIVE_STAGE and ADAPTIVE_ARM_FILE.
#SBATCH --job-name=canto-w500-adapt-eval
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu002
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --mem=224G
#SBATCH --time=36:00:00
#SBATCH --output=/home/bolin/cantonese-asr/logs/%x-%j.out

set -euo pipefail
source /home/bolin/cantonese-asr/slurm/common.sh

export CANTO_ALLOWED_GPU_INDICES=0,1,2,3
require_allowed_gpu_allocation
: "${ADAPTIVE_STAGE:?ADAPTIVE_STAGE must be 40, 45, or 50}"
: "${ADAPTIVE_ARM_FILE:?ADAPTIVE_ARM_FILE is required for live evaluation}"
case "$ADAPTIVE_STAGE" in 40|45|50) ;; *) echo "Invalid ADAPTIVE_STAGE=$ADAPTIVE_STAGE" >&2; exit 2;; esac
POLL_SECONDS=${ADAPTIVE_EVAL_POLL_SECONDS:-15}
if [[ ! "$POLL_SECONDS" =~ ^[0-9]+$ ]] || (( POLL_SECONDS < 5 || POLL_SECONDS > 60 )); then
  echo "ADAPTIVE_EVAL_POLL_SECONDS must be an integer in [5,60]" >&2
  exit 2
fi

ARTIFACT=artifacts/w500_adaptive_continuation_v1
ARMS=$ADAPTIVE_ARM_FILE
ROOT=${ADAPTIVE_TRAIN_ROOT:-outputs/w500-adaptive-continuation-v1/stage${ADAPTIVE_STAGE}}
EVAL=${ADAPTIVE_EVAL_ROOT:-outputs/w500-adaptive-continuation-v1/evaluation/stage${ADAPTIVE_STAGE}}
TASKS=$ARTIFACT/stage${ADAPTIVE_STAGE}/evaluation_tasks.json
LAUNCH=$ARTIFACT/stage${ADAPTIVE_STAGE}/launch/evaluation
VALIDATION=artifacts/manifests/validation.jsonl
PUBLIC=artifacts/datasets/official/template_pre.jsonl
OOD=artifacts/manifests/ood/v1/ood_panel.jsonl

physical_gpus="$(scontrol show job -dd "$SLURM_JOB_ID" | sed -n 's/.*GRES=gpu:[^ ]*(IDX:\([^)]*\)).*/\1/p' | head -n 1)"
if [[ "$physical_gpus" != "0,1,2,3" && "$physical_gpus" != "0-3" ]]; then
  echo "Refusing gpu002 allocation ${physical_gpus:-unknown}; expected physical GPUs 0-3" >&2
  exit 42
fi
for path in "$EVAL" "$TASKS"; do
  if [[ -e "$path" ]]; then echo "Refusing to overwrite evaluation artifact: $path" >&2; exit 3; fi
done
if [[ ! -f "$ARMS" ]]; then
  echo "Missing adaptive arm file: $ARMS" >&2
  exit 3
fi
mkdir -p "$EVAL" "$LAUNCH" "$(dirname "$TASKS")"
scontrol show job -dd "$SLURM_JOB_ID" >"$LAUNCH/slurm_job_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader >"$LAUNCH/gpu_allocation_${SLURM_JOB_ID}.txt"

mapfile -t ARM_NAMES < <("$PYTHON" - "$ARMS" "$ADAPTIVE_STAGE" <<'PY'
import json,sys
arms=json.load(open(sys.argv[1]))['arms']; stage=float(sys.argv[2])
if len(arms)!=4: raise SystemExit(f'Expected four stage arms, got {len(arms)}')
for arm in arms:
 name=str(arm['name'])
 if '/' in name or name in {'.','..'}: raise SystemExit(f'Unsafe arm name: {name}')
 if float(arm['target_hours']) != stage: raise SystemExit(f'Arm {name} target does not match stage {stage:g}')
 print(name)
PY
)

# Exit a watcher promptly after the training side has recorded an impossible
# prerequisite. A healthy stage completion is also treated as proof that a
# still-missing expected artifact can never arrive.
wait_for_training_artifact() {
  local expected=$1
  while [[ ! -e "$expected" ]]; do
    if [[ -f "$ROOT/training_completion.json" ]]; then
      "$PYTHON" - "$ROOT/training_completion.json" "$expected" <<'PY'
import json,sys
value=json.load(open(sys.argv[1]))
state='passed' if value.get('passed') else 'failed'
raise SystemExit(f'Training stage {state} without required artifact: {sys.argv[2]}')
PY
    fi
    sleep "$POLL_SECONDS"
  done
}
export -f wait_for_training_artifact

pids=()
for gpu in 0 1 2 3; do
  arm=${ARM_NAMES[$gpu]}
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    run_config="$ROOT/$arm/run_config.json"
    wait_for_training_artifact "$run_config"
    mapfile -t CHECKPOINT_NAMES < <("$PYTHON" - "$run_config" "$arm" <<'PY'
import json,sys
run=json.load(open(sys.argv[1])); expected=sys.argv[2]
if run['arm']['name'] != expected: raise SystemExit(f'Run-config arm mismatch: {run["arm"]["name"]} != {expected}')
boundaries=run.get('milestone_boundaries',{})
if not boundaries: raise SystemExit(f'No future milestones for {expected}')
for target in sorted(map(float,boundaries)):
 print(f'checkpoint-wenet-{target:g}h')
PY
    )
    for checkpoint_name in "${CHECKPOINT_NAMES[@]}"; do
      checkpoint="$ROOT/$arm/$checkpoint_name"
      # Trainer builds checkpoint-name.staging and atomically renames it only
      # after model, optimizer, state, source log and hashes are complete.
      wait_for_training_artifact "$checkpoint/sha256sums.json"
      task_receipt="$LAUNCH/task_gpu${gpu}_${checkpoint_name}_${SLURM_JOB_ID}.json"
      "$PYTHON" - "$checkpoint" "$run_config" "$ADAPTIVE_STAGE" "$EVAL" "$gpu" "$task_receipt" <<'PY'
import hashlib,json,sys
from pathlib import Path
checkpoint,run_path=map(Path,sys.argv[1:3]); stage=sys.argv[3]; evaluation=Path(sys.argv[4]); gpu=int(sys.argv[5]); task_path=Path(sys.argv[6])
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
 return h.hexdigest()
required=('model.safetensors','adaptive_state.json','optimizer.pt','rng_state.pth','source_steps.jsonl','trainer_state.json')
hashes=json.load(open(checkpoint/'sha256sums.json'))
missing=[name for name in required if not (checkpoint/name).is_file() or hashes.get(name)!=sha(checkpoint/name)]
if missing: raise SystemExit(f'Atomic checkpoint receipt failed for {checkpoint}: {missing}')
state=json.load(open(checkpoint/'adaptive_state.json')); run=json.load(open(run_path)); arm=run['arm']
if state.get('checkpoint_phase')!='after_complete_wenet_official_cycle' or state.get('next_source')!='wenet':
 raise SystemExit(f'Checkpoint is not at a complete source-cycle boundary: {checkpoint}')
if state.get('identity') != run.get('identity'):
 raise SystemExit(f'Checkpoint/run identity mismatch: {checkpoint}')
weight=sha(checkpoint/'model.safetensors'); label=f'stage{stage}/{arm["name"]}/{checkpoint.name}'
output=evaluation/arm['name']/checkpoint.name/'historical_current'
candidate={
 'label':label,'checkpoint':str(checkpoint.resolve()),'processor_dir':str(checkpoint.resolve()),
 'model_safetensors_sha256':weight,'hours':float(state['actual_wenet_hours']),
 'cumulative_wenet_seconds':float(state['cumulative_wenet_seconds']),
 'wenet_lr':float(arm['wenet_lr']),'official_lr':float(arm['official_lr']),
 'wenet_cursor':int(state['wenet_sample_cursor']),'official_cursor':int(state['official_sample_cursor']),
 'wenet_manifest':str(Path(arm['wenet_manifest']).resolve()),
 'adaptive_state_sha256':sha(checkpoint/'adaptive_state.json'),
 'outputs_pre':str(output.resolve()),
}
task={'gpu_slot':gpu,'label':label,'model_dir':str(checkpoint.resolve()),'processor_dir':str(checkpoint.resolve()),
      'weight_sha256':weight,'output_dir':str(output.resolve()),'candidate':candidate}
task_path.write_text(json.dumps(task,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'checkpoint':str(checkpoint),'output':str(output),'weight':weight},indent=2))
PY
      mapfile -t TASK_FIELDS < <("$PYTHON" - "$task_receipt" <<'PY'
import json,sys
task=json.load(open(sys.argv[1]))
for key in ('model_dir','processor_dir','weight_sha256','output_dir'): print(task[key])
PY
      )
      model=${TASK_FIELDS[0]}
      processor=${TASK_FIELDS[1]}
      weight=${TASK_FIELDS[2]}
      output=${TASK_FIELDS[3]}
      "$PYTHON" scripts/evaluate_exp004_decoding_arm.py \
        --arm D_B2_NR4_RP \
        --model-dir "$model" \
        --processor-dir "$processor" \
        --project-root "$PROJECT_DIR" \
        --validation-manifest "$VALIDATION" \
        --public-manifest "$PUBLIC" \
        --ood-manifest "$OOD" \
        --output-dir "$output" \
        --expected-weight-sha256 "$weight" \
        --batch-size 4 \
        --generation-max-length 225
      "$PYTHON" - "$task_receipt" "$output/candidate.json" <<'PY'
import json,sys
from pathlib import Path
task=json.load(open(sys.argv[1])); path=Path(sys.argv[2])
path.write_text(json.dumps(task['candidate'],ensure_ascii=False,indent=2)+'\n')
PY
    done
  ) >"$LAUNCH/gpu${gpu}_${arm}_${SLURM_JOB_ID}.log" 2>&1 &
  pids+=("$!")
done

failures=0
for gpu in 0 1 2 3; do
  arm=${ARM_NAMES[$gpu]}
  if wait "${pids[$gpu]}"; then
    printf 'COMPLETED\n' >"$LAUNCH/gpu${gpu}_${arm}_${SLURM_JOB_ID}.status"
  else
    status=$?
    printf 'FAILED exit=%s\n' "$status" >"$LAUNCH/gpu${gpu}_${arm}_${SLURM_JOB_ID}.status"
    failures=$((failures + 1))
  fi
done
if (( failures > 0 )); then echo "$failures live evaluation worker(s) failed" >&2; exit 10; fi

wait_for_training_artifact "$ROOT/training_completion.json"
"$PYTHON" - "$ROOT/training_completion.json" <<'PY'
import json,sys
value=json.load(open(sys.argv[1]))
if not value.get('passed'): raise SystemExit('Stage training completion did not pass')
PY

"$PYTHON" - "$LAUNCH" "$TASKS" "$ADAPTIVE_STAGE" <<'PY'
import json,sys
from pathlib import Path
launch,tasks_path=map(Path,sys.argv[1:3]); stage=sys.argv[3]
tasks=[json.load(open(path)) for path in launch.glob('task_gpu*_checkpoint-*.json')]
tasks.sort(key=lambda row:(int(row['gpu_slot']),float(json.load(open(Path(row['model_dir'])/'adaptive_state.json'))['actual_wenet_hours'])))
if len(tasks) != 16: raise SystemExit(f'Live evaluator expected 16 checkpoint tasks, found {len(tasks)}')
for index,task in enumerate(tasks): task['task_index']=index
tasks_path.write_text(json.dumps({'stage':stage,'mode':'gpu002_live_checkpoint_pipeline','tasks':tasks},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'stage':stage,'task_count':len(tasks)},indent=2))
PY

"$PYTHON" - "$TASKS" "$EVAL" <<'PY'
import json,sys
from pathlib import Path
tasks=json.load(open(sys.argv[1]))['tasks']; expected={'validation':702,'public':1900,'ood':2000}; rows={}
required=('metrics.json','top_confusions.csv','error_examples.json','generation.json','predictions.jsonl','generation_tokens.jsonl')
for task in tasks:
 root=Path(task['output_dir']); surfaces={}
 for name,count in expected.items():
  path=root/name; counts={}
  for file in ('predictions.jsonl','generation_tokens.jsonl'):
   counts[file]=sum(1 for line in (path/file).open(encoding='utf-8') if line.strip()) if (path/file).is_file() else None
  surfaces[name]={'passed':all((path/file).is_file() for file in required) and all(value==count for value in counts.values()),'counts':counts}
 rows[task['label']]={'passed':all(value['passed'] for value in surfaces.values()),'surfaces':surfaces,'output':str(root)}
out={'passed':bool(rows) and all(value['passed'] for value in rows.values()),
     'mode':'gpu002_live_checkpoint_pipeline','tasks':rows}
path=Path(sys.argv[2])/'evaluation_completion.json'; path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2)); raise SystemExit(0 if out['passed'] else 11)
PY
