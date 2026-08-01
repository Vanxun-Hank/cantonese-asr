#!/usr/bin/env bash
# Serial-after-training recovery evaluator.  This preserves the pre-live
# evaluation path and must not run against the same output root as a live job.
# Optional recovery namespace overrides:
#   ADAPTIVE_EVAL_ROOT=outputs/w500-adaptive-continuation-v1/evaluation_recovery/stage40
#   ADAPTIVE_EVAL_TASKS=artifacts/w500_adaptive_continuation_v1/stage40/evaluation_tasks_recovery.json
# Submit only after the corresponding training job completed successfully.
#SBATCH --job-name=canto-w500-adapt-eval-recover
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu002
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --mem=224G
#SBATCH --time=24:00:00
#SBATCH --output=/home/bolin/cantonese-asr/logs/%x-%j.out

set -euo pipefail
source /home/bolin/cantonese-asr/slurm/common.sh

export CANTO_ALLOWED_GPU_INDICES=0,1,2,3
require_allowed_gpu_allocation
: "${ADAPTIVE_STAGE:?ADAPTIVE_STAGE must be 40, 45, or 50}"
case "$ADAPTIVE_STAGE" in 40|45|50) ;; *) echo "Invalid ADAPTIVE_STAGE=$ADAPTIVE_STAGE" >&2; exit 2;; esac

ARTIFACT=artifacts/w500_adaptive_continuation_v1
ROOT=${ADAPTIVE_TRAIN_ROOT:-outputs/w500-adaptive-continuation-v1/stage${ADAPTIVE_STAGE}}
EVAL=${ADAPTIVE_EVAL_ROOT:-outputs/w500-adaptive-continuation-v1/evaluation_recovery/stage${ADAPTIVE_STAGE}}
TASKS=${ADAPTIVE_EVAL_TASKS:-$ARTIFACT/stage${ADAPTIVE_STAGE}/evaluation_tasks_recovery.json}
LAUNCH=$ARTIFACT/stage${ADAPTIVE_STAGE}/launch/evaluation_recovery_${SLURM_JOB_ID}
VALIDATION=artifacts/manifests/validation.jsonl
PUBLIC=artifacts/datasets/official/template_pre.jsonl
OOD=artifacts/manifests/ood/v1/ood_panel.jsonl

physical_gpus="$(scontrol show job -dd "$SLURM_JOB_ID" | sed -n 's/.*GRES=gpu:[^ ]*(IDX:\([^)]*\)).*/\1/p' | head -n 1)"
if [[ "$physical_gpus" != "0,1,2,3" && "$physical_gpus" != "0-3" ]]; then
  echo "Refusing gpu002 allocation ${physical_gpus:-unknown}; expected physical GPUs 0-3" >&2
  exit 42
fi
"$PYTHON" - "$ROOT/training_completion.json" <<'PY'
import json,sys
value=json.load(open(sys.argv[1]))
if not value.get('passed'): raise SystemExit('Stage training completion did not pass')
PY
for path in "$EVAL" "$TASKS"; do
  if [[ -e "$path" ]]; then echo "Refusing to overwrite recovery evaluation artifact: $path" >&2; exit 3; fi
done
mkdir -p "$EVAL" "$LAUNCH" "$(dirname "$TASKS")"
scontrol show job -dd "$SLURM_JOB_ID" >"$LAUNCH/slurm_job_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader >"$LAUNCH/gpu_allocation_${SLURM_JOB_ID}.txt"

"$PYTHON" - "$ROOT" "$EVAL" "$TASKS" "$ADAPTIVE_STAGE" <<'PY'
import hashlib,json,sys
from pathlib import Path

root,evaluation,tasks_path=map(Path,sys.argv[1:4]); stage=sys.argv[4]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
 return h.hexdigest()
tasks=[]
for completion_path in sorted(root.glob('*/training_completion.json')):
 arm_dir=completion_path.parent; completion=json.load(open(completion_path))
 if not completion.get('passed'): raise SystemExit(f'Incomplete arm: {arm_dir}')
 run=json.load(open(arm_dir/'run_config.json')); arm=run['arm']
 for target,checkpoint_receipt in sorted(completion['checkpoints'].items(),key=lambda item:float(item[0])):
  if target=='smoke': continue
  checkpoint=Path(checkpoint_receipt['path'])
  state=json.load(open(checkpoint/'adaptive_state.json'))
  observed=sha(checkpoint/'model.safetensors')
  expected=checkpoint_receipt['model_safetensors_sha256']
  if observed!=expected: raise SystemExit(f'Checkpoint hash mismatch: {checkpoint}')
  label=f"stage{stage}/{arm_dir.name}/{checkpoint.name}"
  output=evaluation/arm_dir.name/checkpoint.name/'historical_current'
  candidate={
   'label':label,'checkpoint':str(checkpoint.resolve()),'processor_dir':str(checkpoint.resolve()),
   'model_safetensors_sha256':expected,'hours':float(state['actual_wenet_hours']),
   'cumulative_wenet_seconds':float(state['cumulative_wenet_seconds']),
   'wenet_lr':float(arm['wenet_lr']),'official_lr':float(arm['official_lr']),
   'wenet_cursor':int(state['wenet_sample_cursor']),'official_cursor':int(state['official_sample_cursor']),
   'wenet_manifest':str(Path(arm['wenet_manifest']).resolve()),
   'adaptive_state_sha256':sha(checkpoint/'adaptive_state.json'),'outputs_pre':str(output.resolve()),
  }
  tasks.append({'task_index':len(tasks),'gpu_slot':len(tasks)%4,'label':label,
                'model_dir':str(checkpoint.resolve()),'processor_dir':str(checkpoint.resolve()),
                'weight_sha256':expected,'output_dir':str(output.resolve()),'candidate':candidate})
if len(tasks)!=16: raise SystemExit(f'Expected 16 completed stage checkpoints, found {len(tasks)}')
tasks_path.write_text(json.dumps({'stage':stage,'mode':'serial_after_training_recovery','tasks':tasks},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'stage':stage,'task_count':len(tasks),'mode':'serial_after_training_recovery'},indent=2))
PY

pids=()
for gpu in 0 1 2 3; do
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    "$PYTHON" - "$TASKS" "$gpu" <<'PY' | while IFS=$'\t' read -r model processor weight output candidate_json; do
import json,sys
for task in json.load(open(sys.argv[1]))['tasks']:
 if task['gpu_slot']==int(sys.argv[2]):
  print('\t'.join((task['model_dir'],task['processor_dir'],task['weight_sha256'],task['output_dir'],json.dumps(task['candidate'],ensure_ascii=False,separators=(',',':')))))
PY
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
      "$PYTHON" - "$output/candidate.json" "$candidate_json" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1]); value=json.loads(sys.argv[2])
path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
PY
    done
  ) >"$LAUNCH/gpu${gpu}_${SLURM_JOB_ID}.log" 2>&1 &
  pids+=("$!")
done
failures=0
for gpu in 0 1 2 3; do
  if wait "${pids[$gpu]}"; then
    printf 'COMPLETED\n' >"$LAUNCH/gpu${gpu}_${SLURM_JOB_ID}.status"
  else
    status=$?
    printf 'FAILED exit=%s\n' "$status" >"$LAUNCH/gpu${gpu}_${SLURM_JOB_ID}.status"
    failures=$((failures + 1))
  fi
done
if (( failures > 0 )); then echo "$failures recovery evaluation worker(s) failed" >&2; exit 10; fi

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
     'mode':'serial_after_training_recovery','tasks':rows}
path=Path(sys.argv[2])/'evaluation_completion.json'; path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2)); raise SystemExit(0 if out['passed'] else 11)
PY
