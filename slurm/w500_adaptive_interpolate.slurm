#!/usr/bin/env bash
# Build MIX15/30/45/60 from the global acoustic winner and evaluate raw+mixes.
#SBATCH --job-name=canto-w500-adapt-mix
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu001
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --mem=224G
#SBATCH --time=16:00:00
#SBATCH --output=/home/bolin/cantonese-asr/logs/%x-%j.out

set -euo pipefail
source /home/bolin/cantonese-asr/slurm/common.sh
export CANTO_ALLOWED_GPU_INDICES=0,1,2,3
require_allowed_gpu_allocation

CONFIG=configs/rounds/w500_adaptive_continuation.json
ARTIFACT=artifacts/w500_adaptive_continuation_v1
SELECTION=$ARTIFACT/stage50/selection.json
ROOT=outputs/w500-adaptive-continuation-v1/interpolation
EVAL=outputs/w500-adaptive-continuation-v1/interpolation_evaluation
INTERP=$ARTIFACT/interpolation
TASKS=$INTERP/evaluation_tasks.json
FINAL_CANDIDATES=$INTERP/final_candidates.json
INTERP_SELECTION=$INTERP/selection.json
LAUNCH=$INTERP/launch
STABLE=outputs/w500-encoder-decoder-curriculum-v1/stage_a/A_HYBRID_DONLY_LR5E7/checkpoint-5

physical_gpus="$(scontrol show job -dd "$SLURM_JOB_ID" | sed -n 's/.*GRES=gpu:[^ ]*(IDX:\([^)]*\)).*/\1/p' | head -n 1)"
if [[ "$physical_gpus" != "0,1,2,3" && "$physical_gpus" != "0-3" ]]; then
  echo "Refusing allocation ${physical_gpus:-unknown}; expected physical GPUs 0-3" >&2
  exit 42
fi
for path in "$ROOT" "$EVAL" "$INTERP"; do
  if [[ -e "$path" ]]; then echo "Refusing to overwrite interpolation artifact: $path" >&2; exit 3; fi
done
mkdir -p "$ROOT" "$EVAL" "$INTERP" "$LAUNCH"
scontrol show job -dd "$SLURM_JOB_ID" >"$LAUNCH/slurm_job_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader >"$LAUNCH/gpu_allocation_${SLURM_JOB_ID}.txt"

mapfile -t WINNER < <("$PYTHON" - "$SELECTION" <<'PY'
import json,sys
row=json.load(open(sys.argv[1]))['selected']['winner']
for value in (row['checkpoint'],row['model_safetensors_sha256'],row['outputs_pre']): print(value)
PY
)
ACOUSTIC=${WINNER[0]}
ACOUSTIC_SHA=${WINNER[1]}
RAW_OUTPUTS=${WINNER[2]}
test "$(sha256sum "$ACOUSTIC/model.safetensors" | awk '{print $1}')" = "$ACOUSTIC_SHA"

NAMES=(MIX15 MIX30 MIX45 MIX60)
WEIGHTS=(0.15 0.30 0.45 0.60)
pids=()
for gpu in 0 1 2 3; do
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    "$PYTHON" scripts/interpolate_w500_decoder.py \
      --acoustic-checkpoint "$ACOUSTIC" \
      --stable-decoder-checkpoint "$STABLE" \
      --name "${NAMES[$gpu]}" \
      --stable-weight "${WEIGHTS[$gpu]}" \
      --output-dir "$ROOT/${NAMES[$gpu]}" \
      --config "$CONFIG" \
      --project-root "$PROJECT_DIR"
  ) >"$LAUNCH/interpolate_${NAMES[$gpu]}_${SLURM_JOB_ID}.log" 2>&1 &
  pids+=("$!")
done
failures=0
for gpu in 0 1 2 3; do wait "${pids[$gpu]}" || failures=$((failures+1)); done
if (( failures > 0 )); then echo "$failures interpolation worker(s) failed" >&2; exit 10; fi

"$PYTHON" - "$SELECTION" "$ROOT" "$EVAL" "$TASKS" <<'PY'
import hashlib,json,sys
from pathlib import Path
selection=json.load(open(sys.argv[1])); root=Path(sys.argv[2]); evaluation=Path(sys.argv[3]); tasks_path=Path(sys.argv[4])
winner=selection['selected']['winner']
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
tasks=[]
for gpu,name in enumerate(('MIX15','MIX30','MIX45','MIX60')):
 checkpoint=root/name; receipt=json.load(open(checkpoint/'interpolation_receipt.json'))
 weight=sha(checkpoint/'model.safetensors')
 if not receipt.get('passed') or receipt.get('output_model_sha256')!=weight:
  raise SystemExit(f'Interpolation receipt failed: {name}')
 candidate={
  'label':name,'checkpoint':str(checkpoint.resolve()),'processor_dir':str(checkpoint.resolve()),
  'model_safetensors_sha256':weight,'hours':winner['hours'],
  'cumulative_wenet_seconds':winner.get('cumulative_wenet_seconds',winner['hours']*3600),
  'wenet_lr':winner['wenet_lr'],'official_lr':winner['official_lr'],
  'wenet_cursor':winner['wenet_cursor'],'official_cursor':winner['official_cursor'],
  'wenet_manifest':winner.get('wenet_manifest'),'interpolation_receipt':str((checkpoint/'interpolation_receipt.json').resolve()),
 }
 tasks.append({'gpu_slot':gpu,'name':name,'model_dir':str(checkpoint.resolve()),'processor_dir':str(checkpoint.resolve()),
               'weight_sha256':weight,'output_dir':str((evaluation/name/'historical_current').resolve()),'candidate':candidate})
tasks_path.write_text(json.dumps({'tasks':tasks},ensure_ascii=False,indent=2)+'\n')
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
        --arm D_B2_NR4_RP --model-dir "$model" --processor-dir "$processor" \
        --project-root "$PROJECT_DIR" --validation-manifest artifacts/manifests/validation.jsonl \
        --public-manifest artifacts/datasets/official/template_pre.jsonl \
        --ood-manifest artifacts/manifests/ood/v1/ood_panel.jsonl \
        --output-dir "$output" --expected-weight-sha256 "$weight" --batch-size 4 \
        --generation-max-length 225
      "$PYTHON" - "$output/candidate.json" "$candidate_json" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(json.loads(sys.argv[2]),ensure_ascii=False,indent=2)+'\n')
PY
    done
  ) >"$LAUNCH/eval_gpu${gpu}_${SLURM_JOB_ID}.log" 2>&1 &
  pids+=("$!")
done
failures=0
for gpu in 0 1 2 3; do wait "${pids[$gpu]}" || failures=$((failures+1)); done
if (( failures > 0 )); then echo "$failures interpolation evaluation worker(s) failed" >&2; exit 11; fi

# Copy the already-complete raw-winner diagnostics into this immutable final
# comparison root; no model inference is repeated and no checkpoint is changed.
mkdir -p "$EVAL/RAW_WINNER"
cp -a "$RAW_OUTPUTS" "$EVAL/RAW_WINNER/historical_current"
"$PYTHON" - "$SELECTION" "$EVAL/RAW_WINNER/historical_current/candidate.json" <<'PY'
import json,sys
from pathlib import Path
winner=json.load(open(sys.argv[1]))['selected']['winner']; winner=dict(winner); winner['label']='RAW_WINNER'
Path(sys.argv[2]).write_text(json.dumps(winner,ensure_ascii=False,indent=2)+'\n')
PY

"$PYTHON" scripts/select_w500_adaptive_stage.py \
  --config "$CONFIG" \
  --evaluation-root "$EVAL" \
  --target-hours 50 \
  --selection-output "$INTERP_SELECTION" \
  >"$LAUNCH/select_${SLURM_JOB_ID}.log" 2>&1

"$PYTHON" - "$INTERP_SELECTION" "$FINAL_CANDIDATES" <<'PY'
import json,sys
from pathlib import Path
selection=json.load(open(sys.argv[1])); rows=selection['candidates']
expected={'RAW_WINNER','MIX15','MIX30','MIX45','MIX60'}
if {row['label'] for row in rows} != expected: raise SystemExit('Final five-candidate set mismatch')
value={'schema_version':1,'ranking_surface':'fixed_validation_only','public_ood_used_for_ranking':False,
       'candidates':rows,'ranking':selection['ranking'],'selected':selection['selected']}
Path(sys.argv[2]).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'passed':True,'ranking':selection['ranking']},indent=2))
PY
