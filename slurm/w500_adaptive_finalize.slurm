#!/usr/bin/env bash
# Final report plus every guardrail-eligible raw/interpolation offline package.
#SBATCH --job-name=canto-w500-adapt-final
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu001
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --mem=224G
#SBATCH --time=12:00:00
#SBATCH --output=/home/bolin/cantonese-asr/logs/%x-%j.out

set -euo pipefail
source /home/bolin/cantonese-asr/slurm/common.sh
export CANTO_ALLOWED_GPU_INDICES=0,1,2,3
require_allowed_gpu_allocation

CONFIG=configs/rounds/w500_adaptive_continuation.json
ARTIFACT=artifacts/w500_adaptive_continuation_v1
PREPARATION=$ARTIFACT/preparation.json
INTERP=$ARTIFACT/interpolation
FINAL=$ARTIFACT/final
STAGES=$FINAL/stage_selections
MATRIX=$FINAL/matrix.json
CURVES=$FINAL/curves.csv
TASKS=$FINAL/tasks.json
REPORT=reports/w500_adaptive_continuation.md
PACKAGE_ROOT=outputs/submissions/w500-adaptive-continuation-v1
STAGING=$FINAL/package_staging
LAUNCH=$FINAL/launch

physical_gpus="$(scontrol show job -dd "$SLURM_JOB_ID" | sed -n 's/.*GRES=gpu:[^ ]*(IDX:\([^)]*\)).*/\1/p' | head -n 1)"
if [[ "$physical_gpus" != "0,1,2,3" && "$physical_gpus" != "0-3" ]]; then
  echo "Refusing allocation ${physical_gpus:-unknown}; expected physical GPUs 0-3" >&2
  exit 42
fi
for path in "$FINAL" "$PACKAGE_ROOT" "$REPORT"; do
  if [[ -e "$path" ]]; then echo "Refusing to overwrite final artifact: $path" >&2; exit 3; fi
done
mkdir -p "$FINAL" "$STAGES" "$PACKAGE_ROOT" "$STAGING" "$LAUNCH" reports
scontrol show job -dd "$SLURM_JOB_ID" >"$LAUNCH/slurm_job_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader >"$LAUNCH/gpu_allocation_${SLURM_JOB_ID}.txt"
for stage in 40 45 50; do
  cp "$ARTIFACT/stage${stage}/selection.json" "$STAGES/stage${stage}_selection.json"
done

"$PYTHON" scripts/finalize_w500_adaptive_continuation.py \
  --config "$CONFIG" \
  --preparation "$PREPARATION" \
  --stage-root "$STAGES" \
  --interpolation-root "$INTERP" \
  --matrix "$MATRIX" \
  --curves "$CURVES" \
  --tasks "$TASKS" \
  --report "$REPORT" \
  >"$LAUNCH/finalize_${SLURM_JOB_ID}.log" 2>&1

pids=()
for gpu in 0 1 2 3; do
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    "$PYTHON" - "$TASKS" "$gpu" <<'PY' | while IFS=$'\t' read -r label model processor weight evidence rank; do
import json,sys
rows=json.load(open(sys.argv[1]))['tasks']; slot=int(sys.argv[2])
for index,row in enumerate(rows):
 if index%4==slot:
  print('\t'.join((row['label'],row['model_dir'],row['processor_dir'],row['weight_sha256'],row['outputs_pre'],str(row['fixed_validation_rank']))))
PY
      package="$PACKAGE_ROOT/$label"
      stage="$STAGING/$label"
      zip="$package/W500_Adaptive_${label}_submission.zip"
      outputs_pre="$package/outputs_pre"
      parity="$package/parity_audit"
      mkdir -p "$stage" "$outputs_pre"
      cp "$model/model.safetensors" "$model/config.json" "$model/generation_config.json" "$stage/"
      for file in preprocessor_config.json vocab.json merges.txt tokenizer.json tokenizer_config.json special_tokens_map.json normalizer.json added_tokens.json; do
        [[ -f "$processor/$file" ]] && cp "$processor/$file" "$stage/$file"
      done
      for required in model.safetensors config.json generation_config.json preprocessor_config.json vocab.json merges.txt tokenizer_config.json special_tokens_map.json normalizer.json; do
        [[ -f "$stage/$required" ]] || { echo "Missing $stage/$required" >&2; exit 5; }
      done
      for surface in validation public ood; do
        mkdir -p "$outputs_pre/$surface"
        cp "$evidence/$surface/metrics.json" "$evidence/$surface/top_confusions.csv" \
          "$evidence/$surface/error_examples.json" "$evidence/$surface/generation.json" \
          "$evidence/$surface/predictions.jsonl" "$evidence/$surface/generation_tokens.jsonl" \
          "$outputs_pre/$surface/"
      done
      for file in metrics.json top_confusions.csv error_examples.json generation.json; do
        cp "$evidence/public/$file" "$outputs_pre/$file"
      done
      "$PYTHON" scripts/package_submission.py --model-dir "$stage" --predict-py predict.py \
        --requirements requirements-submission.txt --output-zip "$zip"
      "$PYTHON" scripts/verify_submission.py --submission-zip "$zip" \
        --audio-dir artifacts/data/test_audio --test-list artifacts/datasets/official/template.jsonl \
        --max-samples 32 --output-report "$package/offline_submission_verification.json"
      "$PYTHON" scripts/audit_submission_parity.py --raw-model-dir "$model" \
        --raw-processor-dir "$processor" --submission-zip "$zip" \
        --audio-dir artifacts/data/train_raw/4856-10393 \
        --test-list artifacts/datasets/official/template_pre.jsonl --max-samples 32 \
        --generation-max-length 225 --output-dir "$parity"
      "$PYTHON" - "$package" "$zip" "$model" "$processor" "$label" "$weight" "$rank" <<'PY'
import hashlib,json,sys,zipfile
from pathlib import Path
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
package,zip_path,model,processor=map(Path,sys.argv[1:5]); label=sys.argv[5]; expected=sys.argv[6]; rank=int(sys.argv[7])
verification=json.load(open(package/'offline_submission_verification.json'))
parity=json.load(open(package/'parity_audit/parity_receipt.json'))
with zipfile.ZipFile(zip_path) as z:
 names=z.namelist(); payload={name:z.read(name) for name in ('config.json','generation_config.json','predict.py')}
out={'passed':bool(verification['passed']) and bool(parity['passed']) and sha(model/'model.safetensors')==expected
     and all('/' not in name.rstrip('/') for name in names) and names.count('model.safetensors')==1,
 'label':label,'fixed_validation_rank':rank,'checkpoint':str(model.resolve()),'processor_dir':str(processor.resolve()),
 'checkpoint_weight_sha256':expected,'submission_zip':str(zip_path.resolve()),'submission_size_bytes':zip_path.stat().st_size,
 'submission_sha256':sha(zip_path),'flat_zip':all('/' not in name.rstrip('/') for name in names),
 'model_safetensors_count':names.count('model.safetensors'),'config_sha256':hashlib.sha256(payload['config.json']).hexdigest(),
 'generation_config_sha256':hashlib.sha256(payload['generation_config.json']).hexdigest(),
 'predict_py_sha256':hashlib.sha256(payload['predict.py']).hexdigest(),'offline32':bool(verification['passed']),
 'raw_zip_prediction_parity':bool(parity['passed']),'outputs_pre':str((package/'outputs_pre').resolve()),
 'automatic_platform_upload':False}
(package/'submission_manifest.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2)); raise SystemExit(0 if out['passed'] else 12)
PY
    done
  ) >"$LAUNCH/package_gpu${gpu}_${SLURM_JOB_ID}.log" 2>&1 &
  pids+=("$!")
done
failures=0
for gpu in 0 1 2 3; do wait "${pids[$gpu]}" || failures=$((failures+1)); done
if (( failures > 0 )); then echo "$failures package worker(s) failed" >&2; exit 10; fi

"$PYTHON" - "$TASKS" "$PACKAGE_ROOT" "$FINAL/package_completion.json" <<'PY'
import json,sys
from pathlib import Path
tasks=json.load(open(sys.argv[1]))['tasks']; root=Path(sys.argv[2]); rows={}
for task in tasks:
 path=root/task['label']/'submission_manifest.json'; value=json.load(open(path)) if path.is_file() else None
 rows[task['label']]={'passed':bool(value and value.get('passed')),'manifest':value}
out={'passed':bool(rows) and all(row['passed'] for row in rows.values()),'packages':rows,'automatic_platform_upload':False}
Path(sys.argv[3]).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2)); raise SystemExit(0 if out['passed'] else 11)
PY
