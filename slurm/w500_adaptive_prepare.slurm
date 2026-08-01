#!/usr/bin/env bash
# Audit the W500 pool, score every eligible row on four GPUs, then freeze the
# immutable nested continuation manifests.
#SBATCH --job-name=canto-w500-adapt-prep
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu001
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/home/bolin/cantonese-asr/logs/%x-%j.out

set -euo pipefail
source /home/bolin/cantonese-asr/slurm/common.sh

export CANTO_ALLOWED_GPU_INDICES=0,1,2,3
require_allowed_gpu_allocation

CONFIG=configs/rounds/w500_adaptive_continuation.json
ARTIFACT=artifacts/w500_adaptive_continuation_v1
LAUNCH=artifacts/w500_adaptive_continuation_launch/prepare
START=outputs/w500-encoder-only-incremental-v2/W500_EONLY_R1TO1_ELR1E6/checkpoint-wenet-35h

physical_gpus="$(
  scontrol show job -dd "$SLURM_JOB_ID" \
    | sed -n 's/.*GRES=gpu:[^ ]*(IDX:\([^)]*\)).*/\1/p' \
    | head -n 1
)"
if [[ "$physical_gpus" != "0,1,2,3" && "$physical_gpus" != "0-3" ]]; then
  echo "Refusing allocation ${physical_gpus:-unknown}; expected physical GPUs 0-3" >&2
  exit 42
fi
if [[ -e "$ARTIFACT" ]]; then
  echo "Refusing to overwrite immutable preparation output: $ARTIFACT" >&2
  exit 2
fi
mkdir -p "$LAUNCH" logs
scontrol show job -dd "$SLURM_JOB_ID" >"$LAUNCH/slurm_job_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader \
  >"$LAUNCH/gpu_allocation_${SLURM_JOB_ID}.txt"
sha256sum "$CONFIG" cantonese_asr/w500_adaptive.py \
  scripts/prepare_w500_adaptive_continuation.py \
  scripts/evaluate_w500_candidate_losses.py \
  >"$LAUNCH/code_sha256_${SLURM_JOB_ID}.txt"

"$PYTHON" - "$CONFIG" <<'PY'
import hashlib,json,sys
from pathlib import Path

cfg=json.load(open(sys.argv[1]))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()
start=Path(cfg['start']['checkpoint'])
checks={
    'weight': (start/'model.safetensors', cfg['start']['weight_sha256']),
    'config': (start/'config.json', cfg['start']['config_sha256']),
    'generation_config': (start/'generation_config.json', cfg['start']['generation_config_sha256']),
    'predict': (Path('predict.py'), cfg['start']['predict_py_sha256']),
}
bad=[]
for label,(path,expected) in checks.items():
    observed=sha(path)
    print(f'{label}\t{observed}\t{path}')
    if observed != expected: bad.append((label,observed,expected))
if bad: raise SystemExit(f'Frozen provenance mismatch: {bad}')
stable=Path(cfg['start']['stable_decoder_checkpoint'])/'model.safetensors'
if not stable.is_file(): raise SystemExit(f'Missing stable Decoder endpoint: {stable}')
PY

"$PYTHON" scripts/prepare_w500_adaptive_continuation.py \
  --project-root "$PROJECT_DIR" \
  --config "$CONFIG" \
  --mode audit \
  --output-dir "$ARTIFACT" \
  >"$LAUNCH/audit_${SLURM_JOB_ID}.log" 2>&1

ELIGIBLE=$ARTIFACT/eligible_candidates.jsonl
ELIGIBLE_SHA=$(sha256sum "$ELIGIBLE" | awk '{print $1}')
SHARDS=$ARTIFACT/loss_shards
mkdir -p "$SHARDS"
"$PYTHON" - "$ELIGIBLE" "$SHARDS" <<'PY'
import json,sys
from pathlib import Path

source=Path(sys.argv[1]); root=Path(sys.argv[2])
handles=[(root/f'shard_{i}.jsonl').open('x',encoding='utf-8') for i in range(4)]
counts=[0,0,0,0]
try:
    with source.open(encoding='utf-8') as f:
        for index,line in enumerate(f):
            if not line.strip(): continue
            json.loads(line)
            shard=index % 4
            handles[shard].write(line)
            counts[shard]+=1
finally:
    for handle in handles: handle.close()
(root/'shard_counts.json').write_text(json.dumps({'passed':sum(counts)>0,'counts':counts},indent=2)+'\n')
print(json.dumps({'counts':counts},indent=2))
PY

pids=()
for gpu in 0 1 2 3; do
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    "$PYTHON" scripts/evaluate_w500_candidate_losses.py \
      --model-dir "$START" \
      --processor-dir "$START" \
      --manifest "$SHARDS/shard_${gpu}.jsonl" \
      --eligible-manifest-sha256 "$ELIGIBLE_SHA" \
      --project-root "$PROJECT_DIR" \
      --output "$SHARDS/losses_${gpu}.jsonl" \
      --receipt "$SHARDS/losses_${gpu}.receipt.json" \
      --batch-size 8 \
      --num-workers 6 \
      --dtype bfloat16
  ) >"$LAUNCH/loss_gpu${gpu}_${SLURM_JOB_ID}.log" 2>&1 &
  pids+=("$!")
done
failures=0
for gpu in 0 1 2 3; do
  if wait "${pids[$gpu]}"; then
    printf 'COMPLETED\n' >"$LAUNCH/loss_gpu${gpu}_${SLURM_JOB_ID}.status"
  else
    status=$?
    printf 'FAILED exit=%s\n' "$status" >"$LAUNCH/loss_gpu${gpu}_${SLURM_JOB_ID}.status"
    failures=$((failures + 1))
  fi
done
if (( failures > 0 )); then
  echo "$failures candidate-loss worker(s) failed" >&2
  exit 10
fi

LOSSES=$ARTIFACT/candidate_losses.jsonl
"$PYTHON" - "$ELIGIBLE" "$ELIGIBLE_SHA" "$SHARDS" "$LOSSES" <<'PY'
import hashlib,json,sys
from pathlib import Path

eligible,eligible_sha,shards,output=Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3]),Path(sys.argv[4])
ordered=[]
with eligible.open(encoding='utf-8') as f:
    for line in f:
        if line.strip(): ordered.append(str(json.loads(line)['id']))
losses={}
receipts=[]
for index in range(4):
    receipt=json.load(open(shards/f'losses_{index}.receipt.json'))
    receipts.append(receipt)
    if not receipt.get('passed') or receipt.get('eligible_manifest_sha256') != eligible_sha:
        raise SystemExit(f'Invalid loss receipt for shard {index}')
    with (shards/f'losses_{index}.jsonl').open(encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row=json.loads(line); identifier=str(row['id'])
            if identifier in losses: raise SystemExit(f'Duplicate loss ID: {identifier}')
            losses[identifier]=row
missing=set(ordered)-set(losses); extra=set(losses)-set(ordered)
if missing or extra: raise SystemExit(f'Loss coverage mismatch missing={len(missing)} extra={len(extra)}')
with output.open('x',encoding='utf-8') as f:
    for identifier in ordered:
        f.write(json.dumps(losses[identifier],ensure_ascii=False,allow_nan=False)+'\n')
h=hashlib.sha256(output.read_bytes()).hexdigest()
receipt={'passed':True,'eligible_manifest':str(eligible.resolve()),
         'eligible_manifest_sha256':eligible_sha,'rows':len(ordered),
         'output':str(output.resolve()),'output_sha256':h,'shards':receipts}
output.with_suffix('.receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
PY

"$PYTHON" scripts/prepare_w500_adaptive_continuation.py \
  --project-root "$PROJECT_DIR" \
  --config "$CONFIG" \
  --mode freeze \
  --candidate-losses "$LOSSES" \
  --output-dir "$ARTIFACT" \
  >"$LAUNCH/freeze_${SLURM_JOB_ID}.log" 2>&1

test "$(cat "$ARTIFACT/MANIFESTS_FROZEN")" = PASS
printf 'COMPLETED\n' >"$LAUNCH/status_${SLURM_JOB_ID}.txt"
