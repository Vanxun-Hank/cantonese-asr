#!/usr/bin/env bash
#SBATCH --job-name=canto-p2-full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=72:00:00
# Resource count and node MUST be supplied to sbatch after live availability checks.
set -euo pipefail
: "${P2_CODE_ROOT:?isolated code root required}"
: "${P2_ASSET_ROOT:?frozen asset root required}"
: "${P2_PREPARATION:?verified preparation required}"
: "${P2_RUN_ROOT:?new output root required}"
: "${P2_ARM:?registered arm required}"
: "${P2_SEED:?seed required}"
: "${P2_PHASE:?smoke benchmark train or extend required}"
P2_PYTHON=${P2_PYTHON:-/home/bolin/envs/cantonese-asr-whisper/bin/python}
cd "$P2_CODE_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
scontrol show job "$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv
args=(--config "$P2_CODE_ROOT/configs/rounds/raw_winner_p2_full.json"
      --asset-root "$P2_ASSET_ROOT" --preparation "$P2_PREPARATION"
      --run-root "$P2_RUN_ROOT" --arm "$P2_ARM" --seed "$P2_SEED" --phase "$P2_PHASE")
if [[ -n ${P2_ACCEPTANCE:-} ]]; then args+=(--acceptance "$P2_ACCEPTANCE"); fi
if [[ -n ${P2_EXTENSION_RECEIPT:-} ]]; then args+=(--extension-receipt "$P2_EXTENSION_RECEIPT"); fi
exec "$P2_PYTHON" scripts/p2_full_capacity_worker.py "${args[@]}"
