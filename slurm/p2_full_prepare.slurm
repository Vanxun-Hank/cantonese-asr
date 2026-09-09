#!/usr/bin/env bash
#SBATCH --job-name=canto-p2-full-prepare
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
set -euo pipefail
: "${P2_CODE_ROOT:?isolated code root required}"
: "${P2_ASSET_ROOT:?asset root required}"
: "${P2_PREPARATION:?new preparation output required}"
P2_PYTHON=${P2_PYTHON:-/home/bolin/envs/cantonese-asr-whisper/bin/python}
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4
cd "$P2_CODE_ROOT"
exec "$P2_PYTHON" scripts/run_p2_full.py prepare \
    --config configs/rounds/raw_winner_p2_full.json \
    --asset-root "$P2_ASSET_ROOT" --output-dir "$P2_PREPARATION" --decode-audio
