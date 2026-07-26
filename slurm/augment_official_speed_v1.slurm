#!/usr/bin/env bash
#SBATCH --job-name=canto-speed-aug
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

RUN_TAG=${SPEED_AUG_RUN_TAG:-official-speed-v1}
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "SPEED_AUG_RUN_TAG contains unsafe path characters: $RUN_TAG" >&2
  exit 2
fi
if [[ "$RUN_TAG" != "official-speed-v1" ]]; then
  echo "This approved run only permits SPEED_AUG_RUN_TAG=official-speed-v1" >&2
  exit 2
fi

AUDIO_ROOT="artifacts/data/augmented/$RUN_TAG"
MANIFEST_DIR="artifacts/manifests/augmentation/$RUN_TAG"
WORKERS=${AUGMENT_WORKERS:-8}

"$PYTHON" scripts/augment_speed_manifest.py \
  --train-manifest artifacts/manifests/train.jsonl \
  --validation-manifest artifacts/manifests/validation.jsonl \
  --public-manifest artifacts/manifests/public_excluded.jsonl \
  --output-audio-root "$AUDIO_ROOT" \
  --output-manifest-dir "$MANIFEST_DIR" \
  --project-root "$PROJECT_DIR" \
  --expected-parents 6292 \
  --workers "$WORKERS" \
  --version-tag "$RUN_TAG"

"$PYTHON" scripts/verify_speed_augmentation.py \
  --manifest "$MANIFEST_DIR/all.jsonl" \
  --report "$MANIFEST_DIR/data_report.json" \
  --quarantine "$MANIFEST_DIR/quarantine.jsonl" \
  --project-root "$PROJECT_DIR" \
  --sample-parents 32 \
  --output "$MANIFEST_DIR/verification.json"

echo "Speed augmentation completed; no training was launched."
echo "Manifest: $MANIFEST_DIR/all.jsonl"
echo "Audio root: $AUDIO_ROOT"

