#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOCAL_DIR=${LOCAL_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}
REMOTE_HOST=${REMOTE_HOST:?Set REMOTE_HOST, for example user@cluster}
REMOTE_DIR=${REMOTE_DIR:-cantonese-asr}

ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR/logs' '$REMOTE_DIR/outputs' '$REMOTE_DIR/artifacts/reports'"
rsync -av \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache' \
  --exclude '.ipynb_checkpoints' \
  --exclude 'artifacts' \
  --exclude 'outputs' \
  --exclude 'logs' \
  "$LOCAL_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"

echo "Synced code to $REMOTE_HOST:$REMOTE_DIR"
