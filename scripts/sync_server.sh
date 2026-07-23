#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR=${LOCAL_DIR:-/Users/zhangxun/Desktop/service/cantonese-asr}
REMOTE_HOST=${REMOTE_HOST:-pavb}
REMOTE_DIR=${REMOTE_DIR:-/home/bolin/cantonese-asr}

ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR/logs' '$REMOTE_DIR/outputs' '$REMOTE_DIR/artifacts/reports'"
rsync -av \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache' \
  --exclude '.ipynb_checkpoints' \
  --exclude 'artifacts' \
  --exclude 'outputs' \
  --exclude 'logs' \
  --exclude 'reports/server' \
  "$LOCAL_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"

echo "Synced code to $REMOTE_HOST:$REMOTE_DIR"
