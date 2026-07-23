#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/bolin/cantonese-asr}
ENV_DIR=${ENV_DIR:-/home/bolin/envs/cantonese-asr-whisper}
CONDA_BIN=${CONDA_BIN:-/home/public/conda/miniforge3/bin/conda}
REPORT_DIR="$PROJECT_DIR/artifacts/reports/deployment"

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "Conda executable not found: $CONDA_BIN" >&2
  exit 1
fi

mkdir -p \
  "$PROJECT_DIR/artifacts/datasets/official" \
  "$PROJECT_DIR/artifacts/data/train_raw" \
  "$PROJECT_DIR/artifacts/data/test_audio" \
  "$PROJECT_DIR/artifacts/data/quarantine" \
  "$PROJECT_DIR/artifacts/manifests" \
  "$PROJECT_DIR/artifacts/reports" \
  "$PROJECT_DIR/artifacts/models/whisper-small" \
  "$PROJECT_DIR/outputs" \
  "$PROJECT_DIR/logs" \
  "$REPORT_DIR"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$CONDA_BIN" create --yes --prefix "$ENV_DIR" python=3.11 pip
fi

"$CONDA_BIN" run --prefix "$ENV_DIR" python -m pip install --upgrade pip
"$CONDA_BIN" run --prefix "$ENV_DIR" python -m pip install -r "$PROJECT_DIR/requirements-server-torch.txt"
"$CONDA_BIN" run --prefix "$ENV_DIR" python -m pip install -r "$PROJECT_DIR/requirements.txt"

"$ENV_DIR/bin/python" -m pip freeze > "$REPORT_DIR/pip-freeze.txt"
"$ENV_DIR/bin/python" - <<'PY' > "$REPORT_DIR/python-environment.json"
import json
import platform
import torch

print(json.dumps({
    "python": platform.python_version(),
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available_on_login": torch.cuda.is_available(),
}, indent=2))
PY

echo "Environment ready: $ENV_DIR"

