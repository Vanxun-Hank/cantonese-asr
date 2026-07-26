#!/usr/bin/env bash
#SBATCH --job-name=canto-cv-yue-audio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

AUDIT_DIR="artifacts/audits/common_voice_26_yue_vs_zh_hk/v1"
SENSEVOICE_MODEL="artifacts/models/SenseVoiceSmall"
FUNASR_WHEEL="$PROJECT_DIR/third_party/funasr-1.3.28-py3-none-any.whl"

if [[ ! -f "$FUNASR_WHEEL" ]]; then
  echo "FunASR wheel is missing: $FUNASR_WHEEL" >&2
  exit 1
fi
if [[ ! -f "$AUDIT_DIR/style_review_sample.csv" ]]; then
  echo "CPU audit review sample is missing" >&2
  exit 1
fi

RUNTIME_DIR=$(mktemp -d "${TMPDIR:-/tmp}/canto-funasr.XXXXXX")
trap 'rm -rf "$RUNTIME_DIR"' EXIT
unzip -q "$FUNASR_WHEEL" -d "$RUNTIME_DIR"
export PYTHONPATH="$RUNTIME_DIR:$PYTHONPATH"

"$PYTHON" scripts/audit_yue_acoustic_sample.py \
  --audit-dir "$AUDIT_DIR" \
  --sensevoice-model "$SENSEVOICE_MODEL" \
  --whisper-model artifacts/models/whisper-small \
  --project-root "$PROJECT_DIR" \
  --device cuda:0 \
  --batch-size 16

"$PYTHON" - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/audits/common_voice_26_yue_vs_zh_hk/v1")
report = json.loads((root / "data_report.json").read_text(encoding="utf-8"))
summary = json.loads((root / "acoustic_summary.json").read_text(encoding="utf-8"))
if summary["sample_rows"] > 230 or summary["sample_rows"] == 0:
    raise SystemExit("invalid acoustic review sample size")
if summary["training_runs_started"] != 0 or summary["platform_submissions"] != 0:
    raise SystemExit("diagnostic job must not train or submit")
if report["admission"]["decision"] == "pending_acoustic_review":
    raise SystemExit("acoustic review did not finalize admission")
print(json.dumps({
    "decision": report["admission"]["decision"],
    "sample_yue_rate": summary["sample_yue_rate"],
    "languages": summary["languages"],
}, ensure_ascii=False, indent=2))
PY
