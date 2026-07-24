#!/usr/bin/env bash
#SBATCH --job-name=canto-ext-data
#SBATCH --partition=gpu
#SBATCH --exclude=gpu002
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CV_ROOT="artifacts/datasets/external/common_voice_26_zh_HK/extracted_train/cv-corpus-26.0-2026-06-12/zh-HK"
MDCC_ROOT="artifacts/datasets/external/mdcc/hf_snapshot/data"
MDCC_AUDIO_ROOT="artifacts/data/external/mdcc_train"
OUTPUT_DIR="artifacts/manifests/external/v1"

if [[ ! -f "$CV_ROOT/train.tsv" ]]; then
  echo "Common Voice train extraction is incomplete: $CV_ROOT/train.tsv" >&2
  exit 1
fi
EXPECTED_CV_ROWS=$(($(wc -l < "$CV_ROOT/train.tsv") - 1))
ACTUAL_CV_CLIPS=$(find "$CV_ROOT/clips" -maxdepth 1 -type f | wc -l)
if [[ "$ACTUAL_CV_CLIPS" -ne "$EXPECTED_CV_ROWS" ]]; then
  echo "Common Voice clip count mismatch: $ACTUAL_CV_CLIPS != $EXPECTED_CV_ROWS" >&2
  exit 1
fi

"$PYTHON" scripts/prepare_external_manifest.py \
  --common-voice-root "$CV_ROOT" \
  --mdcc-parquet-root "$MDCC_ROOT" \
  --mdcc-audio-root "$MDCC_AUDIO_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --project-root "$PROJECT_DIR" \
  --protected-text-jsonl artifacts/manifests/validation.jsonl \
  --protected-text-jsonl artifacts/datasets/official/template_pre.jsonl \
  --max-duration 30

"$PYTHON" - <<'PY'
import json
from pathlib import Path

report_path = Path("artifacts/manifests/external/v1/data_report.json")
report = json.loads(report_path.read_text(encoding="utf-8"))
checks = report["leakage_checks"]
if checks["protected_text_overlap"] != 0:
    raise SystemExit(f"Protected text leakage: {checks}")
if checks["duplicate_audio_sha256"] != 0:
    raise SystemExit(f"Duplicate audio leakage: {checks}")
if report["total"]["rows"] == 0:
    raise SystemExit("External manifest is empty")
print(json.dumps(report, ensure_ascii=False, indent=2))
PY
