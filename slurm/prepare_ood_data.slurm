#!/usr/bin/env bash
#SBATCH --job-name=canto-ood-data
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CV_DATASET_ROOT=artifacts/datasets/external/common_voice_26_zh_HK
CV_ROOT="$CV_DATASET_ROOT/partial_full_extract_20260723/cv-corpus-26.0-2026-06-12/zh-HK"
CV_ARCHIVE="$CV_DATASET_ROOT/common-voice-scripted-speech-26-0-chines-0240e64b.tar.gz"
CV_AUDIO_ROOT=artifacts/data/ood/common_voice
MDCC_ROOT=artifacts/datasets/external/mdcc/hf_snapshot/data
MDCC_AUDIO_ROOT=artifacts/data/ood/mdcc
OUTPUT_DIR=artifacts/manifests/ood/v1

for required in "$CV_ROOT/dev.tsv" "$CV_ROOT/test.tsv" "$CV_ROOT/clip_durations.tsv" "$CV_ARCHIVE"; do
  if [[ ! -f "$required" ]]; then
    echo "Required Common Voice asset is missing: $required" >&2
    exit 1
  fi
done

"$PYTHON" scripts/prepare_ood_manifest.py \
  --common-voice-root "$CV_ROOT" \
  --common-voice-archive "$CV_ARCHIVE" \
  --common-voice-audio-root "$CV_AUDIO_ROOT" \
  --mdcc-parquet-root "$MDCC_ROOT" \
  --mdcc-audio-root "$MDCC_AUDIO_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --project-root "$PROJECT_DIR" \
  --excluded-manifest artifacts/manifests/train.jsonl \
  --excluded-manifest artifacts/manifests/validation.jsonl \
  --excluded-manifest artifacts/manifests/public_excluded.jsonl \
  --excluded-manifest artifacts/manifests/external/v1/external_all.jsonl \
  --panel-per-split 500 \
  --seed 42 \
  --max-duration 30

"$PYTHON" - <<'PY'
import json
from pathlib import Path

report_path = Path("artifacts/manifests/ood/v1/data_report.json")
report = json.loads(report_path.read_text(encoding="utf-8"))
checks = report["leakage_checks"]
if any(checks.values()):
    raise SystemExit(f"OOD isolation failure: {checks}")
if report["panel"]["total"]["rows"] > 2000:
    raise SystemExit(f"OOD panel exceeds 2,000 rows: {report['panel']['total']['rows']}")
if set(report["raw_candidate_rows"]) != {
    "common_voice_26_zh_HK:dev",
    "common_voice_26_zh_HK:test",
    "mdcc:validation",
    "mdcc:test",
}:
    raise SystemExit("OOD publisher split coverage is incomplete")
print(json.dumps(report, ensure_ascii=False, indent=2))
PY
