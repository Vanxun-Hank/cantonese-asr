#!/usr/bin/env bash
#SBATCH --job-name=canto-cv-yue-audit
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

MDC_ENV="$HOME/.config/cantonese-asr/mdc.env"
if [[ -z "${MDC_API_KEY:-}" && -f "$MDC_ENV" ]]; then
  # The file contains only private environment exports and must never be logged.
  # shellcheck disable=SC1090
  set -a
  source "$MDC_ENV"
  set +a
fi
if [[ -z "${MDC_API_KEY:-}" ]]; then
  echo "MDC_API_KEY is not configured" >&2
  exit 1
fi

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

DATASET_ID="cmqinjd7x00vynq07pwzo3lmp"
DATASET_DIR="artifacts/datasets/external/common_voice_26_yue"
AUDIT_DIR="artifacts/audits/common_voice_26_yue_vs_zh_hk/v1"
EXTRACT_ROOT="$DATASET_DIR"

mkdir -p "$DATASET_DIR" "$AUDIT_DIR"

"$PYTHON" scripts/download_mdc_dataset.py \
  --dataset-id "$DATASET_ID" \
  --output-dir "$DATASET_DIR" \
  --workers 8 \
  --segment-mib 64

mapfile -t ARCHIVES < <(
  find "$DATASET_DIR" -maxdepth 1 -type f \
    ! -name '*.part' ! -name '*.sha256' | sort
)
if [[ "${#ARCHIVES[@]}" -ne 1 ]]; then
  echo "Expected exactly one completed yue archive, found ${#ARCHIVES[@]}" >&2
  printf '%s\n' "${ARCHIVES[@]}" >&2
  exit 1
fi
ARCHIVE="${ARCHIVES[0]}"
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"

"$PYTHON" scripts/audit_common_voice_yue.py \
  --archive "$ARCHIVE" \
  --output-dir "$AUDIT_DIR" \
  --extract-root "$EXTRACT_ROOT" \
  --project-root "$PROJECT_DIR" \
  --dataset-id "$DATASET_ID" \
  --whisper-model artifacts/models/whisper-small \
  --reference-manifest official_train=artifacts/manifests/train.jsonl \
  --reference-manifest zh_hk_train=artifacts/manifests/external/v1/common_voice_train.jsonl \
  --reference-manifest zh_hk_dev=artifacts/manifests/ood/v1/common_voice_dev.jsonl \
  --reference-manifest zh_hk_test=artifacts/manifests/ood/v1/common_voice_test.jsonl \
  --reference-manifest mdcc_train=artifacts/manifests/external/v1/mdcc_train.jsonl \
  --reference-manifest mdcc_validation=artifacts/manifests/ood/v1/mdcc_validation.jsonl \
  --reference-manifest mdcc_test=artifacts/manifests/ood/v1/mdcc_test.jsonl \
  --protected-manifest artifacts/manifests/validation.jsonl \
  --protected-manifest artifacts/datasets/official/template_pre.jsonl \
  --seed 42 \
  --max-duration 30 \
  --max-label-tokens 225

"$PYTHON" - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/audits/common_voice_26_yue_vs_zh_hk/v1")
report = json.loads((root / "data_report.json").read_text(encoding="utf-8"))
provenance = report["provenance"]
if provenance["dataset_id"] != "cmqinjd7x00vynq07pwzo3lmp":
    raise SystemExit("wrong dataset ID")
if not str(provenance["archive_root"]).endswith("/yue"):
    raise SystemExit(f"wrong archive locale: {provenance['archive_root']}")
if provenance["training_runs_started"] != 0 or provenance["platform_submissions"] != 0:
    raise SystemExit("audit must not train or submit")
if report["incremental"]["protected_text_overlap"] != 0:
    raise SystemExit("protected text leakage is nonzero")
if report["incremental"]["protected_audio_overlap"] != 0:
    raise SystemExit("protected audio leakage is nonzero")
print(json.dumps({
    "decision": report["admission"]["decision"],
    "quality": report["quality"],
    "incremental": report["incremental"],
}, ensure_ascii=False, indent=2))
PY
