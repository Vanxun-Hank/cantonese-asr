#!/usr/bin/env bash
#SBATCH --job-name=canto-r8-data
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/bolin/cantonese-asr/logs/%x-%j.out

set -euo pipefail
source /home/bolin/cantonese-asr/slurm/common.sh

OUTPUT=artifacts/manifests/round8/official-mdcc73
if [[ -e "$OUTPUT" ]] && [[ -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty immutable Round 8 manifest directory: $OUTPUT" >&2
  exit 3
fi

"$PYTHON" scripts/build_mdcc_source_isolation_manifest.py \
  --official-train artifacts/manifests/train.jsonl \
  --mdcc-train artifacts/manifests/external/v1/mdcc_train.jsonl \
  --round2-manifest artifacts/manifests/external/round2/external73.jsonl \
  --excluded-manifest artifacts/manifests/validation.jsonl \
  --excluded-manifest artifacts/manifests/public_excluded.jsonl \
  --excluded-manifest artifacts/manifests/ood/v1/ood_all.jsonl \
  --output-dir "$OUTPUT" \
  --project-root "$PROJECT_DIR" \
  --seed 42 \
  --expected-official-rows 6292 \
  --expected-mdcc-pool-rows 64779 \
  --mdcc-sample-count 17012 \
  --expected-official-sha256 0d3d1df4b90f89907443ad9706f13d9ff0ae5fe66d6b32c939af99645867cb81 \
  --expected-mdcc-sha256 fc2c33f6d10f2603de3aa60f474f94889e0d140e35f5dfc29ba78fedee3013bc \
  --expected-round2-sha256 2ba7edef6811b0d3e96b3347aa70f4237ec27735652d3d313bce62525f1cdea6 \
  --verify-audio-exists

"$PYTHON" - "$OUTPUT" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

from cantonese_asr.io import read_jsonl, sha256_file, write_jsonl

root = Path(sys.argv[1])
report = json.loads((root / "data_report.json").read_text(encoding="utf-8"))
manifest = root / "train.jsonl"
if report.get("passed") is not True:
    raise SystemExit("Round 8 data report is not marked passed")
if report["output"]["rows"] != 23_304:
    raise SystemExit(f"Unexpected output rows: {report['output']}")
if report["output"]["unique_ids"] != 23_304:
    raise SystemExit(f"Unexpected unique ID count: {report['output']}")
if report["output"]["sources"] != {"mdcc": 17_012, "official": 6_292}:
    raise SystemExit(f"Unexpected source mix: {report['output']['sources']}")
if report["output"]["sha256"] != sha256_file(manifest):
    raise SystemExit("Round 8 manifest SHA-256 does not match its report")

panel = read_jsonl(Path("artifacts/manifests/ood/v1/ood_panel.jsonl"))
sources = Counter(str(row.get("source")) for row in panel)
if sources != {"common_voice_26_zh_HK": 1_000, "mdcc": 1_000}:
    raise SystemExit(f"Unexpected OOD panel sources: {sources}")
write_jsonl(
    root / "ood_cv.jsonl",
    [row for row in panel if row.get("source") == "common_voice_26_zh_HK"],
)
write_jsonl(
    root / "ood_mdcc.jsonl",
    [row for row in panel if row.get("source") == "mdcc"],
)
print(json.dumps(report, ensure_ascii=False, indent=2))
PY
