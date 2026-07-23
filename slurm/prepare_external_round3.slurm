#!/usr/bin/env bash
#SBATCH --job-name=canto-ext-r3-data
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/bolin/cantonese-asr/logs/%x-%j.out

set -euo pipefail

source /home/bolin/cantonese-asr/slurm/common.sh

OUTPUT_DIR=artifacts/manifests/external/round3
mkdir -p "$OUTPUT_DIR"

"$PYTHON" scripts/build_external_training_mixes.py \
  --official-train artifacts/manifests/train.jsonl \
  --common-voice artifacts/manifests/external/v1/common_voice_train.jsonl \
  --mdcc artifacts/manifests/external/v1/mdcc_train.jsonl \
  --output-dir "$OUTPUT_DIR" \
  --seed 42 \
  --external-fraction 0.73 \
  --external-fraction 0.80 \
  --external-fraction 0.85 \
  --nested-fractions \
  --include-all-external

"$PYTHON" - <<'PY'
import json
from pathlib import Path

from cantonese_asr.io import read_jsonl

output_dir = Path("artifacts/manifests/external/round3")
report_path = output_dir / "mix_report.json"
report = json.loads(report_path.read_text(encoding="utf-8"))

input_paths = {
    "official": Path("artifacts/manifests/train.jsonl"),
    "common_voice": Path(
        "artifacts/manifests/external/v1/common_voice_train.jsonl"
    ),
    "mdcc": Path("artifacts/manifests/external/v1/mdcc_train.jsonl"),
}
forbidden_paths = [
    Path("artifacts/manifests/validation.jsonl"),
    Path("artifacts/manifests/public_excluded.jsonl"),
    Path("artifacts/manifests/ood/v1/ood_all.jsonl"),
]
manifest_names = ["external73", "external80", "external85", "all_train"]


def ids_for(rows, label):
    ids = [str(row.get("id", "")) for row in rows]
    if any(not sample_id for sample_id in ids):
        raise AssertionError(f"{label} contains one or more empty IDs")
    duplicate_count = len(ids) - len(set(ids))
    if duplicate_count:
        raise AssertionError(
            f"{label} contains {duplicate_count} duplicate ID(s)"
        )
    return set(ids)


for path in [report_path, *input_paths.values(), *forbidden_paths]:
    if not path.is_file():
        raise FileNotFoundError(f"Required Round 3 verification input is missing: {path}")

forbidden_ids = set()
for path in forbidden_paths:
    forbidden_ids.update(ids_for(read_jsonl(path), str(path)))

manifests = {}
manifest_ids = {}
external_ids = {}
for name in manifest_names:
    path = output_dir / f"{name}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Generated manifest is missing: {path}")
    rows = read_jsonl(path)
    sample_ids = ids_for(rows, name)
    overlap = sample_ids & forbidden_ids
    if overlap:
        examples = sorted(overlap)[:5]
        raise AssertionError(
            f"{name} contains {len(overlap)} forbidden ID(s), examples: {examples}"
        )
    if name not in report.get("manifests", {}):
        raise AssertionError(f"mix_report.json has no entry for {name}")
    reported_rows = report["manifests"][name].get("rows")
    if reported_rows != len(rows):
        raise AssertionError(
            f"{name} row count mismatch: file={len(rows)}, report={reported_rows}"
        )
    manifests[name] = rows
    manifest_ids[name] = sample_ids
    external_ids[name] = {
        str(row["id"])
        for row in rows
        if str(row.get("source") or "official") != "official"
    }

if not (
    external_ids["external73"]
    < external_ids["external80"]
    < external_ids["external85"]
    < external_ids["all_train"]
):
    raise AssertionError(
        "External ID sets are not strict nested subsets in the required order: "
        "external73 < external80 < external85 < all_train"
    )

expected_all_train_ids = set()
for label, path in input_paths.items():
    expected_all_train_ids.update(ids_for(read_jsonl(path), f"{label} input"))
if manifest_ids["all_train"] != expected_all_train_ids:
    missing = expected_all_train_ids - manifest_ids["all_train"]
    unexpected = manifest_ids["all_train"] - expected_all_train_ids
    raise AssertionError(
        "all_train does not equal the exact union of publisher-train inputs: "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )

summary = {
    name: {
        "rows": len(manifests[name]),
        "external_rows": len(external_ids[name]),
        "sha256": report["manifests"][name]["sha256"],
    }
    for name in manifest_names
}
print("Round 3 manifest verification passed.")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
