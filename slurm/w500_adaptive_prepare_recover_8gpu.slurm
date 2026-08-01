#!/usr/bin/env bash
# Recover failed preparation job 1541 without rebuilding its audited pool.
# The preserved four shards are split into eight immutable recovery views and
# scored atomically across all four GPUs on gpu001 and gpu002.
#SBATCH --job-name=canto-w500-prep-r8
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=gpu001,gpu002
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/home/bolin/cantonese-asr/logs/%x-%j.out

set -euo pipefail
source /home/bolin/cantonese-asr/slurm/common.sh

FAILED_PREP_JOB_ID=1541
CONFIG=configs/rounds/w500_adaptive_continuation.json
ARTIFACT=artifacts/w500_adaptive_continuation_v1
PRESERVED_SHARDS=$ARTIFACT/loss_shards
ELIGIBLE=$ARTIFACT/eligible_candidates.jsonl
START=outputs/w500-encoder-only-incremental-v2/W500_EONLY_R1TO1_ELR1E6/checkpoint-wenet-35h
PROCESSOR=outputs/w500-encoder-only-incremental-v2/W500_EONLY_R1TO1_ELR1E6
RECOVERY=$ARTIFACT/recovery/${FAILED_PREP_JOB_ID}_processor_root_8gpu_${SLURM_JOB_ID}
DERIVED_SHARDS=$RECOVERY/shards
LOSSES=$RECOVERY/candidate_losses.jsonl
LOCK=$ARTIFACT/.adaptive_prepare_recovery.lock

EXPECTED_ELIGIBLE_SHA=237d1ad3ae238f0281f6f6dfd56a33c546e4344deb0938423626e8e588e1f4db
EXPECTED_ELIGIBLE_ROWS=99296
EXPECTED_PARENT_ROWS=24824
EXPECTED_PARENT_SHAS=(
  7d04d8b12c397f1f5e2edd2d26a831d6f219f9726fa69ec8eb1fe06af8a602e1
  06ab9240146241d6054b481cda02122c258a1dfbfc7cc8f14caa7e96df672b22
  2e50226771089ee5658596b00a15865fc1032e8dda8f21fd5ac10fc97cb77409
  f06b39116fe351a046de76881aa028ce979683fe3c5fb33770b3bc22edb31f59
)

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Another adaptive preparation/freeze recovery holds $LOCK" >&2
  exit 42
fi
if [[ -e "$RECOVERY" ]]; then
  echo "Refusing to overwrite recovery evidence: $RECOVERY" >&2
  exit 2
fi
mkdir -p "$DERIVED_SHARDS" logs

mapfile -t allocated_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ "${allocated_nodes[*]}" != "gpu001 gpu002" ]]; then
  echo "Refusing nodes: ${allocated_nodes[*]}; expected gpu001 gpu002" >&2
  exit 42
fi

scontrol show job -dd "$SLURM_JOB_ID" >"$RECOVERY/slurm_job_${SLURM_JOB_ID}.txt"
if ! sacct -j "$FAILED_PREP_JOB_ID" \
  --format=JobID,JobName%28,State,ExitCode,Elapsed,NodeList%24 -P \
  >"$RECOVERY/failed_job_${FAILED_PREP_JOB_ID}.sacct.txt" 2>"$RECOVERY/failed_job_${FAILED_PREP_JOB_ID}.sacct.err"; then
  # Some compute nodes cannot reach slurmdbd.  The original logs and the
  # controller-side evidence remain immutable; accounting transcription is
  # useful provenance but must not block model/data recovery.
  printf 'UNAVAILABLE_FROM_COMPUTE_NODE\n' >"$RECOVERY/failed_job_${FAILED_PREP_JOB_ID}.sacct.txt"
fi
for gpu in 0 1 2 3; do
  status=artifacts/w500_adaptive_continuation_launch/prepare/loss_gpu${gpu}_${FAILED_PREP_JOB_ID}.status
  if [[ ! -f "$status" || "$(cat "$status")" != "FAILED exit=1" ]]; then
    echo "Preserved worker failure receipt changed: $status" >&2
    exit 2
  fi
done
if ! grep -q '^4 candidate-loss worker(s) failed$' \
  "logs/canto-w500-adapt-prep-${FAILED_PREP_JOB_ID}.out"; then
  echo "Preserved preparation failure log changed" >&2
  exit 2
fi
sha256sum "$CONFIG" cantonese_asr/w500_adaptive.py \
  scripts/prepare_w500_adaptive_continuation.py \
  scripts/evaluate_w500_candidate_losses.py \
  scripts/run_w500_adaptive_loss_node.sh \
  slurm/w500_adaptive_prepare_recover_8gpu.slurm \
  >"$RECOVERY/code_sha256_${SLURM_JOB_ID}.txt"

if [[ ! -d "$ARTIFACT" || -e "$ARTIFACT/MANIFESTS_FROZEN" ]]; then
  echo "Recovery requires the failed, unfrozen preparation artifact" >&2
  exit 2
fi
for unexpected in \
  "$ARTIFACT/candidate_losses.jsonl" \
  "$ARTIFACT/candidate_losses.receipt.json" \
  "$ARTIFACT/manifests" \
  "$ARTIFACT/stage40_arms.json" \
  "$ARTIFACT/preparation.json"; do
  if [[ -e "$unexpected" ]]; then
    echo "Refusing unexpected post-audit artifact: $unexpected" >&2
    exit 2
  fi
done
if compgen -G "$PRESERVED_SHARDS/losses_*.jsonl" >/dev/null || \
  compgen -G "$PRESERVED_SHARDS/losses_*.receipt.json" >/dev/null; then
  echo "Refusing stale partial loss artifacts beside the preserved shards" >&2
  exit 2
fi

"$PYTHON" - \
  "$CONFIG" "$ARTIFACT" "$ELIGIBLE" "$PRESERVED_SHARDS" "$DERIVED_SHARDS" \
  "$EXPECTED_ELIGIBLE_SHA" "$EXPECTED_ELIGIBLE_ROWS" "$EXPECTED_PARENT_ROWS" \
  "${EXPECTED_PARENT_SHAS[@]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    config_path,
    artifact,
    eligible,
    parent_root,
    derived_root,
    expected_eligible_sha,
    expected_eligible_rows,
    expected_parent_rows,
    *expected_parent_shas,
) = sys.argv[1:]
config_path = Path(config_path)
artifact = Path(artifact)
eligible = Path(eligible)
parent_root = Path(parent_root)
derived_root = Path(derived_root)
expected_eligible_rows = int(expected_eligible_rows)
expected_parent_rows = int(expected_parent_rows)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


summary_path = artifact / "candidate_audit_summary.json"
audit_path = artifact / "candidate_audit.jsonl"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if not summary.get("passed"):
    raise SystemExit("Preserved candidate audit did not pass")
if int(summary["eligible"]["rows"]) != expected_eligible_rows:
    raise SystemExit("Preserved eligible row count changed")
if summary["eligible"]["sha256"] != expected_eligible_sha:
    raise SystemExit("Preserved eligible receipt SHA changed")
if sha(eligible) != expected_eligible_sha:
    raise SystemExit("Preserved eligible file SHA changed")
if sha(audit_path) != summary["candidate_audit"]["sha256"]:
    raise SystemExit("Preserved candidate audit file SHA changed")
if any(int(value) for value in summary["overlaps"].values()):
    raise SystemExit(f"Preserved audit now reports overlap: {summary['overlaps']}")

eligible_ids = []
with eligible.open(encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            eligible_ids.append(str(json.loads(line)["id"]))
if len(eligible_ids) != expected_eligible_rows or len(set(eligible_ids)) != len(eligible_ids):
    raise SystemExit("Preserved eligible IDs are incomplete or duplicated")

counts_receipt = json.loads((parent_root / "shard_counts.json").read_text(encoding="utf-8"))
if counts_receipt.get("counts") != [expected_parent_rows] * 4:
    raise SystemExit(f"Preserved shard counts changed: {counts_receipt}")

parent_ids: set[str] = set()
derived_receipts = []
for parent_index, expected_sha in enumerate(expected_parent_shas):
    parent = parent_root / f"shard_{parent_index}.jsonl"
    if sha(parent) != expected_sha:
        raise SystemExit(f"Preserved shard {parent_index} SHA changed")
    rows = [line for line in parent.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != expected_parent_rows:
        raise SystemExit(f"Preserved shard {parent_index} row count changed")
    for line in rows:
        identifier = str(json.loads(line)["id"])
        if identifier in parent_ids:
            raise SystemExit(f"Duplicate ID across preserved shards: {identifier}")
        parent_ids.add(identifier)
    for parity in (0, 1):
        worker_index = parent_index * 2 + parity
        output = derived_root / f"shard_{worker_index}.jsonl"
        selected = rows[parity::2]
        with output.open("x", encoding="utf-8") as handle:
            for line in selected:
                handle.write(line + "\n")
        receipt = {
            "passed": True,
            "worker_index": worker_index,
            "parent_shard_index": parent_index,
            "parent_shard": str(parent.resolve()),
            "parent_shard_sha256": expected_sha,
            "line_parity": parity,
            "rows": len(selected),
            "manifest": str(output.resolve()),
            "manifest_sha256": sha(output),
            "eligible_manifest_sha256": expected_eligible_sha,
        }
        (derived_root / f"shard_{worker_index}.receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        derived_receipts.append(receipt)
if parent_ids != set(eligible_ids):
    raise SystemExit(
        f"Preserved shard coverage changed: missing={len(set(eligible_ids)-parent_ids)} "
        f"extra={len(parent_ids-set(eligible_ids))}"
    )
if sum(item["rows"] for item in derived_receipts) != expected_eligible_rows:
    raise SystemExit("Derived recovery shards do not preserve all eligible rows")

config = json.loads(config_path.read_text(encoding="utf-8"))
receipt = {
    "passed": True,
    "recovery_of_job": 1541,
    "failure": "checkpoint directory lacked processor files",
    "audit_reused_without_rebuild": True,
    "eligible_manifest": str(eligible.resolve()),
    "eligible_manifest_sha256": expected_eligible_sha,
    "eligible_rows": expected_eligible_rows,
    "processor_checkpoint": config["start"]["processor_checkpoint"],
    "derived_shards": derived_receipts,
}
(derived_root.parent / "preflight.json").write_text(
    json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(receipt, ensure_ascii=False, indent=2))
PY

"$PYTHON" - "$CONFIG" "$START" "$PROCESSOR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
start = Path(sys.argv[2])
processor = Path(sys.argv[3])
if str(processor) != config["start"]["processor_checkpoint"]:
    raise SystemExit("Processor must be the W500 arm root registered in config")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


checks = {
    "weight": (start / "model.safetensors", config["start"]["weight_sha256"]),
    "config": (start / "config.json", config["start"]["config_sha256"]),
    "generation_config": (
        start / "generation_config.json",
        config["start"]["generation_config_sha256"],
    ),
    "predict": (Path("predict.py"), config["start"]["predict_py_sha256"]),
}
for name, expected in config["start"]["processor_files_sha256"].items():
    checks[f"processor:{name}"] = (processor / name, expected)
failures = []
for label, (path, expected) in checks.items():
    observed = sha(path)
    print(f"{label}\t{observed}\t{path}")
    if observed != expected:
        failures.append((label, observed, expected))
if failures:
    raise SystemExit(f"Frozen model/processor provenance mismatch: {failures}")
PY

export PROJECT_DIR PYTHON START PROCESSOR DERIVED_SHARDS RECOVERY EXPECTED_ELIGIBLE_SHA
if ! srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=0 \
  bash scripts/run_w500_adaptive_loss_node.sh
then
  printf 'FAILED worker stage\n' >"$RECOVERY/status.txt"
  echo "At least one 8-GPU recovery worker failed; merge/freeze skipped" >&2
  exit 10
fi

"$PYTHON" - \
  "$ELIGIBLE" "$EXPECTED_ELIGIBLE_SHA" "$DERIVED_SHARDS" "$RECOVERY" "$LOSSES" \
  "$START" "$PROCESSOR" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

eligible = Path(sys.argv[1])
eligible_sha = sys.argv[2]
shards = Path(sys.argv[3])
recovery = Path(sys.argv[4])
output = Path(sys.argv[5])
expected_model = str(Path(sys.argv[6]).resolve())
expected_processor = str(Path(sys.argv[7]).resolve())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


ordered = []
with eligible.open(encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            ordered.append(str(json.loads(line)["id"]))
losses = {}
worker_receipts = []
for worker_index in range(8):
    status_path = recovery / f"worker_{worker_index}.status"
    if status_path.read_text(encoding="utf-8").strip() != "COMPLETED":
        raise SystemExit(f"Worker {worker_index} did not complete")
    shard_receipt = json.loads(
        (shards / f"shard_{worker_index}.receipt.json").read_text(encoding="utf-8")
    )
    receipt_path = recovery / f"losses_{worker_index}.receipt.json"
    loss_path = recovery / f"losses_{worker_index}.jsonl"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    checks = {
        "passed": receipt.get("passed") is True,
        "eligible_sha": receipt.get("eligible_manifest_sha256") == eligible_sha,
        "model": receipt.get("model_dir") == expected_model,
        "processor": receipt.get("processor_dir") == expected_processor,
        "manifest": receipt.get("manifest") == shard_receipt["manifest"],
        "manifest_sha": receipt.get("manifest_sha256") == shard_receipt["manifest_sha256"],
        "rows": int(receipt.get("rows", -1)) == int(shard_receipt["rows"]),
        "output": receipt.get("output") == str(loss_path.resolve()),
        "output_sha": receipt.get("output_sha256") == sha(loss_path),
        "dtype": receipt.get("dtype") == "bfloat16",
    }
    if not all(checks.values()):
        raise SystemExit(f"Invalid worker {worker_index} receipt: {checks}")
    with loss_path.open(encoding="utf-8") as handle:
        rows = 0
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = str(row["id"])
            value = float(row["normalized_loss"])
            if row.get("eligible_manifest_sha256") != eligible_sha or not math.isfinite(value):
                raise SystemExit(f"Invalid loss row from worker {worker_index}: {identifier}")
            if identifier in losses:
                raise SystemExit(f"Duplicate candidate loss ID: {identifier}")
            losses[identifier] = row
            rows += 1
    if rows != int(shard_receipt["rows"]):
        raise SystemExit(f"Worker {worker_index} output row count changed")
    worker_receipts.append(receipt)

missing = set(ordered) - set(losses)
extra = set(losses) - set(ordered)
if missing or extra or len(losses) != len(ordered):
    raise SystemExit(
        f"Loss coverage mismatch: rows={len(losses)} missing={len(missing)} extra={len(extra)}"
    )
with output.open("x", encoding="utf-8") as handle:
    for identifier in ordered:
        handle.write(json.dumps(losses[identifier], ensure_ascii=False, allow_nan=False) + "\n")
receipt = {
    "passed": True,
    "recovery_of_job": 1541,
    "eligible_manifest": str(eligible.resolve()),
    "eligible_manifest_sha256": eligible_sha,
    "rows": len(ordered),
    "output": str(output.resolve()),
    "output_sha256": sha(output),
    "model_dir": expected_model,
    "processor_dir": expected_processor,
    "workers": worker_receipts,
}
output.with_suffix(".receipt.json").write_text(
    json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(receipt, ensure_ascii=False, indent=2))
PY

"$PYTHON" scripts/prepare_w500_adaptive_continuation.py \
  --project-root "$PROJECT_DIR" \
  --config "$CONFIG" \
  --mode freeze \
  --candidate-losses "$LOSSES" \
  --output-dir "$ARTIFACT" \
  >"$RECOVERY/freeze.log" 2>&1

if [[ "$(cat "$ARTIFACT/MANIFESTS_FROZEN")" != PASS ]]; then
  echo "Adaptive freeze marker is missing after recovery" >&2
  exit 11
fi
"$PYTHON" - "$RECOVERY" "$ARTIFACT" "$LOSSES" "$PROCESSOR" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

recovery, artifact, losses, processor = map(Path, sys.argv[1:])
preparation = json.loads((artifact / "preparation.json").read_text(encoding="utf-8"))
if not preparation.get("passed"):
    raise SystemExit("Recovered preparation receipt did not pass")
if preparation["candidate_losses"]["path"] != str(losses.resolve()):
    raise SystemExit("Freeze did not consume the recovery loss sidecar")
receipt = {
    "passed": True,
    "recovery_of_job": 1541,
    "recovery_job_id": int(os.environ["SLURM_JOB_ID"]),
    "nodes": ["gpu001", "gpu002"],
    "gpu_workers": 8,
    "audit_and_parent_shards_reused_without_mutation": True,
    "model_and_data_modified": False,
    "processor_checkpoint": str(processor.resolve()),
    "candidate_losses": str(losses.resolve()),
    "candidate_losses_sha256": hashlib.sha256(losses.read_bytes()).hexdigest(),
    "preparation": str((artifact / "preparation.json").resolve()),
    "preparation_sha256": hashlib.sha256(
        (artifact / "preparation.json").read_bytes()
    ).hexdigest(),
    "manifests_frozen": True,
}
(recovery / "recovery_receipt.json").write_text(
    json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(receipt, ensure_ascii=False, indent=2))
PY
printf 'COMPLETED\n' >"$RECOVERY/status.txt"
