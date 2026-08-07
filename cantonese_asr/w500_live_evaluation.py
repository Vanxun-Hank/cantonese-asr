"""Readiness gates for evaluating immutable W500 checkpoints while training.

The adaptive trainer publishes a checkpoint by renaming a fully populated
``*.staging`` directory.  These helpers deliberately ignore staging paths and
accept only a final directory whose state and hashes prove it was saved after a
complete Wenet -> Official source cycle.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable


STAGE_TARGETS = {
    40: (36.25, 37.5, 38.75, 40.0),
    45: (41.25, 42.5, 43.75, 45.0),
    50: (46.25, 47.5, 48.75, 50.0),
}
REQUIRED_READY_FILES = (
    "adaptive_state.json",
    "trainer_state.json",
    "source_steps.jsonl",
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "sha256sums.json",
)
HASH_GATED_FILES = (
    "adaptive_state.json",
    "source_steps.jsonl",
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
)


def stage_targets(stage: int | str) -> tuple[float, ...]:
    value = int(stage)
    try:
        return STAGE_TARGETS[value]
    except KeyError as exc:
        raise ValueError(f"Adaptive stage must be 40, 45, or 50; got {stage!r}") from exc


def checkpoint_label(target_hours: float) -> str:
    return f"checkpoint-wenet-{float(target_hours):g}h"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing JSON receipt: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON receipt {path}: {exc}") from exc


def verify_ready_checkpoint(
    checkpoint: Path,
    *,
    expected_arm: str,
    expected_target_hours: float,
    alignment_rows: int = 16,
) -> dict[str, Any]:
    """Verify a final checkpoint is immutable and safe for concurrent reading."""

    checkpoint = checkpoint.resolve()
    if checkpoint.name.endswith(".staging") or ".staging-" in checkpoint.name:
        raise ValueError(f"A staging checkpoint is never evaluation-ready: {checkpoint}")
    if not checkpoint.is_dir():
        raise ValueError(f"Checkpoint directory is not published: {checkpoint}")
    missing = [name for name in REQUIRED_READY_FILES if not (checkpoint / name).is_file()]
    if missing:
        raise ValueError(f"Published checkpoint is incomplete {checkpoint}: {missing}")

    recorded_hashes = read_json(checkpoint / "sha256sums.json")
    if not isinstance(recorded_hashes, dict):
        raise ValueError(f"sha256sums.json must contain an object: {checkpoint}")
    absent_hashes = [name for name in HASH_GATED_FILES if name not in recorded_hashes]
    if absent_hashes:
        raise ValueError(f"Checkpoint hash receipt is incomplete {checkpoint}: {absent_hashes}")
    observed_hashes = {
        name: sha256_file(checkpoint / name) for name in HASH_GATED_FILES
    }
    mismatches = {
        name: {"recorded": recorded_hashes[name], "observed": observed}
        for name, observed in observed_hashes.items()
        if recorded_hashes[name] != observed
    }
    if mismatches:
        raise ValueError(f"Checkpoint hash mismatch {checkpoint}: {mismatches}")

    state = read_json(checkpoint / "adaptive_state.json")
    target = float(expected_target_hours)
    actual_target = float(state.get("wenet_hours_target", float("nan")))
    actual_hours = float(state.get("actual_wenet_hours", float("nan")))
    if not math.isfinite(actual_target) or abs(actual_target - target) > 1e-9:
        raise ValueError(
            f"Checkpoint target mismatch {checkpoint}: expected={target}, actual={actual_target}"
        )
    if not math.isfinite(actual_hours):
        raise ValueError(f"Checkpoint has no finite actual W500 hours: {checkpoint}")
    if state.get("arm") != expected_arm or state.get("identity", {}).get("arm_name") != expected_arm:
        raise ValueError(
            f"Checkpoint arm mismatch {checkpoint}: expected={expected_arm!r}, "
            f"state={state.get('arm')!r}, identity={state.get('identity', {}).get('arm_name')!r}"
        )
    if state.get("checkpoint_phase") != "after_complete_wenet_official_cycle":
        raise ValueError(f"Checkpoint is not after a complete source cycle: {checkpoint}")
    if state.get("next_source") != "wenet":
        raise ValueError(f"Checkpoint would resume inside a source cycle: {checkpoint}")
    model_hash = observed_hashes["model.safetensors"]
    if state.get("model_safetensors_sha256") != model_hash:
        raise ValueError(f"adaptive_state model hash mismatch: {checkpoint}")
    if state.get("source_steps_sha256") != observed_hashes["source_steps.jsonl"]:
        raise ValueError(f"adaptive_state source-step hash mismatch: {checkpoint}")

    wenet_steps = int(state.get("wenet_optimizer_steps", -1))
    official_steps = int(state.get("official_optimizer_steps", -2))
    cycles = int(state.get("completed_cycles", -3))
    if wenet_steps != official_steps or cycles != wenet_steps:
        raise ValueError(
            f"Checkpoint source-cycle counts differ {checkpoint}: "
            f"wenet={wenet_steps}, official={official_steps}, cycles={cycles}"
        )
    wenet_cursor = int(state.get("wenet_sample_cursor", -1))
    official_cursor = int(state.get("official_sample_cursor", -1))
    if alignment_rows <= 0 or wenet_cursor < 0 or official_cursor < 0:
        raise ValueError(f"Invalid checkpoint cursors: {checkpoint}")
    if wenet_cursor % alignment_rows or official_cursor % alignment_rows:
        raise ValueError(
            f"Checkpoint cursors are not {alignment_rows}-row aligned: {checkpoint}"
        )
    audits = state.get("gradient_audits", {})
    for name in ("wenet_encoder_only", "official_full_sft"):
        if audits.get(name) is not True:
            raise ValueError(f"Checkpoint gradient audit {name} did not pass: {checkpoint}")

    deviation = float(state.get("target_deviation_seconds", float("nan")))
    expected_deviation = actual_hours * 3600.0 - target * 3600.0
    if not math.isfinite(deviation) or abs(deviation - expected_deviation) > 1e-5:
        raise ValueError(f"Checkpoint exposure deviation receipt mismatch: {checkpoint}")
    return {
        "passed": True,
        "checkpoint": str(checkpoint),
        "arm": expected_arm,
        "target_hours": target,
        "actual_hours": actual_hours,
        "model_safetensors_sha256": model_hash,
        "adaptive_state_sha256": observed_hashes["adaptive_state.json"],
        "source_steps_sha256": observed_hashes["source_steps.jsonl"],
        "checkpoint_phase": state["checkpoint_phase"],
        "next_source": state["next_source"],
        "wenet_cursor": wenet_cursor,
        "official_cursor": official_cursor,
        "state": state,
        "hashes": observed_hashes,
    }


def _terminal_training_failure(train_root: Path, arm_name: str) -> str | None:
    root_completion = train_root / "training_completion.json"
    if root_completion.is_file():
        value = read_json(root_completion)
        if value.get("passed") is not True:
            return f"stage training completion failed: {root_completion}"
    arm_completion = train_root / arm_name / "training_completion.json"
    if arm_completion.is_file():
        value = read_json(arm_completion)
        if value.get("passed") is not True:
            return f"arm training completion failed: {arm_completion}"
    return None


def wait_for_ready_checkpoint(
    train_root: Path,
    *,
    arm_name: str,
    target_hours: float,
    alignment_rows: int = 16,
    timeout_seconds: float = 82_800.0,
    poll_seconds: float = 20.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Poll for one atomically published checkpoint, failing closed."""

    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("timeout_seconds and poll_seconds must be positive")
    train_root = train_root.resolve()
    checkpoint = train_root / arm_name / checkpoint_label(target_hours)
    deadline = monotonic() + float(timeout_seconds)
    while True:
        if checkpoint.exists():
            # A final directory is immutable by contract.  If it is malformed,
            # do not wait for it to change and do not evaluate it.
            return verify_ready_checkpoint(
                checkpoint,
                expected_arm=arm_name,
                expected_target_hours=target_hours,
                alignment_rows=alignment_rows,
            )
        failure = _terminal_training_failure(train_root, arm_name)
        if failure:
            raise RuntimeError(failure)
        now = monotonic()
        if now >= deadline:
            raise TimeoutError(
                f"Timed out waiting for immutable checkpoint {checkpoint} after "
                f"{timeout_seconds:g}s"
            )
        sleeper(min(float(poll_seconds), max(0.0, deadline - now)))


def candidate_from_ready_checkpoint(
    ready: dict[str, Any],
    *,
    stage: int,
    arm: dict[str, Any],
) -> dict[str, Any]:
    """Build the same candidate metadata used by the serial evaluator."""

    state = ready["state"]
    checkpoint = Path(ready["checkpoint"])
    label = f"stage{int(stage)}/{ready['arm']}/{checkpoint.name}"
    return {
        "label": label,
        "checkpoint": str(checkpoint.resolve()),
        "processor_dir": str(checkpoint.resolve()),
        "model_safetensors_sha256": ready["model_safetensors_sha256"],
        "hours": float(state["actual_wenet_hours"]),
        "cumulative_wenet_seconds": float(state["cumulative_wenet_seconds"]),
        "wenet_lr": float(arm["wenet_lr"]),
        "official_lr": float(arm["official_lr"]),
        "wenet_cursor": int(state["wenet_sample_cursor"]),
        "official_cursor": int(state["official_sample_cursor"]),
        "wenet_manifest": str(Path(arm["wenet_manifest"]).resolve()),
        "adaptive_state_sha256": ready["adaptive_state_sha256"],
        "live_evaluation_readiness": {
            "checkpoint_phase": ready["checkpoint_phase"],
            "next_source": ready["next_source"],
            "model_safetensors_sha256": ready["model_safetensors_sha256"],
        },
    }


def verify_stage_training_completion(
    train_root: Path,
    *,
    arm_names: list[str],
    expected_targets: tuple[float, ...],
) -> dict[str, Any]:
    """Require both root and every arm completion after live evaluations finish."""

    train_root = train_root.resolve()
    root_path = train_root / "training_completion.json"
    root = read_json(root_path)
    if root.get("passed") is not True:
        raise ValueError(f"Stage training did not pass: {root_path}")
    arms: dict[str, Any] = {}
    expected_keys = {str(float(target)) for target in expected_targets}
    for name in arm_names:
        path = train_root / name / "training_completion.json"
        value = read_json(path)
        if value.get("passed") is not True:
            raise ValueError(f"Arm training did not pass: {path}")
        actual_keys = {key for key in value.get("checkpoints", {}) if key != "smoke"}
        if actual_keys != expected_keys:
            raise ValueError(
                f"Arm checkpoint completion mismatch {name}: "
                f"expected={sorted(expected_keys)}, actual={sorted(actual_keys)}"
            )
        arms[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    return {
        "passed": True,
        "root": {"path": str(root_path.resolve()), "sha256": sha256_file(root_path)},
        "arms": arms,
    }
