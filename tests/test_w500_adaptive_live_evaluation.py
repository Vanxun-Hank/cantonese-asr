from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cantonese_asr.w500_live_evaluation import (
    HASH_GATED_FILES,
    REQUIRED_READY_FILES,
    candidate_from_ready_checkpoint,
    checkpoint_label,
    stage_targets,
    verify_ready_checkpoint,
    verify_stage_training_completion,
    wait_for_ready_checkpoint,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _ready_checkpoint(root: Path, *, arm: str, target: float) -> Path:
    checkpoint = root / arm / checkpoint_label(target)
    checkpoint.mkdir(parents=True)
    state = {
        "arm": arm,
        "identity": {"arm_name": arm},
        "checkpoint_phase": "after_complete_wenet_official_cycle",
        "next_source": "wenet",
        "wenet_hours_target": target,
        "actual_wenet_hours": target + 0.001,
        "target_deviation_seconds": 3.6,
        "cumulative_wenet_seconds": (target + 0.001) * 3600,
        "wenet_optimizer_steps": 8,
        "official_optimizer_steps": 8,
        "completed_cycles": 8,
        "wenet_sample_cursor": 128,
        "official_sample_cursor": 7840,
        "gradient_audits": {
            "wenet_encoder_only": True,
            "post_official_wenet_decoder_and_moments_unchanged": True,
            "official_full_sft": True,
        },
    }
    for name in REQUIRED_READY_FILES:
        if name in {"adaptive_state.json", "sha256sums.json"}:
            continue
        (checkpoint / name).write_bytes((name + "\n").encode())
    state["model_safetensors_sha256"] = _sha(checkpoint / "model.safetensors")
    state["source_steps_sha256"] = _sha(checkpoint / "source_steps.jsonl")
    _write_json(checkpoint / "adaptive_state.json", state)
    _write_json(
        checkpoint / "sha256sums.json",
        {name: _sha(checkpoint / name) for name in HASH_GATED_FILES},
    )
    return checkpoint


def test_stage_targets_are_the_frozen_tournament_grid() -> None:
    assert stage_targets(40) == (36.25, 37.5, 38.75, 40.0)
    assert stage_targets("45") == (41.25, 42.5, 43.75, 45.0)
    assert stage_targets(50) == (46.25, 47.5, 48.75, 50.0)
    with pytest.raises(ValueError, match="40, 45, or 50"):
        stage_targets(55)


def test_ready_gate_requires_atomic_final_name_hashes_and_complete_cycle(
    tmp_path: Path,
) -> None:
    checkpoint = _ready_checkpoint(tmp_path, arm="ARM_A", target=36.25)
    ready = verify_ready_checkpoint(
        checkpoint, expected_arm="ARM_A", expected_target_hours=36.25
    )
    assert ready["passed"] is True
    assert ready["checkpoint_phase"] == "after_complete_wenet_official_cycle"
    assert ready["next_source"] == "wenet"

    (checkpoint / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_ready_checkpoint(
            checkpoint, expected_arm="ARM_A", expected_target_hours=36.25
        )


def test_waiter_ignores_staging_directory_and_fails_on_terminal_training_error(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "ARM_A" / (checkpoint_label(36.25) + ".staging")
    staging.mkdir(parents=True)
    ticks = iter((0.0, 2.0))
    with pytest.raises(TimeoutError, match="Timed out"):
        wait_for_ready_checkpoint(
            tmp_path,
            arm_name="ARM_A",
            target_hours=36.25,
            timeout_seconds=1.0,
            poll_seconds=0.1,
            monotonic=lambda: next(ticks),
            sleeper=lambda _: None,
        )

    _write_json(tmp_path / "training_completion.json", {"passed": False})
    ticks = iter((0.0, 0.1))
    with pytest.raises(RuntimeError, match="stage training completion failed"):
        wait_for_ready_checkpoint(
            tmp_path,
            arm_name="ARM_A",
            target_hours=36.25,
            timeout_seconds=1.0,
            poll_seconds=0.1,
            monotonic=lambda: next(ticks),
            sleeper=lambda _: None,
        )


def test_ready_candidate_matches_serial_evaluator_metadata(tmp_path: Path) -> None:
    checkpoint = _ready_checkpoint(tmp_path, arm="ARM_A", target=36.25)
    ready = verify_ready_checkpoint(
        checkpoint, expected_arm="ARM_A", expected_target_hours=36.25
    )
    candidate = candidate_from_ready_checkpoint(
        ready,
        stage=40,
        arm={
            "wenet_lr": 5e-7,
            "official_lr": 2.5e-7,
            "wenet_manifest": str(tmp_path / "balanced.jsonl"),
        },
    )
    assert candidate["label"] == "stage40/ARM_A/checkpoint-wenet-36.25h"
    assert candidate["model_safetensors_sha256"] == ready["model_safetensors_sha256"]
    assert candidate["hours"] == pytest.approx(36.251)
    assert candidate["live_evaluation_readiness"]["next_source"] == "wenet"


def test_stage_completion_requires_root_and_every_arm_checkpoint_receipt(
    tmp_path: Path,
) -> None:
    targets = stage_targets(40)
    names = [f"ARM_{index}" for index in range(4)]
    root_rows = {}
    for name in names:
        path = tmp_path / name
        path.mkdir()
        completion = {
            "passed": True,
            "checkpoints": {str(target): {"path": "unused"} for target in targets},
        }
        _write_json(path / "training_completion.json", completion)
        root_rows[name] = {"passed": True}
    _write_json(tmp_path / "training_completion.json", {"passed": True, "arms": root_rows})

    result = verify_stage_training_completion(
        tmp_path, arm_names=names, expected_targets=targets
    )
    assert result["passed"] is True
    assert set(result["arms"]) == set(names)

    value = json.loads((tmp_path / names[0] / "training_completion.json").read_text())
    value["checkpoints"].pop("40.0")
    _write_json(tmp_path / names[0] / "training_completion.json", value)
    with pytest.raises(ValueError, match="checkpoint completion mismatch"):
        verify_stage_training_completion(
            tmp_path, arm_names=names, expected_targets=targets
        )
