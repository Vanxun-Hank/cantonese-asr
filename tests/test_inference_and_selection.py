from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cantonese_asr.io import write_jsonl
from predict import generation_kwargs, validate_inference_args
from scripts.evaluate_checkpoints import (
    named_manifest,
    prediction_command,
    resolve_best_checkpoint_name,
)
from scripts.evaluate_manifest_loss import weighted_mean_loss
from scripts.select_global_candidate import load_candidates


def test_offline_inference_uses_canonical_generation_limit() -> None:
    args = argparse.Namespace(batch_size=4, num_beams=1, generation_max_length=225)
    validate_inference_args(args)
    assert generation_kwargs(args) == {
        "language": "zh",
        "task": "transcribe",
        "num_beams": 1,
        "max_length": 225,
    }


def test_checkpoint_evaluator_forwards_the_same_generation_limit(tmp_path: Path) -> None:
    command = prediction_command(
        Path("python"),
        tmp_path / "checkpoint-1",
        tmp_path / "run",
        tmp_path,
        tmp_path / "validation.jsonl",
        tmp_path / "predictions.jsonl",
        batch_size=8,
        generation_max_length=225,
    )
    position = command.index("--generation-max-length")
    assert command[position + 1] == "225"


def test_checkpoint_evaluator_resolves_best_model_copy(tmp_path: Path) -> None:
    (tmp_path / "trainer_state.json").write_text(
        json.dumps({"best_model_checkpoint": "/server/run/checkpoint-321"}),
        encoding="utf-8",
    )
    assert resolve_best_checkpoint_name(tmp_path) == "checkpoint-321"


def test_checkpoint_evaluator_parses_named_diagnostic_manifest() -> None:
    assert named_manifest("ood_mdcc=/tmp/mdcc.jsonl") == (
        "ood_mdcc",
        Path("/tmp/mdcc.jsonl"),
    )
    with pytest.raises(argparse.ArgumentTypeError):
        named_manifest("validation=/tmp/validation.jsonl")
    with pytest.raises(argparse.ArgumentTypeError):
        named_manifest("missing-separator")


def test_teacher_forced_loss_is_weighted_by_non_padding_tokens() -> None:
    loss, tokens = weighted_mean_loss([(2.0, 2), (1.0, 6)])
    assert loss == pytest.approx(1.25)
    assert tokens == 8


@pytest.mark.parametrize("field", ["batch_size", "num_beams", "generation_max_length"])
def test_offline_inference_rejects_non_positive_settings(field: str) -> None:
    values = {"batch_size": 4, "num_beams": 1, "generation_max_length": 225}
    values[field] = 0
    with pytest.raises(SystemExit):
        validate_inference_args(argparse.Namespace(**values))


def write_diagnostic(run_dir: Path, name: str, accuracy: float, cer: float) -> None:
    metrics_dir = run_dir / "diagnostics" / name / "validation"
    metrics_dir.mkdir(parents=True)
    (run_dir / name).mkdir(parents=True)
    (metrics_dir / "metrics.json").write_text(
        json.dumps({"sentence_accuracy_tol2": accuracy, "cer": cer}),
        encoding="utf-8",
    )


def test_selector_enforces_guardrails_and_documented_tie_breaks(tmp_path: Path) -> None:
    project_root = Path(__file__).parents[1]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_diagnostic(run_dir, "checkpoint-10", 0.8218, 0.09)  # accuracy fails
    write_diagnostic(run_dir, "checkpoint-20", 0.90, 0.1164)  # CER fails
    write_diagnostic(run_dir, "checkpoint-30", 0.84, 0.10)
    write_diagnostic(run_dir, "checkpoint-40", 0.84, 0.10)
    write_diagnostic(run_dir, "checkpoint-50", 0.84, 0.10)
    write_diagnostic(run_dir, "checkpoint-60", 0.99, 0.08)  # loss missing
    write_jsonl(
        run_dir / "metrics.jsonl",
        [
            {"global_step": 10, "eval_validation_loss": 0.20},
            {"global_step": 20, "eval_validation_loss": 0.10},
            {"global_step": 30, "eval_validation_loss": 0.50},
            {"global_step": 40, "eval_validation_loss": 0.40},
            {"global_step": 50, "eval_validation_loss": 0.40},
        ],
    )
    output = tmp_path / "selection.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/select_best_checkpoint.py",
            "--run-dir",
            str(run_dir),
            "--output",
            str(output),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    # 40 beats 30 on validation loss; 40 beats 50 because the earlier step wins.
    assert report["selected"]["checkpoint"] == "checkpoint-40"
    by_name = {row["checkpoint"]: row for row in report["candidates"]}
    assert by_name["checkpoint-10"]["rejection_reasons"] == [
        "sentence_accuracy_below_guardrail"
    ]
    assert by_name["checkpoint-20"]["rejection_reasons"] == [
        "cer_above_guardrail"
    ]
    assert "validation_loss_missing" in by_name["checkpoint-60"]["rejection_reasons"]


def test_selector_resolves_and_deduplicates_best_model_copy(tmp_path: Path) -> None:
    project_root = Path(__file__).parents[1]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_diagnostic(run_dir, "checkpoint-12", 0.83, 0.10)
    write_diagnostic(run_dir, "best_model", 0.83, 0.10)
    (run_dir / "trainer_state.json").write_text(
        json.dumps({"best_model_checkpoint": "/server/run/checkpoint-12"}),
        encoding="utf-8",
    )
    write_jsonl(
        run_dir / "metrics.jsonl",
        [{"global_step": 12, "eval_validation_loss": 0.7}],
    )
    output = tmp_path / "selection.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/select_best_checkpoint.py",
            "--run-dir",
            str(run_dir),
            "--output",
            str(output),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(report["candidates"]) == 1
    assert report["selected"]["checkpoint"] == "checkpoint-12"


def test_global_selector_rejects_non_validation_selection(tmp_path: Path) -> None:
    invalid = tmp_path / "checkpoint_selection.json"
    invalid.write_text(
        json.dumps(
            {
                "selection_surface": "ood_panel",
                "selected": {"eligible": True, "model_dir": "checkpoint-1"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not validation-only"):
        load_candidates([invalid])
