from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from scripts.finalize_w500_adaptive_continuation import (
    PACKAGE_LABELS,
    EXPECTED_ROWS,
    finalize,
    validate_outputs_pre,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_outputs_pre(path: Path) -> None:
    for surface, count in EXPECTED_ROWS.items():
        directory = path / surface
        directory.mkdir(parents=True)
        (directory / "metrics.json").write_text(
            json.dumps({"rows": count, "surface": surface}) + "\n"
        )
        (directory / "top_confusions.csv").write_text("reference,prediction,count\n")
        (directory / "error_examples.json").write_text("[]\n")
        (directory / "generation.json").write_text("{}\n")
        lines = "".join(json.dumps({"index": index}) + "\n" for index in range(count))
        (directory / "predictions.jsonl").write_text(lines)
        (directory / "generation_tokens.jsonl").write_text(lines)


def surface_row(*, cer: float, tol2: float, public_tol2: float = 0.88) -> dict[str, object]:
    return {
        "validation": {"cer": cer, "tol2": tol2, "severe": 10},
        "public": {
            "cer": 0.09,
            "tol2": public_tol2,
            "severe": 20,
            "insertions": 100,
            "repeated_runaway": 0,
            "replacement": 0,
            "effective_max": 0,
        },
        "ood": {"cer": 0.34, "tol2": 0.33, "severe": 100},
    }


def write_model(path: Path, label: str, outputs_pre: Path) -> dict[str, object]:
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(("weight:" + label).encode())
    return {
        "checkpoint": str(path),
        "weight_sha256": sha256(path / "model.safetensors"),
        "outputs_pre": str(outputs_pre),
        "complete": True,
    }


def build_fixture(tmp_path: Path) -> argparse.Namespace:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment": "test adaptive W500",
                "start": {
                    "checkpoint": "/baseline35",
                    "weight_sha256": "a" * 64,
                    "actual_wenet_hours": 35.101,
                },
                "selection": {
                    "ranking_surface": "fixed_validation_only",
                    "proxy_cer_weight": 70.0,
                    "proxy_tol2_weight": 20.0,
                    "proxy_tie": 0.05,
                    "guardrails": {
                        "validation_tol2_min": 0.836880,
                        "public_tol2_min": 0.869211,
                        "public_insertions_max": 200,
                        "public_severe_max": 25,
                        "public_repeated_runaway_max": 0,
                        "public_replacement_max": 10,
                        "public_effective_max_length_max": 0,
                        "ood_tol2_min": 0.325,
                        "ood_cer_max": 0.346632,
                    },
                },
            }
        )
        + "\n"
    )
    preparation = tmp_path / "preparation.json"
    preparation.write_text('{"passed": true}\n')
    # Production final_candidates explicitly point outputs_pre at this evaluator
    # decoder root; the directory is not required to be literally named outputs_pre.
    shared_outputs = tmp_path / "evaluation" / "shared" / "historical_current"
    write_outputs_pre(shared_outputs)

    stage_root = tmp_path / "stages"
    stage_root.mkdir()
    stage_rows = []
    for index, (cer, tol2, hours) in enumerate(
        ((0.10, 0.84, 36.25), (0.09, 0.845, 40.0), (0.085, 0.848, 45.0), (0.08, 0.85, 50.0)),
        start=1,
    ):
        row = {
            "label": f"STAGE_{index}",
            "hours": hours,
            **write_model(tmp_path / "models" / f"stage-{index}", f"stage-{index}", shared_outputs),
            **surface_row(cer=cer, tol2=tol2),
        }
        stage_rows.append(row)
    (stage_root / "stage50_selection.json").write_text(
        json.dumps(
            {
                "ranking_surface": "fixed_validation_only",
                "public_ood_used_for_ranking": False,
                "candidates": stage_rows,
                "selected": stage_rows[-1],
                "branches": [],
            }
        )
        + "\n"
    )

    interpolation_root = tmp_path / "interpolation"
    interpolation_root.mkdir()
    candidates = []
    raw = {
        "label": "RAW_WINNER",
        "hours": 50.0,
        **stage_rows[-1],
    }
    raw["label"] = "RAW_WINNER"
    candidates.append(raw)
    final_metrics = {
        "MIX15": (0.081, 0.849, 0.88),
        "MIX30": (0.082, 0.848, 0.89),
        "MIX45": (0.083, 0.847, 0.90),
        # Deliberately best Public, but worst fixed validation.
        "MIX60": (0.09, 0.84, 0.99),
    }
    for label, (cer, tol2, public_tol2) in final_metrics.items():
        model = tmp_path / "models" / label
        row = {
            "label": label,
            "hours": 50.0,
            **write_model(model, label, shared_outputs),
            **surface_row(cer=cer, tol2=tol2, public_tol2=public_tol2),
        }
        receipt = {
            "passed": True,
            "name": label,
            "alpha": int(label[-2:]) / 100,
            "stable_weight": int(label[-2:]) / 100,
            "output_model_sha256": row["weight_sha256"],
            "encoder_tensor_sha256": "e" * 64,
            "endpoints": {
                "acoustic_model_sha256": stage_rows[-1]["weight_sha256"]
            },
            "immutable_file_hashes": {
                "config.json": "c" * 64,
                "generation_config.json": "g" * 64,
                "predict.py": "p" * 64,
            },
            "persisted_tensor_audit": {
                "checks": {
                    "encoder_exact": True,
                    "alpha_0_reproduces_acoustic_endpoint": True,
                    "alpha_1_reproduces_stable_decoder_with_acoustic_encoder": True,
                    "sampled_formula_checks_passed": True,
                },
                "digests": {
                    "acoustic_encoder": "e" * 64,
                    "mixed_encoder": "e" * 64,
                },
            },
        }
        (model / "interpolation_receipt.json").write_text(json.dumps(receipt) + "\n")
        candidates.append(row)
    (interpolation_root / "final_candidates.json").write_text(
        json.dumps({"candidates": candidates}) + "\n"
    )

    final = tmp_path / "final"
    return argparse.Namespace(
        config=config_path,
        preparation=preparation,
        stage_root=stage_root,
        interpolation_root=interpolation_root,
        matrix=final / "matrix.json",
        curves=final / "curves.csv",
        tasks=final / "tasks.json",
        report=final / "report.md",
    )


def test_finalizer_is_fixed_validation_only_and_emits_exact_five_tasks(
    tmp_path: Path,
) -> None:
    args = build_fixture(tmp_path)
    completion = finalize(args)
    matrix = json.loads(args.matrix.read_text())
    tasks = json.loads(args.tasks.read_text())

    assert completion["passed"] is True
    assert matrix["ranking_surface"] == "fixed_validation_only"
    assert matrix["public_ood_used_for_ranking"] is False
    assert [row["label"] for row in matrix["package_candidates"]] == list(PACKAGE_LABELS)
    assert [row["label"] for row in tasks["tasks"]] == list(PACKAGE_LABELS)
    assert matrix["final_ranking"][0]["label"] == "RAW_WINNER"
    assert len(matrix["all_checkpoint_curves"]) == 4
    assert (args.matrix.parent / "SHA256SUMS").is_file()
    assert (args.matrix.parent / "EXPERIMENT_COMPLETE").is_file()
    assert "Public and OOD are catastrophic vetoes" in args.report.read_text()


def test_outputs_pre_requires_exact_prediction_and_token_rows(tmp_path: Path) -> None:
    outputs_pre = tmp_path / "outputs_pre"
    write_outputs_pre(outputs_pre)
    audit = validate_outputs_pre(outputs_pre)
    assert audit["surfaces"]["validation"]["prediction_rows"] == 702

    tokens = outputs_pre / "public" / "generation_tokens.jsonl"
    tokens.write_text("".join(tokens.read_text().splitlines(keepends=True)[:-1]))
    with pytest.raises(ValueError, match="Incomplete public rows"):
        validate_outputs_pre(outputs_pre)


def test_finalizer_excludes_public_ood_vetoed_package_candidate(tmp_path: Path) -> None:
    args = build_fixture(tmp_path)
    payload_path = args.interpolation_root / "final_candidates.json"
    payload = json.loads(payload_path.read_text())
    mix30 = next(row for row in payload["candidates"] if row["label"] == "MIX30")
    mix30["public"]["repeated_runaway"] = 1
    payload_path.write_text(json.dumps(payload) + "\n")

    completion = finalize(args)
    matrix = json.loads(args.matrix.read_text())
    tasks = json.loads(args.tasks.read_text())
    assert completion["passed"] is True
    assert "MIX30" not in {row["label"] for row in tasks["tasks"]}
    vetoed = next(row for row in matrix["all_final_candidates"] if row["label"] == "MIX30")
    assert vetoed["guardrail_failures"] == ["public_repeated_runaway"]
    assert vetoed["eligible_for_packaging"] is False
    assert (args.matrix.parent / "EXPERIMENT_COMPLETE").is_file()
