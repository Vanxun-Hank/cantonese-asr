from __future__ import annotations

import json
from pathlib import Path

from scripts.select_w500_adaptive_stage import (
    EXPECTED_ROWS,
    aligned_cursor_for_continuation,
    build_selection,
    discover_candidates,
    remap_branch_cursor,
    write_json_exclusive,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: int) -> None:
    path.write_text("".join(json.dumps({"index": index}) + "\n" for index in range(rows)))


def _surface(
    path: Path,
    *,
    name: str,
    cer: float,
    tol2: float,
    severe: int,
    insertions: int = 100,
    repeated: int = 0,
    replacement: int = 0,
    effective_max: int = 0,
    row_delta: int = 0,
) -> None:
    path.mkdir(parents=True)
    summary = {
        "surface": name,
        "rows": EXPECTED_ROWS[name] + row_delta,
        "metrics": {"cer": cer, "sentence_accuracy_tol2": tol2},
        "operations": {
            "substitutions": 10,
            "deletions": 10,
            "insertions": insertions,
        },
        "severe_error_count": severe,
        "generation": {
            "repeated_runaway_count": repeated,
            "replacement_character_count": replacement,
            "effective_max_length_count": effective_max,
        },
    }
    _write_json(path / "summary.json", summary)
    _write_json(
        path / "metrics.json",
        {
            "cer": cer,
            "sentence_accuracy_tol2": tol2,
            "diagnostics": {
                "insertions": insertions,
                "severe_error_count": severe,
            },
        },
    )
    _write_json(path / "generation.json", summary["generation"])
    (path / "top_confusions.csv").write_text(
        "type,reference,prediction,count\n", encoding="utf-8"
    )
    _write_json(path / "error_examples.json", [])
    _write_jsonl(path / "predictions.jsonl", EXPECTED_ROWS[name] + row_delta)
    _write_jsonl(path / "generation_tokens.jsonl", EXPECTED_ROWS[name] + row_delta)


def _candidate(
    root: Path,
    *,
    label: str,
    validation_cer: float,
    validation_tol2: float,
    public_tol2: float,
    public_repeated: int = 0,
    row_delta: int = 0,
    lr: float = 5e-7,
) -> Path:
    base = root / label / "D_B2_NR4_RP"
    checkpoint = root / "checkpoints" / label
    checkpoint.mkdir(parents=True, exist_ok=True)
    model_hash = (label.encode("utf-8").hex() + "0" * 64)[:64]
    _write_json(
        base / "candidate.json",
        {
            "label": label,
            "checkpoint": str(checkpoint),
            "model_safetensors_sha256": model_hash,
            "hours": 40.0,
            "wenet_lr": lr,
            "official_lr": lr / 2,
            "wenet_cursor": 1200,
            "official_cursor": 8912,
        },
    )
    _surface(
        base / "validation",
        name="validation",
        cer=validation_cer,
        tol2=validation_tol2,
        severe=17,
        row_delta=row_delta,
    )
    _surface(
        base / "public",
        name="public",
        cer=0.09,
        tol2=public_tol2,
        severe=20,
        repeated=public_repeated,
        row_delta=row_delta,
    )
    _surface(
        base / "ood",
        name="ood",
        cer=0.342,
        tol2=0.33,
        severe=500,
        row_delta=row_delta,
    )
    return base


def test_fixed_validation_ranks_and_public_ood_only_veto(tmp_path: Path) -> None:
    root = tmp_path / "evaluations"
    # A has the best eligible validation proxy but deliberately worse Public.
    _candidate(
        root,
        label="A_VALIDATION_WINNER",
        validation_cer=0.09,
        validation_tol2=0.84,
        public_tol2=0.87,
    )
    _candidate(
        root,
        label="B_PUBLIC_WINNER",
        validation_cer=0.095,
        validation_tol2=0.845,
        public_tol2=0.95,
        lr=2.5e-7,
    )
    # C would win validation but its Public runaway must make it ineligible.
    _candidate(
        root,
        label="C_VETOED",
        validation_cer=0.08,
        validation_tol2=0.85,
        public_tol2=0.95,
        public_repeated=1,
    )
    rows = discover_candidates(root)
    manifest = tmp_path / "balanced45.jsonl"
    manifest.write_text("\n", encoding="utf-8")
    selection, branches = build_selection(
        rows,
        target_hours=40.0,
        balanced_next_manifest=manifest,
    )

    assert selection["ranking_surface"] == "fixed_validation_only"
    assert selection["public_ood_used_for_ranking"] is False
    assert selection["ranking"] == ["A_VALIDATION_WINNER", "B_PUBLIC_WINNER"]
    vetoed = next(row for row in selection["ineligible"] if row["label"] == "C_VETOED")
    assert "public_repeated_runaway" in vetoed["guardrail_failures"]
    assert len(branches) == 4
    assert [branch["lr_mode"] for branch in branches] == [
        "constant",
        "half",
        "constant",
        "half",
    ]
    assert len({branch["parent_model_sha256"] for branch in branches}) == 2


def test_incomplete_prediction_or_token_rows_are_vetoed(tmp_path: Path) -> None:
    root = tmp_path / "evaluations"
    _candidate(
        root,
        label="COMPLETE",
        validation_cer=0.09,
        validation_tol2=0.84,
        public_tol2=0.88,
    )
    _candidate(
        root,
        label="INCOMPLETE",
        validation_cer=0.08,
        validation_tol2=0.85,
        public_tol2=0.95,
        row_delta=-1,
    )
    rows = discover_candidates(root)
    incomplete = next(row for row in rows if row["label"] == "INCOMPLETE")
    assert incomplete["complete"] is False
    assert any(
        "predictions.jsonl_rows" in error
        for error in incomplete["diagnostic_evidence"]["validation"]["errors"]
    )


def test_missing_error_analysis_artifact_is_vetoed(tmp_path: Path) -> None:
    root = tmp_path / "evaluations"
    base = _candidate(
        root,
        label="MISSING_CONFUSIONS",
        validation_cer=0.09,
        validation_tol2=0.84,
        public_tol2=0.88,
    )
    (base / "public" / "top_confusions.csv").unlink()
    row = discover_candidates(root)[0]
    assert row["complete"] is False
    assert "missing_top_confusions.csv" in row["diagnostic_evidence"]["public"]["errors"]


def test_stage50_selects_one_global_winner_and_emits_no_branches(tmp_path: Path) -> None:
    root = tmp_path / "evaluations"
    _candidate(
        root,
        label="EARLY_BEST",
        validation_cer=0.085,
        validation_tol2=0.845,
        public_tol2=0.88,
    )
    late = _candidate(
        root,
        label="LATE",
        validation_cer=0.09,
        validation_tol2=0.84,
        public_tol2=0.93,
    )
    metadata_path = late / "candidate.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["hours"] = 50.0
    _write_json(metadata_path, metadata)

    selection, branches = build_selection(
        discover_candidates(root),
        target_hours=50.0,
        balanced_next_manifest=None,
    )
    assert selection["selected"]["winner"]["label"] == "EARLY_BEST"
    assert branches == []
    assert selection["branches"] == []


def test_selection_and_arm_files_use_stable_schema(tmp_path: Path) -> None:
    output = tmp_path / "selection.json"
    value = {
        "candidates": [],
        "selected": {"top2": [], "winner": None},
        "branches": [],
    }
    write_json_exclusive(output, value)
    assert json.loads(output.read_text()) == value
    try:
        write_json_exclusive(output, value)
    except SystemExit as error:
        assert "Refusing to overwrite" in str(error)
    else:  # pragma: no cover
        raise AssertionError("immutable selection output was overwritten")


def test_oldstream_parent_cursor_is_remapped_by_balanced_duration(tmp_path: Path) -> None:
    manifest = tmp_path / "balanced45.jsonl"
    manifest.write_text(
        "".join(
            json.dumps({"id": str(index), "duration": 60.0}) + "\n"
            for index in range(200)
        ),
        encoding="utf-8",
    )
    parent = {
        "label": "OLDSTREAM_PARENT",
        "wenet_manifest": "/oldstream.jsonl",
        "wenet_cursor": 64,
        "cumulative_wenet_seconds": 36.25 * 3600,
    }
    receipt = remap_branch_cursor(
        parent,
        balanced_manifest=manifest,
        root_wenet_seconds=35.0 * 3600,
        alignment=16,
    )
    # 1.25h is 4500 seconds; the nearest 16-row boundary is 80 minutes.
    assert receipt["mode"] == "remap_to_balanced_stream"
    assert receipt["source_cursor"] == 64
    assert receipt["target_cursor"] == 80
    assert receipt["actual_prefix_seconds"] == 4800.0
    assert receipt["deviation_seconds"] == 300.0


def test_balanced_parent_reuses_its_exact_cursor(tmp_path: Path) -> None:
    manifest = tmp_path / "balanced50.jsonl"
    manifest.write_text(
        "".join(
            json.dumps({"id": str(index), "duration": 30.0}) + "\n"
            for index in range(320)
        ),
        encoding="utf-8",
    )
    parent = {
        "label": "BALANCED_PARENT",
        "wenet_manifest": str(manifest),
        "wenet_cursor": 160,
        "cumulative_wenet_seconds": 45.0 * 3600,
    }
    receipt = remap_branch_cursor(
        parent,
        balanced_manifest=manifest,
        root_wenet_seconds=35.0 * 3600,
        alignment=16,
    )
    assert receipt["mode"] == "reuse_same_balanced_stream"
    assert receipt["source_cursor"] == receipt["target_cursor"] == 160
    assert receipt["actual_prefix_seconds"] == 4800.0


def test_aligned_cursor_accepts_zero_prefix_for_root_branch(tmp_path: Path) -> None:
    manifest = tmp_path / "balanced.jsonl"
    manifest.write_text(
        "".join(
            json.dumps({"id": str(index), "duration": 10.0}) + "\n"
            for index in range(32)
        ),
        encoding="utf-8",
    )
    receipt = aligned_cursor_for_continuation(
        manifest, target_seconds=0.0, alignment=16
    )
    assert receipt["cursor"] == 0
    assert receipt["actual_prefix_seconds"] == 0.0
