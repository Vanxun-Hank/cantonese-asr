#!/usr/bin/env python3
"""Audit data, model-selection, reporting, and submission evidence."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cantonese_asr.io import read_jsonl


WEIGHT_NAMES = {
    "model.safetensors",
    "pytorch_model.bin",
    "model.pt",
    "model.pth",
    "model.ckpt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, default=Path("artifacts/manifests"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--experiment-report",
        type=Path,
        default=Path("artifacts/reports/experiments/all"),
    )
    parser.add_argument("--submission-zip", type=Path, required=True)
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("artifacts/reports/submission_acceptance.json"),
    )
    parser.add_argument("--required-improvement", type=float, default=0.02)
    parser.add_argument("--max-cer", type=float, default=0.1163)
    parser.add_argument("--min-sentence-accuracy", type=float, default=0.8219)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def result(passed: bool, evidence: Any, requirement: str) -> dict[str, Any]:
    return {"passed": bool(passed), "requirement": requirement, "evidence": evidence}


def main() -> None:
    args = parse_args()
    checks: dict[str, dict[str, Any]] = {}

    data_report_path = args.manifests / "data_report.json"
    if data_report_path.is_file():
        data_report = load_json(data_report_path)
        counts = data_report.get("counts", {})
        leakage = data_report.get("leakage_checks", {})
        data_ok = (
            counts.get("train", 0) > 0
            and counts.get("validation", 0) > 0
            and leakage.get("normalized_text_overlap") == 0
            and leakage.get("audio_path_overlap") == 0
        )
        checks["data_qc"] = result(
            data_ok,
            {"counts": counts, "leakage_checks": leakage},
            "Valid train/validation manifests with zero text/audio overlap",
        )
    else:
        checks["data_qc"] = result(False, str(data_report_path), "Data-QC report exists")

    public_refs = []
    training_ids: set[str] = set()
    for name in ("train.jsonl", "validation.jsonl", "train_probe.jsonl"):
        path = args.manifests / name
        if path.is_file():
            rows = read_jsonl(path)
            training_ids.update(str(row.get("id")) for row in rows)
            public_refs.extend(
                row.get("audio_path")
                for row in rows
                if "test_audio" in str(row.get("audio_path", ""))
                or "template_pre" in str(row.get("audio_path", ""))
            )
    public_manifest_path = args.manifests / "public_excluded.jsonl"
    public_rows = (
        read_jsonl(public_manifest_path) if public_manifest_path.is_file() else []
    )
    public_ids = {str(row.get("id")) for row in public_rows}
    report_public_count = (
        data_report.get("counts", {}).get("public_excluded", 0)
        if data_report_path.is_file()
        else 0
    )
    report_public_overlap = (
        data_report.get("leakage_checks", {}).get("public_excluded_id_overlap")
        if data_report_path.is_file()
        else None
    )
    public_ok = (
        not public_refs
        and len(public_rows) == 1900
        and report_public_count == 1900
        and report_public_overlap == 0
        and not (training_ids & public_ids)
    )
    checks["public_test_exclusion"] = result(
        public_ok,
        {
            "path_violations": public_refs[:10],
            "public_excluded_rows": len(public_rows),
            "report_public_excluded": report_public_count,
            "id_overlap": len(training_ids & public_ids),
        },
        "Public test/template samples are absent from train, validation and train-probe",
    )

    ood_report_path = args.manifests / "ood" / "v1" / "data_report.json"
    ood_report = load_json(ood_report_path) if ood_report_path.is_file() else None
    ood_checks = ood_report.get("leakage_checks", {}) if ood_report else {}
    ood_groups = ood_report.get("panel", {}).get("groups", {}) if ood_report else {}
    ood_panel_rows = (
        ood_report.get("panel", {}).get("total", {}).get("rows", 0)
        if ood_report
        else 0
    )
    ood_ok = bool(
        ood_report
        and ood_panel_rows > 0
        and ood_panel_rows <= 2000
        and len(ood_groups) == 4
        and not any(ood_checks.values())
    )
    checks["ood_isolation"] = result(
        ood_ok,
        {
            "report": str(ood_report_path),
            "panel_rows": ood_panel_rows,
            "groups": sorted(ood_groups),
            "leakage_checks": ood_checks,
        },
        "Fixed OOD panel covers four held-out publisher splits with zero leakage",
    )

    baseline_path = args.outputs / "zero-shot" / "validation" / "metrics.json"
    baseline = load_json(baseline_path) if baseline_path.is_file() else None
    checks["zero_shot"] = result(
        baseline is not None,
        baseline or str(baseline_path),
        "Zero-shot internal-validation metrics exist",
    )

    smoke_path = args.outputs / "smoke-overfit" / "metrics.jsonl"
    smoke_logs = read_jsonl(smoke_path) if smoke_path.is_file() else []
    smoke_losses = [
        float(row["loss"]) for row in smoke_logs if row.get("loss") is not None
    ]
    smoke_ok = len(smoke_losses) >= 2 and min(smoke_losses[-3:]) < smoke_losses[0]
    checks["smoke_overfit"] = result(
        smoke_ok,
        {"num_loss_points": len(smoke_losses), "first": smoke_losses[:1], "last": smoke_losses[-3:]},
        "Smoke run has multiple loss points and later loss is lower than initial loss",
    )

    grid_runs = sorted((args.outputs / "grid-round1").glob("*/metrics.jsonl"))
    grid_trials = {path.parent.name for path in grid_runs}
    checks["four_grid_trials"] = result(
        len(grid_trials) >= 4,
        sorted(grid_trials),
        "At least four independent round-one trial logs exist",
    )

    scoreboard_path = args.experiment_report / "scoreboard.json"
    scoreboard = json.loads(scoreboard_path.read_text(encoding="utf-8")) if scoreboard_path.is_file() else []
    trained = [
        row
        for row in scoreboard
        if row.get("trial") not in {"zero-shot", "smoke-overfit"}
        and not str(row.get("trial", "")).startswith("mem-")
    ]
    best = trained[0] if trained else None
    if baseline and best:
        accuracy_gain = float(best["sentence_accuracy_tol2"]) - float(
            baseline["sentence_accuracy_tol2"]
        )
        cer_regression = float(best["cer"]) - float(baseline["cer"])
        performance_ok = accuracy_gain >= args.required_improvement and cer_regression <= 0
        performance_evidence = {
            "baseline": baseline,
            "best": best,
            "accuracy_absolute_gain": accuracy_gain,
            "cer_change": cer_regression,
        }
    else:
        performance_ok = False
        performance_evidence = {
            "baseline_available": baseline is not None,
            "trained_scoreboard_rows": len(trained),
        }
    checks["performance"] = result(
        performance_ok,
        performance_evidence,
        "Best validation accuracy improves >=2 points and CER does not regress",
    )

    all_selection_paths = sorted(args.outputs.glob("**/checkpoint_selection.json"))
    selection_paths = [
        path
        for path in all_selection_paths
        if load_json(path).get("selection_surface") == "internal_validation_only"
    ]
    selection_evidence = []
    selection_ok = len(selection_paths) >= 5
    for path in selection_paths:
        selection = load_json(path)
        selected = selection.get("selected")
        valid = bool(
            selected
            and selected.get("eligible") is True
            and float(selected["cer"]) <= args.max_cer
            and float(selected["sentence_accuracy_tol2"])
            >= args.min_sentence_accuracy
            and selected.get("validation_loss") is not None
        )
        selection_ok = selection_ok and valid
        selection_evidence.append(
            {"path": str(path), "valid": valid, "selected": selected}
        )
    checks["guarded_checkpoint_selection"] = result(
        selection_ok,
        selection_evidence,
        "Checkpoint selection enforces accuracy/CER guardrails and true validation loss",
    )

    required_reports = [
        "scoreboard.csv",
        "scoreboard.json",
        "report.html",
        "training_curves.png",
        "diagnostic_curves.png",
        "trial_comparison.png",
        "data_distribution.png",
        "scene_comparison.png",
        "diagnostic_scoreboard.csv",
        "diagnostic_scoreboard.json",
        "ood_data_distribution.png",
        "ood_comparison.png",
    ]
    missing_reports = [
        name
        for name in required_reports
        if not (args.experiment_report / name).is_file()
        or (args.experiment_report / name).stat().st_size == 0
    ]
    checks["reusable_reports"] = result(
        not missing_reports,
        {"report_dir": str(args.experiment_report), "missing": missing_reports},
        "Reusable PNG/CSV/JSON/HTML experiment artifacts exist",
    )

    submission_evidence: dict[str, Any] = {"path": str(args.submission_zip)}
    submission_ok = False
    if args.submission_zip.is_file():
        with zipfile.ZipFile(args.submission_zip) as archive:
            names = archive.namelist()
            weights = [name for name in names if name in WEIGHT_NAMES]
            flat = all("/" not in name and "\\" not in name for name in names)
            corrupt = archive.testzip()
        submission_evidence.update(
            {"files": names, "weights": weights, "flat": flat, "corrupt_member": corrupt}
        )
        submission_ok = (
            flat
            and corrupt is None
            and weights == ["model.safetensors"]
            and {"predict.py", "config.json", "preprocessor_config.json"} <= set(names)
        )
    checks["submission_structure"] = result(
        submission_ok,
        submission_evidence,
        "Flat offline submission contains exactly one model.safetensors and required files",
    )

    diagnostics = list(args.outputs.glob("**/diagnostics/*/validation/scene_metrics.csv"))
    confusions = list(args.outputs.glob("**/diagnostics/*/validation/top_confusions.csv"))
    errors = list(args.outputs.glob("**/diagnostics/*/validation/error_examples.json"))
    checks["error_diagnostics"] = result(
        bool(diagnostics and confusions and errors),
        {
            "scene_reports": len(diagnostics),
            "confusion_reports": len(confusions),
            "error_reports": len(errors),
        },
        "Per-checkpoint scene metrics, character confusions and error examples exist",
    )

    ood_metrics = list(args.outputs.glob("**/diagnostics/*/ood_panel/metrics.json"))
    ood_losses = list(
        args.outputs.glob("**/diagnostics/*/ood_panel/teacher_forced_loss.json")
    )
    checks["ood_checkpoint_diagnostics"] = result(
        bool(ood_metrics and len(ood_metrics) == len(ood_losses)),
        {
            "generation_metrics": len(ood_metrics),
            "teacher_forced_losses": len(ood_losses),
        },
        "Each generated OOD checkpoint report has a teacher-forced loss report",
    )

    passed = all(check["passed"] for check in checks.values())
    report = {"passed": passed, "checks": checks}
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
