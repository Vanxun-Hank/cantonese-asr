#!/usr/bin/env python3
"""Generate repeatable PNG/CSV/JSON/HTML reports from all experiment logs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cantonese_asr.io import read_jsonl


def configure_fonts() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in (
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "Droid Sans Fallback",
        "Arial Unicode MS",
    ):
        if candidate in available:
            matplotlib.rcParams["font.family"] = [candidate, "DejaVu Sans"]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


configure_fonts()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--data-report", type=Path)
    parser.add_argument("--ood-data-report", type=Path)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def collect_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("metrics.jsonl")):
        for row in read_jsonl(path):
            row.setdefault("trial", path.parent.name)
            row["metrics_path"] = str(path)
            records.append(row)
    return records


def merge_epoch_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, float, int], dict[str, Any]] = {}
    for row in records:
        epoch = finite(row.get("epoch"))
        if epoch is None:
            continue
        key = (
            str(row.get("trial", "unknown")),
            round(epoch, 6),
            int(row.get("global_step") or 0),
        )
        merged.setdefault(key, {}).update(row)
    return list(merged.values())


def values(records: Iterable[dict[str, Any]], key: str) -> list[tuple[float, float, str]]:
    points = []
    for row in records:
        y_value = finite(row.get(key))
        x_value = finite(row.get("epoch"))
        if y_value is not None and x_value is not None:
            points.append((x_value, y_value, str(row.get("trial", "unknown"))))
    return points


def plot_by_trial(
    axis: Any,
    points: list[tuple[float, float, str]],
    title: str,
    ylabel: str,
) -> None:
    grouped: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)
    for x_value, y_value, trial in points:
        grouped[trial].append((x_value, y_value))
    for trial, trial_points in sorted(grouped.items()):
        trial_points.sort()
        axis.plot(
            [point[0] for point in trial_points],
            [point[1] for point in trial_points],
            marker="o",
            linewidth=1.6,
            label=trial,
        )
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    if grouped:
        axis.legend(fontsize=7)


def build_scoreboard(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    accuracy_keys = (
        "eval_validation_sentence_accuracy_tol2",
        "eval_sentence_accuracy_tol2",
        "final_validation_sentence_accuracy_tol2",
    )
    for row in merge_epoch_records(records):
        accuracy = next(
            (finite(row.get(key)) for key in accuracy_keys if finite(row.get(key)) is not None),
            None,
        )
        if accuracy is not None:
            enriched = dict(row)
            enriched["_accuracy"] = accuracy
            candidates[str(row.get("trial", "unknown"))].append(enriched)

    scoreboard = []
    for trial, trial_rows in sorted(candidates.items()):
        if trial == "zero-shot":
            stage = "baseline"
        elif trial == "smoke-overfit" or trial.startswith("mem-"):
            stage = "diagnostic"
        elif trial.startswith("grid-"):
            stage = "grid"
        elif trial.startswith("batch-"):
            stage = "batch"
        elif trial.startswith("final-"):
            stage = "final"
        else:
            stage = "experiment"
        def ranking(row: dict[str, Any]) -> tuple[float, float]:
            cer = finite(row.get("eval_validation_cer"))
            if cer is None:
                cer = finite(row.get("eval_cer"))
            return row["_accuracy"], -(cer if cer is not None else float("inf"))

        best = max(
            trial_rows,
            key=ranking,
        )
        scoreboard.append(
            {
                "trial": trial,
                "stage": stage,
                "comparable_validation": stage not in {"diagnostic"},
                "best_epoch": finite(best.get("epoch")),
                "global_step": best.get("global_step"),
                "sentence_accuracy_tol2": best["_accuracy"],
                "cer": finite(best.get("eval_validation_cer"))
                if finite(best.get("eval_validation_cer")) is not None
                else finite(best.get("eval_cer")),
                "validation_loss": finite(best.get("eval_validation_loss"))
                if finite(best.get("eval_validation_loss")) is not None
                else finite(best.get("eval_loss")),
                "train_probe_accuracy": finite(
                    best.get("eval_train_probe_sentence_accuracy_tol2")
                ),
                "gpu_peak_allocated_gb": finite(best.get("gpu_peak_allocated_gb")),
                "elapsed_seconds": finite(best.get("elapsed_seconds")),
                "slurm_job_id": best.get("slurm_job_id"),
                "metrics_path": best.get("metrics_path"),
            }
        )
    scoreboard.sort(
        key=lambda row: (
            row["stage"] == "diagnostic",
            row["stage"] == "baseline",
            -(row["sentence_accuracy_tol2"] or 0),
            row["cer"] if row["cer"] is not None else float("inf"),
        )
    )
    return scoreboard


def write_scoreboard(report_dir: Path, scoreboard: list[dict[str, Any]]) -> None:
    (report_dir / "scoreboard.json").write_text(
        json.dumps(scoreboard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fieldnames = list(scoreboard[0]) if scoreboard else [
        "trial",
        "best_epoch",
        "sentence_accuracy_tol2",
        "cer",
    ]
    with (report_dir / "scoreboard.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scoreboard)


def collect_diagnostic_scoreboard(outputs_root: Path) -> list[dict[str, Any]]:
    """Collect independent generation and teacher-forced diagnostics."""
    records = []
    indexes: dict[Path, dict[Path, dict[str, Any]]] = {}
    for index_path in outputs_root.glob("**/diagnostics/index.json"):
        run_dir = index_path.parents[1].resolve()
        indexes[run_dir] = {
            Path(str(row["report_dir"])).resolve(): row
            for row in json.loads(index_path.read_text(encoding="utf-8"))
        }
    for metrics_path in sorted(
        outputs_root.glob("**/diagnostics/*/*/metrics.json")
    ):
        split_dir = metrics_path.parent
        checkpoint_dir = split_dir.parent
        run_dir = checkpoint_dir.parents[1]
        indexed = indexes.get(run_dir.resolve())
        index_row = indexed.get(split_dir.resolve()) if indexed is not None else None
        if indexed is not None and index_row is None:
            # Ignore stale diagnostic folders from a previous decoding configuration.
            continue
        run_config_path = run_dir / "run_config.json"
        trial = run_dir.name
        if run_config_path.is_file():
            config = json.loads(run_config_path.read_text(encoding="utf-8"))
            trial = str(config.get("arguments", {}).get("trial_name") or trial)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        symmetric_path = split_dir / "metrics_symmetric_t2s.json"
        symmetric = (
            json.loads(symmetric_path.read_text(encoding="utf-8"))
            if symmetric_path.is_file()
            else {}
        )
        loss_path = split_dir / "teacher_forced_loss.json"
        loss = (
            json.loads(loss_path.read_text(encoding="utf-8"))
            if loss_path.is_file()
            else {}
        )
        records.append(
            {
                "trial": trial,
                "checkpoint": checkpoint_dir.name,
                "split": split_dir.name,
                "num_samples": metrics.get("num_samples"),
                "sentence_accuracy_tol2": finite(
                    metrics.get("sentence_accuracy_tol2")
                ),
                "sentence_accuracy_exact": finite(
                    metrics.get("sentence_accuracy_exact")
                ),
                "cer": finite(metrics.get("cer")),
                "symmetric_t2s_sentence_accuracy_tol2": finite(
                    symmetric.get("sentence_accuracy_tol2")
                ),
                "symmetric_t2s_cer": finite(symmetric.get("cer")),
                "teacher_forced_loss": finite(loss.get("loss")),
                "target_tokens": loss.get("target_tokens"),
                "manifest_sha256": loss.get("manifest_sha256"),
                "generation_max_length": (
                    index_row.get("generation_max_length") if index_row else None
                ),
                "metrics_path": str(metrics_path),
            }
        )
    return records


def write_diagnostic_scoreboard(
    report_dir: Path, records: list[dict[str, Any]]
) -> None:
    (report_dir / "diagnostic_scoreboard.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fieldnames = list(records[0]) if records else [
        "trial",
        "checkpoint",
        "split",
        "sentence_accuracy_tol2",
        "cer",
        "teacher_forced_loss",
    ]
    with (report_dir / "diagnostic_scoreboard.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def ood_comparison(report_dir: Path, records: list[dict[str, Any]]) -> None:
    rows = [row for row in records if row["split"] == "ood_panel"]
    if not rows:
        return
    labels = [f"{row['trial']}\n{row['checkpoint']}" for row in rows]
    positions = list(range(len(rows)))
    figure, axes = plt.subplots(
        2, 2, figsize=(max(14, len(rows) * 1.8), 10), constrained_layout=True
    )
    series = (
        ("sentence_accuracy_tol2", "OOD official-asymmetric accuracy", "#64748b"),
        (
            "symmetric_t2s_sentence_accuracy_tol2",
            "OOD symmetric-t2s accuracy",
            "#2563eb",
        ),
        ("symmetric_t2s_cer", "OOD symmetric-t2s CER", "#d97706"),
        ("teacher_forced_loss", "OOD teacher-forced loss", "#0f766e"),
    )
    for axis, (key, title, color) in zip(axes.flat, series):
        values_by_row = [row[key] if row[key] is not None else float("nan") for row in rows]
        axis.bar(positions, values_by_row, color=color)
        axis.set_xticks(positions, labels, rotation=35, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(report_dir / "ood_comparison.png", dpi=170)
    plt.close(figure)


def ood_data_distribution(report_dir: Path, report_path: Path | None) -> None:
    if not report_path or not report_path.is_file():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    groups = report.get("accepted", {}).get("groups", {})
    if not groups:
        return
    labels = list(groups)
    rows = [groups[label].get("rows", 0) for label in labels]
    hours = [groups[label].get("hours", 0) for label in labels]
    positions = list(range(len(labels)))
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].bar(positions, rows, color="#2563eb")
    axes[0].set_title("Clean full OOD rows by publisher split")
    axes[1].bar(positions, hours, color="#0f766e")
    axes[1].set_title("Clean full OOD hours by publisher split")
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(report_dir / "ood_data_distribution.png", dpi=170)
    plt.close(figure)


def training_curves(report_dir: Path, records: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    plot_by_trial(axes[0, 0], values(records, "loss"), "Training loss", "Loss")
    validation_loss = values(records, "eval_validation_loss") or values(records, "eval_loss")
    plot_by_trial(axes[0, 1], validation_loss, "Validation loss", "Loss")
    validation_accuracy = values(records, "eval_validation_sentence_accuracy_tol2")
    if not validation_accuracy:
        validation_accuracy = values(records, "eval_sentence_accuracy_tol2")
    plot_by_trial(
        axes[1, 0], validation_accuracy, "Validation sentence accuracy (tol=2)", "Accuracy"
    )
    validation_cer = values(records, "eval_validation_cer") or values(records, "eval_cer")
    plot_by_trial(axes[1, 1], validation_cer, "Validation CER", "CER")
    figure.savefig(report_dir / "training_curves.png", dpi=170)
    plt.close(figure)


def diagnostic_curves(report_dir: Path, records: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    plot_by_trial(
        axes[0, 0],
        values(records, "eval_train_probe_sentence_accuracy_tol2"),
        "Train-probe sentence accuracy",
        "Accuracy",
    )
    gap_points = []
    for row in merge_epoch_records(records):
        validation = finite(row.get("eval_validation_sentence_accuracy_tol2"))
        train_probe = finite(row.get("eval_train_probe_sentence_accuracy_tol2"))
        epoch = finite(row.get("epoch"))
        if validation is not None and train_probe is not None and epoch is not None:
            gap_points.append((epoch, train_probe - validation, str(row.get("trial"))))
    plot_by_trial(axes[0, 1], gap_points, "Generalization gap", "Train probe - validation")
    plot_by_trial(axes[1, 0], values(records, "learning_rate"), "Learning rate", "LR")
    memory_points = values(records, "gpu_peak_allocated_gb")
    plot_by_trial(axes[1, 1], memory_points, "Peak allocated GPU memory", "GiB")
    gradient_points = values(records, "grad_norm")
    plot_by_trial(axes[0, 2], gradient_points, "Gradient norm", "L2 norm")
    throughput_points = values(records, "eval_validation_samples_per_second")
    if not throughput_points:
        throughput_points = values(records, "eval_samples_per_second")
    plot_by_trial(
        axes[1, 2],
        throughput_points,
        "Validation inference throughput",
        "Samples / second",
    )
    figure.savefig(report_dir / "diagnostic_curves.png", dpi=170)
    plt.close(figure)


def trial_comparison(report_dir: Path, scoreboard: list[dict[str, Any]]) -> None:
    scoreboard = [row for row in scoreboard if row.get("stage") != "diagnostic"]
    if not scoreboard:
        return
    names = [row["trial"] for row in scoreboard]
    accuracy = [row["sentence_accuracy_tol2"] or 0 for row in scoreboard]
    cer = [row["cer"] or 0 for row in scoreboard]
    figure, axes = plt.subplots(1, 2, figsize=(max(10, len(names) * 1.8), 5), constrained_layout=True)
    axes[0].bar(names, accuracy)
    axes[0].set_title("Best validation sentence accuracy")
    axes[0].set_ylim(0, max(1.0, max(accuracy) * 1.1))
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(names, cer, color="#d97706")
    axes[1].set_title("CER at best checkpoint")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(report_dir / "trial_comparison.png", dpi=170)
    plt.close(figure)


def scene_label(scene: str) -> str:
    match = re.match(r"\s*(\d+)", scene)
    return match.group(1) if match else scene[:8]


def data_distribution(report_dir: Path, report_path: Path | None) -> None:
    if not report_path or not report_path.is_file():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    scenes = report.get("scene_distribution", {})
    train = scenes.get("train", {})
    validation = scenes.get("validation", {})
    labels = sorted(set(train) | set(validation))
    duration_histogram = report.get("duration_histogram", {})
    bins = list(duration_histogram)
    counts = [duration_histogram[key] for key in bins]
    figure, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    positions = list(range(len(labels)))
    axes[0].bar(
        [position - 0.2 for position in positions],
        [train.get(label, 0) for label in labels],
        width=0.4,
        label="train",
    )
    axes[0].bar(
        [position + 0.2 for position in positions],
        [validation.get(label, 0) for label in labels],
        width=0.4,
        label="validation",
    )
    axes[0].set_xticks(positions, [scene_label(label) for label in labels], rotation=45)
    axes[0].set_title("Scene distribution (numeric scene id)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(bins, counts, color="#0f766e")
    axes[1].set_title("Audio duration distribution")
    axes[1].set_xlabel("Duration bucket (seconds)")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(report_dir / "data_distribution.png", dpi=170)
    plt.close(figure)

    with (report_dir / "scene_distribution.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["scene", "train", "validation", "train_probe"])
        probe = scenes.get("train_probe", {})
        for label in labels:
            writer.writerow(
                [label, train.get(label, 0), validation.get(label, 0), probe.get(label, 0)]
            )


def scene_comparison(report_dir: Path, outputs_root: Path) -> None:
    reports = sorted(
        outputs_root.glob("**/diagnostics/best_model/validation/scene_metrics.csv")
    )
    if not reports:
        return
    per_trial: dict[str, dict[str, float]] = {}
    scene_counts: dict[str, int] = {}
    for path in reports:
        try:
            trial = path.parents[3].name
        except IndexError:
            trial = path.as_posix()
        if trial == "smoke-overfit" or trial.startswith("mem-"):
            continue
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        per_trial[trial] = {}
        for row in rows:
            scene = str(row.get("scene", "")).strip()
            accuracy = finite(row.get("sentence_accuracy_tol2"))
            if not scene or accuracy is None:
                continue
            per_trial[trial][scene] = accuracy
            scene_counts[scene] = max(
                scene_counts.get(scene, 0), int(float(row.get("num_samples", 0) or 0))
            )
    all_scenes = {scene for values in per_trial.values() for scene in values}
    scenes = sorted(all_scenes, key=lambda scene: (-scene_counts.get(scene, 0), scene))[:30]
    trials = sorted(per_trial)
    if not scenes or not trials:
        return
    labels = [
        f"{scene[:28]}{'…' if len(scene) > 28 else ''} (n={scene_counts.get(scene, 0)})"
        for scene in scenes
    ]
    if len(trials) == 1:
        trial = trials[0]
        values_by_scene = [per_trial[trial].get(scene, float("nan")) for scene in scenes]
        figure, axis = plt.subplots(
            figsize=(12, max(7, len(scenes) * 0.32)), constrained_layout=True
        )
        positions = list(range(len(scenes)))
        axis.barh(positions, values_by_scene, color="#0f766e")
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_xlim(0, 1)
        axis.set_xlabel("Sentence accuracy (tol=2)")
        axis.set_title(f"{trial}: top validation scenes by sample count")
        axis.grid(axis="x", alpha=0.25)
        figure.savefig(report_dir / "scene_comparison.png", dpi=170)
        plt.close(figure)
        return
    matrix = [
        [per_trial[trial].get(scene, float("nan")) for scene in scenes]
        for trial in trials
    ]
    figure, axis = plt.subplots(
        figsize=(max(14, len(scenes) * 0.55), max(4.5, len(trials) * 0.65)),
        constrained_layout=True,
    )
    image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(len(scenes)), labels, rotation=60, ha="right")
    axis.set_yticks(range(len(trials)), trials)
    axis.set_xlabel("Top validation scenes by sample count")
    axis.set_title("Best-model validation accuracy by scene (top 30)")
    figure.colorbar(image, ax=axis, label="Sentence accuracy (tol=2)")
    figure.savefig(report_dir / "scene_comparison.png", dpi=170)
    plt.close(figure)


def html_table(rows_data: list[dict[str, Any]]) -> str:
    headers = list(rows_data[0]) if rows_data else []
    rows = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(row.get(key, '')))}</td>" for key in headers)
        + "</tr>"
        for row in rows_data
    )
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(key)}</th>" for key in headers)
        + "</tr></thead><tbody>"
        + rows
        + "</tbody></table>"
        if headers
        else "<p>No completed records yet.</p>"
    )


def write_html(
    report_dir: Path,
    scoreboard: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> None:
    headers = list(scoreboard[0]) if scoreboard else []
    rows = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(row.get(key, '')))}</td>" for key in headers)
        + "</tr>"
        for row in scoreboard
    )
    table = html_table(scoreboard)
    diagnostic_table = html_table(diagnostics)
    image_names = [
        name
        for name in (
            "training_curves.png",
            "diagnostic_curves.png",
            "trial_comparison.png",
            "data_distribution.png",
            "scene_comparison.png",
            "ood_data_distribution.png",
            "ood_comparison.png",
        )
        if (report_dir / name).is_file()
    ]
    images = "".join(
        f'<section><h2>{html.escape(name)}</h2><img src="{html.escape(name)}" alt="{html.escape(name)}"></section>'
        for name in image_names
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Cantonese ASR experiment report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;width:100%;font-size:.85rem}}th,td{{border:1px solid #ddd;padding:.4rem;text-align:left}}img{{max-width:100%;height:auto}}section{{margin-top:2rem}}</style>
</head><body><h1>Cantonese ASR experiment report</h1>{table}
<h2>Independent checkpoint diagnostics</h2>{diagnostic_table}{images}</body></html>
"""
    (report_dir / "report.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    records = collect_records(args.outputs_root)
    scoreboard = build_scoreboard(records)
    diagnostics = collect_diagnostic_scoreboard(args.outputs_root)
    write_scoreboard(args.report_dir, scoreboard)
    write_diagnostic_scoreboard(args.report_dir, diagnostics)
    if records:
        training_curves(args.report_dir, records)
        diagnostic_curves(args.report_dir, records)
        trial_comparison(args.report_dir, scoreboard)
    data_distribution(args.report_dir, args.data_report)
    ood_data_distribution(args.report_dir, args.ood_data_report)
    scene_comparison(args.report_dir, args.outputs_root)
    ood_comparison(args.report_dir, diagnostics)
    write_html(args.report_dir, scoreboard, diagnostics)
    print(
        json.dumps(
            {
                "metric_records": len(records),
                "trials": len(scoreboard),
                "checkpoint_diagnostics": len(diagnostics),
                "report": str((args.report_dir / "report.html").resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
