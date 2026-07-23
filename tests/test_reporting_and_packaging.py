from __future__ import annotations

import json
import csv
import subprocess
import sys
import zipfile
from pathlib import Path

from cantonese_asr.io import write_jsonl


def test_plot_script_builds_reusable_report(tmp_path: Path) -> None:
    project_root = Path(__file__).parents[1]
    outputs = tmp_path / "outputs" / "trial-a"
    outputs.mkdir(parents=True)
    write_jsonl(
        outputs / "metrics.jsonl",
        [
            {"trial": "trial-a", "event": "log", "epoch": 1, "global_step": 10, "loss": 2.0, "learning_rate": 1e-5, "gpu_peak_allocated_gb": 8.0},
            {"trial": "trial-a", "event": "evaluation", "epoch": 1, "global_step": 10, "eval_validation_loss": 1.5, "eval_validation_sentence_accuracy_tol2": 0.4, "eval_validation_cer": 0.3},
            {"trial": "trial-a", "event": "evaluation", "epoch": 1, "global_step": 10, "eval_train_probe_sentence_accuracy_tol2": 0.6, "eval_train_probe_cer": 0.2},
            {"trial": "trial-a", "event": "evaluation", "epoch": 2, "global_step": 20, "eval_validation_loss": 1.0, "eval_validation_sentence_accuracy_tol2": 0.7, "eval_validation_cer": 0.15},
            {"trial": "trial-a", "event": "evaluation", "epoch": 2, "global_step": 20, "eval_train_probe_sentence_accuracy_tol2": 0.85, "eval_train_probe_cer": 0.08},
        ],
    )
    data_report = tmp_path / "data_report.json"
    data_report.write_text(
        json.dumps(
            {
                "scene_distribution": {
                    "train": {"1问候": 8},
                    "validation": {"1问候": 2},
                    "train_probe": {"1问候": 2},
                },
                "duration_histogram": {"0-2": 5, "2-4": 5},
            }
        ),
        encoding="utf-8",
    )
    scene_dir = outputs / "diagnostics" / "best_model" / "validation"
    scene_dir.mkdir(parents=True)
    with (scene_dir / "scene_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scene", "num_samples", "sentence_accuracy_tol2"],
        )
        writer.writeheader()
        writer.writerow(
            {"scene": "1问候", "num_samples": 2, "sentence_accuracy_tol2": 0.5}
        )
    (scene_dir / "metrics.json").write_text(
        json.dumps(
            {
                "num_samples": 2,
                "sentence_accuracy_tol2": 0.5,
                "sentence_accuracy_exact": 0.4,
                "cer": 0.2,
            }
        ),
        encoding="utf-8",
    )
    ood_dir = outputs / "diagnostics" / "best_model" / "ood_panel"
    ood_dir.mkdir(parents=True)
    (ood_dir / "metrics.json").write_text(
        json.dumps(
            {
                "num_samples": 4,
                "sentence_accuracy_tol2": 0.25,
                "sentence_accuracy_exact": 0.1,
                "cer": 0.4,
            }
        ),
        encoding="utf-8",
    )
    (ood_dir / "teacher_forced_loss.json").write_text(
        json.dumps(
            {"loss": 1.25, "target_tokens": 40, "manifest_sha256": "a" * 64}
        ),
        encoding="utf-8",
    )
    (ood_dir / "metrics_symmetric_t2s.json").write_text(
        json.dumps({"sentence_accuracy_tol2": 0.75, "cer": 0.12}),
        encoding="utf-8",
    )
    report_dir = tmp_path / "report"
    subprocess.run(
        [
            sys.executable,
            "scripts/plot_experiments.py",
            "--outputs-root",
            str(tmp_path / "outputs"),
            "--data-report",
            str(data_report),
            "--report-dir",
            str(report_dir),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (report_dir / "report.html").is_file()
    assert (report_dir / "training_curves.png").stat().st_size > 0
    assert (report_dir / "scene_comparison.png").stat().st_size > 0
    assert (report_dir / "ood_comparison.png").stat().st_size > 0
    assert (report_dir / "diagnostic_scoreboard.csv").stat().st_size > 0
    diagnostic_scoreboard = json.loads(
        (report_dir / "diagnostic_scoreboard.json").read_text()
    )
    assert any(
        row["split"] == "ood_panel"
        and row["teacher_forced_loss"] == 1.25
        and row["symmetric_t2s_sentence_accuracy_tol2"] == 0.75
        for row in diagnostic_scoreboard
    )
    scoreboard = json.loads((report_dir / "scoreboard.json").read_text())
    assert scoreboard[0]["sentence_accuracy_tol2"] == 0.7
    assert scoreboard[0]["train_probe_accuracy"] == 0.85


def test_package_script_enforces_flat_single_weight_zip(tmp_path: Path) -> None:
    project_root = Path(__file__).parents[1]
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in (
        "model.safetensors",
        "config.json",
        "preprocessor_config.json",
        "generation_config.json",
        "tokenizer.json",
    ):
        (model_dir / name).write_bytes(b"{}" if name.endswith(".json") else b"weight")
    (model_dir / "optimizer.pt").write_bytes(b"training-only")
    output_zip = tmp_path / "submission.zip"
    subprocess.run(
        [
            sys.executable,
            "scripts/package_submission.py",
            "--model-dir",
            str(model_dir),
            "--predict-py",
            str(project_root / "predict.py"),
            "--requirements",
            str(project_root / "requirements-submission.txt"),
            "--output-zip",
            str(output_zip),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    with zipfile.ZipFile(output_zip) as archive:
        names = archive.namelist()
    assert "model.safetensors" in names
    assert "predict.py" in names
    assert "optimizer.pt" not in names
    assert all("/" not in name for name in names)
