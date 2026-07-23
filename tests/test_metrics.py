from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cantonese_asr.metrics import (
    build_error_analysis,
    compute_official_metrics,
    normalize_prediction,
    normalize_reference,
)


def test_official_normalization_converts_prediction_only() -> None:
    assert normalize_reference("[NORM] 發展， 好！") == "發展好"
    assert normalize_prediction("[RAW] 發展， 好！") == "发展好"


def test_tolerance_and_cer() -> None:
    metrics = compute_official_metrics(
        ["我今日去饮茶", "天气很好"],
        ["我今日饮茶", "天气糟糕啊"],
    )
    assert metrics["num_samples"] == 2
    assert metrics["sentence_accuracy_tol2"] == pytest.approx(0.5)
    assert metrics["sentence_accuracy_exact"] == 0
    assert metrics["cer"] > 0


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        compute_official_metrics(["一"], [])


def test_error_analysis_has_scenes_and_confusions() -> None:
    report = build_error_analysis(
        [
            {
                "audio_path": "a.wav",
                "scene": "1问候",
                "reference": "你好",
                "prediction": "你号",
            }
        ]
    )
    assert report["scene_metrics"]["1问候"]["num_samples"] == 1
    assert report["substitutions"][0]["reference"] == "好"


def test_ood_symmetric_t2s_is_separate_from_official_metric(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    references = tmp_path / "references.jsonl"
    report_dir = tmp_path / "report"
    predictions.write_text(
        json.dumps({"audio_path": "a.wav", "pred_text": "发展"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    references.write_text(
        json.dumps({"audio_path": "a.wav", "text": "發展"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_predictions.py",
            "--pred-jsonl",
            str(predictions),
            "--reference",
            str(references),
            "--report-dir",
            str(report_dir),
            "--write-symmetric-t2s",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    official = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    symmetric = json.loads(
        (report_dir / "metrics_symmetric_t2s.json").read_text(encoding="utf-8")
    )
    assert official["sentence_accuracy_exact"] == 0.0
    assert official["cer"] == 0.5
    assert symmetric["sentence_accuracy_exact"] == 1.0
    assert symmetric["cer"] == 0.0
