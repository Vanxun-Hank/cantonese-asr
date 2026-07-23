#!/usr/bin/env python3
"""Score predictions with the official metric and write diagnostic reports."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cantonese_asr.io import read_jsonl
from cantonese_asr.metrics import (
    build_error_analysis,
    compute_official_metrics,
    to_simplified,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-jsonl", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--reference-field", default="text")
    parser.add_argument("--metrics-jsonl", type=Path)
    parser.add_argument("--trial")
    parser.add_argument("--split", choices=["validation", "train_probe", "template_pre"])
    parser.add_argument(
        "--write-symmetric-t2s",
        action="store_true",
        help="Also normalize references to simplified Chinese for cross-corpus OOD diagnosis",
    )
    return parser.parse_args()


def read_references(path: Path, preferred_field: str) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
        for row in rows:
            if preferred_field not in row:
                for fallback in ("ref_text", "text:label", "reference", "text"):
                    if fallback in row:
                        row[preferred_field] = row[fallback]
                        break
        return rows
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Empty reference CSV: {path}")
        audio_key = next(
            (
                key
                for key in reader.fieldnames
                if key.strip().lower() in {"audio_path", "audio:file"}
            ),
            None,
        )
        text_key = next(
            (
                key
                for key in reader.fieldnames
                if key.strip().lower()
                in {preferred_field.lower(), "ref_text", "text:label", "text"}
            ),
            None,
        )
        if not audio_key or not text_key:
            raise ValueError(
                f"Reference CSV needs audio_path and text columns; got {reader.fieldnames}"
            )
        return [
            {
                "audio_path": str(row.get(audio_key, "")).strip(),
                preferred_field: row.get(text_key, ""),
                "scene": row.get("scene") or row.get("场景") or "unknown",
            }
            for row in reader
        ]


def main() -> None:
    args = parse_args()
    predictions = read_jsonl(args.pred_jsonl)
    references = read_references(args.reference, args.reference_field)
    if len(predictions) != len(references):
        raise ValueError(
            f"Row count mismatch: predictions={len(predictions)}, references={len(references)}"
        )

    analysis_rows = []
    seen: set[str] = set()
    for line_no, (prediction, reference) in enumerate(
        zip(predictions, references), start=1
    ):
        pred_audio = str(prediction.get("audio_path", "")).strip()
        ref_audio = str(reference.get("audio_path", "")).strip()
        if not pred_audio or not ref_audio:
            raise ValueError(f"Missing audio_path at row {line_no}")
        if pred_audio != ref_audio:
            raise ValueError(
                f"audio_path order mismatch at row {line_no}: {pred_audio!r} != {ref_audio!r}"
            )
        if pred_audio in seen:
            raise ValueError(f"Duplicate audio_path: {pred_audio}")
        seen.add(pred_audio)
        pred_text = prediction.get("pred_text")
        if pred_text is None:
            raise ValueError(f"Missing pred_text at row {line_no}: {pred_audio}")
        ref_text = reference.get(args.reference_field)
        if ref_text is None:
            raise ValueError(
                f"Missing {args.reference_field!r} at row {line_no}: {ref_audio}"
            )
        analysis_rows.append(
            {
                "audio_path": ref_audio,
                "scene": reference.get("scene") or "unknown",
                "reference": str(ref_text),
                "prediction": str(pred_text),
            }
        )

    metrics = compute_official_metrics(
        [row["reference"] for row in analysis_rows],
        [row["prediction"] for row in analysis_rows],
    )
    diagnostics = build_error_analysis(analysis_rows)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    symmetric_metrics = None
    if args.write_symmetric_t2s:
        symmetric_metrics = compute_official_metrics(
            [to_simplified(row["reference"]) for row in analysis_rows],
            [row["prediction"] for row in analysis_rows],
        )
        (args.report_dir / "metrics_symmetric_t2s.json").write_text(
            json.dumps(symmetric_metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    (args.report_dir / "error_examples.json").write_text(
        json.dumps(diagnostics["error_examples"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.report_dir / "top_confusions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["type", "reference", "prediction", "count"])
        for row in diagnostics["substitutions"]:
            writer.writerow(
                ["substitution", row["reference"], row["prediction"], row["count"]]
            )
        for row in diagnostics["deletions"]:
            writer.writerow(["deletion", row["reference"], "<del>", row["count"]])
        for row in diagnostics["insertions"]:
            writer.writerow(["insertion", "<ins>", row["prediction"], row["count"]])
    with (args.report_dir / "scene_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["scene", "num_samples", "sentence_accuracy_tol2"])
        for scene, values in diagnostics["scene_metrics"].items():
            writer.writerow(
                [scene, values["num_samples"], values["sentence_accuracy_tol2"]]
            )
    if args.metrics_jsonl:
        if not args.trial or not args.split:
            raise ValueError("--metrics-jsonl requires --trial and --split")
        args.metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)
        prefix = f"eval_{args.split}_"
        event = {
            "event": "evaluation",
            "trial": args.trial,
            "epoch": 0.0,
            "global_step": 0,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            **{prefix + key: value for key, value in metrics.items()},
        }
        with args.metrics_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    rendered: dict[str, Any] = metrics
    if symmetric_metrics is not None:
        rendered = {
            "official_asymmetric": metrics,
            "symmetric_t2s": symmetric_metrics,
        }
    print(json.dumps(rendered, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
