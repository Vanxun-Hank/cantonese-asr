#!/usr/bin/env python3
"""Materialize per-epoch validation/train-probe predictions and error reports."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--train-probe-manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    if path.name == "best_model":
        return 10**18
    return int(path.name.rsplit("-", 1)[-1])


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    checkpoints = sorted(
        [path for path in args.run_dir.glob("checkpoint-*") if path.is_dir()],
        key=checkpoint_step,
    )
    best_model = args.run_dir / "best_model"
    if best_model.is_dir():
        checkpoints.append(best_model)
    if not checkpoints:
        raise SystemExit(f"No checkpoint-* or best_model below {args.run_dir}")

    generated = []
    for checkpoint in checkpoints:
        checkpoint_name = checkpoint.name
        for split, manifest in (
            ("validation", args.validation_manifest),
            ("train_probe", args.train_probe_manifest),
        ):
            report_dir = args.run_dir / "diagnostics" / checkpoint_name / split
            prediction_path = report_dir / "predictions.jsonl"
            report_dir.mkdir(parents=True, exist_ok=True)
            run(
                [
                    str(args.python),
                    "predict.py",
                    "--model_dir",
                    str(checkpoint),
                    "--processor_dir",
                    str(args.run_dir),
                    "--audio_dir",
                    str(args.audio_dir),
                    "--test_list",
                    str(manifest),
                    "--output_jsonl",
                    str(prediction_path),
                    "--batch_size",
                    str(args.batch_size),
                ]
            )
            run(
                [
                    str(args.python),
                    "scripts/evaluate_predictions.py",
                    "--pred-jsonl",
                    str(prediction_path),
                    "--reference",
                    str(manifest),
                    "--reference-field",
                    "text",
                    "--report-dir",
                    str(report_dir),
                ]
            )
            generated.append(
                {
                    "checkpoint": checkpoint_name,
                    "split": split,
                    "report_dir": str(report_dir),
                }
            )
    (args.run_dir / "diagnostics" / "index.json").write_text(
        json.dumps(generated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

