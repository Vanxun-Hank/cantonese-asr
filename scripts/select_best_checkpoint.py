#!/usr/bin/env python3
"""Select a checkpoint from independent raw-manifest validation diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = json.loads(args.baseline_metrics.read_text(encoding="utf-8"))
    baseline_cer = float(baseline["cer"])
    candidates = []
    for path in sorted(args.run_dir.glob("diagnostics/*/validation/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        name = path.parents[1].name
        model_dir = args.run_dir / name
        if not model_dir.is_dir():
            continue
        candidates.append(
            {
                "checkpoint": name,
                "model_dir": str(model_dir),
                "sentence_accuracy_tol2": float(metrics["sentence_accuracy_tol2"]),
                "cer": float(metrics["cer"]),
                "cer_guardrail_passed": float(metrics["cer"]) <= baseline_cer,
            }
        )
    eligible = [row for row in candidates if row["cer_guardrail_passed"]]
    if not eligible:
        raise SystemExit("No independently evaluated checkpoint passes the CER guardrail")
    selected = max(
        eligible,
        key=lambda row: (
            row["sentence_accuracy_tol2"],
            -row["cer"],
            row["checkpoint"] == "best_model",
        ),
    )
    report = {
        "baseline_cer": baseline_cer,
        "selected": selected,
        "candidates": sorted(
            candidates,
            key=lambda row: (-row["sentence_accuracy_tol2"], row["cer"]),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
