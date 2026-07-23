#!/usr/bin/env python3
"""Prove local competition metrics match the downloaded official evaluator.py."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cantonese_asr.io import write_jsonl
from cantonese_asr.metrics import compute_official_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-evaluator", type=Path, required=True)
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("artifacts/reports/deployment/metric_parity.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.official_evaluator.is_file():
        raise SystemExit(f"Official evaluator not found: {args.official_evaluator}")
    references = ["发展很好。", "我今日去饮茶", "天气很好"]
    predictions = ["發展很好!", "我今日饮茶", "天气糟糕啊"]
    audio_paths = [f"test_audio/{index}.wav" for index in range(len(references))]
    local = compute_official_metrics(references, predictions)

    with tempfile.TemporaryDirectory(prefix="canto-metric-parity-") as temp_name:
        temp = Path(temp_name)
        verify = temp / "verify.csv"
        with verify.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["audio_path", "ref_text"])
            writer.writeheader()
            writer.writerows(
                {"audio_path": audio, "ref_text": reference}
                for audio, reference in zip(audio_paths, references)
            )
        predictions_path = temp / "predictions.jsonl"
        write_jsonl(
            predictions_path,
            [
                {"audio_path": audio, "pred_text": prediction}
                for audio, prediction in zip(audio_paths, predictions)
            ],
        )
        official_dir = temp / "official"
        subprocess.run(
            [
                sys.executable,
                str(args.official_evaluator),
                str(predictions_path),
                str(verify),
                str(official_dir),
            ],
            check=True,
        )
        official = json.loads((official_dir / "metrics.json").read_text(encoding="utf-8"))

    checks = {
        "cer": abs(float(local["cer"]) - float(official["cer"])) < 1e-12,
        "sentence_accuracy": abs(
            float(local["sentence_accuracy_tol2"])
            - float(official["sentence_accuracy"])
        )
        < 1e-12,
        "num_samples": int(local["num_samples"]) == int(official["num_samples"]),
    }
    report = {"passed": all(checks.values()), "checks": checks, "local": local, "official": official}
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

