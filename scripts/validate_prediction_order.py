#!/usr/bin/env python3
"""Verify prediction rows exactly preserve the official test-list order."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cantonese_asr.io import read_jsonl
from predict import read_test_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-list", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    expected = [str(row["audio_path"]) for row in read_test_rows(args.test_list)]
    rows = read_jsonl(args.predictions)
    actual = [str(row.get("audio_path")) for row in rows]
    report = {
        "passed": actual == expected,
        "expected_rows": len(expected),
        "actual_rows": len(actual),
        "order_exact": actual == expected,
        "all_predictions_are_strings": all(
            isinstance(row.get("pred_text"), str) for row in rows
        ),
    }
    report["passed"] = report["passed"] and report["all_predictions_are_strings"]
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
