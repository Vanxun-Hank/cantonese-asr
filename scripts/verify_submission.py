#!/usr/bin/env python3
"""Extract a submission and prove its bundled predict.py works fully offline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cantonese_asr.io import read_jsonl, sha256_file, write_jsonl
from predict import read_test_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-zip", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--test-list", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = read_test_rows(args.test_list)[: args.max_samples]
    expected_paths = [str(row["audio_path"]) for row in source_rows]
    with tempfile.TemporaryDirectory(prefix="canto-submission-verify-") as temp_name:
        temp = Path(temp_name)
        package = temp / "package"
        package.mkdir()
        with zipfile.ZipFile(args.submission_zip) as archive:
            archive.extractall(package)
            names = archive.namelist()
        subset = temp / "test_subset.jsonl"
        write_jsonl(subset, [{"audio_path": path} for path in expected_paths])
        output = temp / "predictions.jsonl"
        env = os.environ.copy()
        env.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            }
        )
        subprocess.run(
            [
                sys.executable,
                "predict.py",
                "--audio_dir",
                str(args.audio_dir.resolve()),
                "--test_list",
                str(subset),
                "--output_jsonl",
                str(output),
            ],
            cwd=package,
            env=env,
            check=True,
        )
        predictions = read_jsonl(output)
        actual_paths = [str(row.get("audio_path")) for row in predictions]
        passed = actual_paths == expected_paths and all(
            isinstance(row.get("pred_text"), str) for row in predictions
        )
    report = {
        "passed": passed,
        "submission_zip": str(args.submission_zip.resolve()),
        "submission_sha256": sha256_file(args.submission_zip),
        "flat_archive": all("/" not in name and "\\" not in name for name in names),
        "num_samples": len(expected_paths),
        "order_exact": actual_paths == expected_paths,
        "offline_environment": True,
        "files": names,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
