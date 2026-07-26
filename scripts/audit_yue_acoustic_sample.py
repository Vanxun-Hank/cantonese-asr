#!/usr/bin/env python3
"""Run diagnostic-only SenseVoice review and finalize the yue admission report."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cantonese_asr.common_voice_audit import decide_admission  # noqa: E402
from cantonese_asr.data_quality import (  # noqa: E402
    FunASRSenseVoiceRunner,
    WhisperTokenCounter,
    audit_rows,
)
from cantonese_asr.io import write_jsonl  # noqa: E402
from scripts.audit_common_voice_yue import write_html, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--sensevoice-model", type=Path, required=True)
    parser.add_argument("--whisper-model", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def read_review_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: list[dict[str, object]] = []
    for row in rows:
        result.append(
            {
                "id": row["id"],
                "source": "common_voice_26_yue",
                "publisher_split": row["publisher_split"],
                "scene": row["review_bucket"],
                "review_bucket": row["review_bucket"],
                "client_id": row["client_id"],
                "duration_s": float(row["duration_s"]),
                "text": row["text"],
                "audio_path": row["audio_path"],
                "audio_sha256": row["audio_sha256"],
                "pcm_sha256": row["pcm_sha256"],
            }
        )
    return result


def main() -> None:
    args = parse_args()
    sample_path = args.audit_dir / "style_review_sample.csv"
    report_path = args.audit_dir / "data_report.json"
    if not sample_path.is_file() or not report_path.is_file():
        raise SystemExit("CPU audit outputs are incomplete")
    rows = read_review_rows(sample_path)
    if not rows:
        raise SystemExit("Acoustic review sample is empty")
    if len(rows) > 230:
        raise SystemExit(f"Acoustic review exceeds fixed cap: {len(rows)}")

    runner = FunASRSenseVoiceRunner(
        args.sensevoice_model, args.device, args.batch_size
    )
    token_counter = WhisperTokenCounter(args.whisper_model)
    records = audit_rows(rows, args.project_root, runner, token_counter)
    decoded = [
        record
        for record in records
        if isinstance(record.get("sensevoice"), dict)
    ]
    languages = Counter(
        str(record["sensevoice"]["language"]) for record in decoded
    )
    yue_rate = languages["yue"] / len(decoded) if decoded else 0.0
    cer_values = sorted(
        float(record["sensevoice"]["normalized_cer"]) for record in decoded
    )
    summary = {
        "sample_rows": len(rows),
        "decoded_rows": len(decoded),
        "languages": dict(sorted(languages.items())),
        "sample_yue_rate": yue_rate,
        "normalized_cer_mean": (
            sum(cer_values) / len(cer_values) if cer_values else None
        ),
        "normalized_cer_p50": (
            cer_values[round((len(cer_values) - 1) * 0.5)] if cer_values else None
        ),
        "runner": runner.provenance(),
        "training_runs_started": 0,
        "platform_submissions": 0,
    }
    write_jsonl(args.audit_dir / "acoustic_review.jsonl", records)
    write_json(args.audit_dir / "acoustic_summary.json", summary)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["acoustic_review"] = summary
    report["incremental"]["sample_yue_rate"] = yue_rate
    report["admission"] = decide_admission(report["incremental"])
    write_json(report_path, report)
    write_html(args.audit_dir / "report.html", report)
    print(json.dumps(
        {
            "sample_rows": len(rows),
            "languages": dict(sorted(languages.items())),
            "sample_yue_rate": yue_rate,
            "decision": report["admission"]["decision"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
