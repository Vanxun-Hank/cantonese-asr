#!/usr/bin/env python3
"""Read-only verification for the official speed-perturbation artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cantonese_asr.io import read_jsonl, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--sample-parents", type=int, default=32)
    parser.add_argument("--expected-parents", type=int, default=6292)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def resolve_audio_path(value: Any, project_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else project_root / path


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.manifest)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    quarantine = read_jsonl(args.quarantine)
    if quarantine:
        raise SystemExit(f"quarantine is not empty: {len(quarantine)} rows")

    counts = Counter(f"{float(row['speed_factor']):.1f}" for row in rows)
    expected = {"0.9": args.expected_parents, "1.0": args.expected_parents, "1.1": args.expected_parents}
    if dict(counts) != expected:
        raise SystemExit(f"unexpected factor counts: {dict(counts)}")
    expected_rows = args.expected_parents * 3
    if len(rows) != expected_rows:
        raise SystemExit(f"unexpected manifest rows: {len(rows)}")
    if report.get("counts", {}).get("rows") != expected_rows:
        raise SystemExit(f"data report does not record {expected_rows} rows")

    by_parent: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_parent.setdefault(str(row["parent_id"]), []).append(row)
    if len(by_parent) != args.expected_parents:
        raise SystemExit(f"unexpected parent count: {len(by_parent)}")

    checked: list[dict[str, Any]] = []
    for parent_id in sorted(by_parent, key=int)[: args.sample_parents]:
        views = by_parent[parent_id]
        if {float(row["speed_factor"]) for row in views} != {0.9, 1.0, 1.1}:
            raise SystemExit(f"missing speed view for parent {parent_id}")
        if len({str(row["text"]) for row in views}) != 1:
            raise SystemExit(f"label changed for parent {parent_id}")
        for row in views:
            if float(row["speed_factor"]) == 1.0:
                continue
            path = resolve_audio_path(row["audio_path"], args.project_root)
            info = sf.info(str(path))
            if info.samplerate != 16_000 or info.channels != 1:
                raise SystemExit(f"bad derived format: {path}")
            checked.append(
                {
                    "parent_id": parent_id,
                    "speed_factor": row["speed_factor"],
                    "audio_path": str(row["audio_path"]),
                    "duration_s": float(info.duration),
                }
            )

    result = {
        "passed": True,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "rows": len(rows),
        "factor_counts": dict(sorted(counts.items())),
        "sampled_derived_views": len(checked),
        "samples": checked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
