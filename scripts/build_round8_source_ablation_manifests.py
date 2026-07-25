#!/usr/bin/env python3
"""Build the two matched-size source ablations for Round 8."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cantonese_asr.io import read_jsonl, sha256_file, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-mdcc73", type=Path, required=True)
    parser.add_argument("--round2-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--matched-external-count", type=int, default=8_451)
    parser.add_argument("--expected-official-mdcc73-sha256")
    parser.add_argument("--expected-round2-sha256")
    return parser.parse_args()


def require_hash(path: Path, expected: str | None, label: str) -> str:
    actual = sha256_file(path)
    if expected and actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")
    return actual


def require_unique(rows: list[dict[str, Any]], label: str) -> None:
    ids = [str(row.get("id", "")).strip() for row in rows]
    if not all(ids):
        raise ValueError(f"{label} contains an empty id")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} contains duplicate ids")


def sources(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("source")) for row in rows).items()))


def write_mix(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    result = [dict(row) for row in rows]
    require_unique(result, path.stem)
    random.Random(seed).shuffle(result)
    write_jsonl(path, result)
    return {
        "path": str(path),
        "rows": len(result),
        "unique_ids": len({str(row["id"]) for row in result}),
        "sources": sources(result),
        "hours": round(
            sum(float(row.get("duration_s") or 0.0) for row in result) / 3600,
            6,
        ),
        "sha256": sha256_file(path),
    }


def main() -> None:
    args = parse_args()
    mdcc73_sha = require_hash(
        args.official_mdcc73,
        args.expected_official_mdcc73_sha256,
        "Official+MDCC73 manifest",
    )
    round2_sha = require_hash(
        args.round2_manifest,
        args.expected_round2_sha256,
        "Round 2 manifest",
    )
    mdcc73 = read_jsonl(args.official_mdcc73)
    round2 = read_jsonl(args.round2_manifest)
    require_unique(mdcc73, "Official+MDCC73 manifest")
    require_unique(round2, "Round 2 manifest")
    if len(mdcc73) != 23_304 or sources(mdcc73) != {
        "mdcc": 17_012,
        "official": 6_292,
    }:
        raise ValueError(f"Unexpected Official+MDCC73 composition: {sources(mdcc73)}")
    if len(round2) != 23_304 or sources(round2) != {
        "common_voice_26_zh_HK": 8_451,
        "mdcc": 8_561,
        "official": 6_292,
    }:
        raise ValueError(f"Unexpected Round 2 composition: {sources(round2)}")

    mdcc_official = [row for row in mdcc73 if row["source"] == "official"]
    round2_official = [row for row in round2 if row["source"] == "official"]
    if {str(row["id"]) for row in mdcc_official} != {
        str(row["id"]) for row in round2_official
    }:
        raise ValueError("Official membership differs between source manifests")

    mdcc_selected = sorted(
        (row for row in mdcc73 if row["source"] == "mdcc"),
        key=lambda row: int(row["sampling_rank"]),
    )[: args.matched_external_count]
    cv_selected = [
        row for row in round2 if row["source"] == "common_voice_26_zh_HK"
    ]
    if len(mdcc_selected) != args.matched_external_count:
        raise ValueError("Insufficient nested MDCC rows")
    if len(cv_selected) != args.matched_external_count:
        raise ValueError(
            f"Expected {args.matched_external_count} CV rows, found {len(cv_selected)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {
        "official_mdcc57": write_mix(
            args.output_dir / "official_mdcc57.jsonl",
            mdcc_official + mdcc_selected,
            seed=args.seed + 10,
        ),
        "official_cv57": write_mix(
            args.output_dir / "official_cv57.jsonl",
            round2_official + cv_selected,
            seed=args.seed + 20,
        ),
    }
    expected = {
        "official_mdcc57": {"mdcc": 8_451, "official": 6_292},
        "official_cv57": {
            "common_voice_26_zh_HK": 8_451,
            "official": 6_292,
        },
    }
    for name, source_counts in expected.items():
        summary = manifests[name]
        if summary["rows"] != 14_743 or summary["sources"] != source_counts:
            raise RuntimeError(f"Unexpected {name} output: {summary}")

    report = {
        "passed": True,
        "seed": args.seed,
        "matched_external_count": args.matched_external_count,
        "policy": {
            "same_official_membership": True,
            "mdcc_subset": "sampling_rank prefix from Official+MDCC73",
            "common_voice_subset": "all Round 2 Common Voice train rows",
            "sampling_without_replacement": True,
        },
        "inputs": {
            "official_mdcc73": {
                "path": str(args.official_mdcc73),
                "rows": len(mdcc73),
                "sha256": mdcc73_sha,
            },
            "round2": {
                "path": str(args.round2_manifest),
                "rows": len(round2),
                "sha256": round2_sha,
            },
        },
        "manifests": manifests,
    }
    (args.output_dir / "data_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
