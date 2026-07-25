#!/usr/bin/env python3
"""Build the fixed Official + MDCC source-isolation training manifest."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cantonese_asr.io import read_jsonl, sha256_file, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-train", type=Path, required=True)
    parser.add_argument("--mdcc-train", type=Path, required=True)
    parser.add_argument("--round2-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--excluded-manifest",
        type=Path,
        action="append",
        default=[],
        help="Protected validation, public-test or OOD manifest.",
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-official-rows", type=int, default=6_292)
    parser.add_argument("--expected-mdcc-pool-rows", type=int, default=64_779)
    parser.add_argument("--mdcc-sample-count", type=int, default=17_012)
    parser.add_argument("--expected-official-sha256")
    parser.add_argument("--expected-mdcc-sha256")
    parser.add_argument("--expected-round2-sha256")
    parser.add_argument("--verify-audio-exists", action="store_true")
    return parser.parse_args()


def require_sha256(path: Path, expected: str | None, label: str) -> str:
    actual = sha256_file(path)
    if expected and actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")
    return actual


def require_unique_ids(rows: list[dict[str, Any]], label: str) -> set[str]:
    ids = [str(row.get("id", "")).strip() for row in rows]
    if not all(ids):
        raise ValueError(f"{label} contains an empty id")
    duplicates = len(ids) - len(set(ids))
    if duplicates:
        raise ValueError(f"{label} contains {duplicates} duplicate ids")
    return set(ids)


def validate_source_rows(
    rows: list[dict[str, Any]],
    *,
    label: str,
    expected_source: str,
    project_root: Path,
    verify_audio_exists: bool,
) -> None:
    for index, row in enumerate(rows, start=1):
        if str(row.get("source", "")).strip() != expected_source:
            raise ValueError(
                f"{label} row {index} has unexpected source: {row.get('source')!r}"
            )
        if str(row.get("split", "train")).strip() != "train":
            raise ValueError(
                f"{label} row {index} is not publisher train: {row.get('split')!r}"
            )
        if not str(row.get("text", "")).strip():
            raise ValueError(f"{label} row {index} has empty text")
        if float(row.get("duration_s") or 0.0) <= 0.0:
            raise ValueError(f"{label} row {index} has invalid duration")
        audio_value = str(row.get("audio_path", "")).strip()
        if not audio_value:
            raise ValueError(f"{label} row {index} has empty audio_path")
        if verify_audio_exists:
            audio_path = Path(audio_value)
            if not audio_path.is_absolute():
                audio_path = project_root / audio_path
            if not audio_path.is_file():
                raise ValueError(f"{label} row {index} audio is missing: {audio_path}")


def row_fingerprints(row: dict[str, Any]) -> set[str]:
    fingerprints: set[str] = set()
    row_id = str(row.get("id", "")).strip()
    if row_id:
        fingerprints.add(f"id:{row_id}")
    audio_sha256 = str(row.get("audio_sha256", "")).strip()
    if audio_sha256:
        fingerprints.add(f"audio_sha256:{audio_sha256}")
    audio_path = str(row.get("audio_path", "")).strip()
    if audio_path:
        fingerprints.add(f"audio_path:{audio_path}")
    source = str(row.get("source", "")).strip()
    original_audio_path = str(row.get("original_audio_path", "")).strip()
    if source and original_audio_path:
        fingerprints.add(f"source_original:{source}:{original_audio_path}")
    return fingerprints


def collect_fingerprints(rows: Iterable[dict[str, Any]]) -> set[str]:
    result: set[str] = set()
    for row in rows:
        result.update(row_fingerprints(row))
    return result


def main() -> None:
    args = parse_args()
    if args.mdcc_sample_count <= 0:
        raise ValueError("--mdcc-sample-count must be positive")

    official_sha256 = require_sha256(
        args.official_train,
        args.expected_official_sha256,
        "official train manifest",
    )
    mdcc_sha256 = require_sha256(
        args.mdcc_train,
        args.expected_mdcc_sha256,
        "MDCC train manifest",
    )
    round2_sha256 = require_sha256(
        args.round2_manifest,
        args.expected_round2_sha256,
        "Round 2 manifest",
    )
    official = read_jsonl(args.official_train)
    mdcc_pool = read_jsonl(args.mdcc_train)
    round2 = read_jsonl(args.round2_manifest)

    if len(official) != args.expected_official_rows:
        raise ValueError(
            f"Expected {args.expected_official_rows} official rows, "
            f"found {len(official)}"
        )
    if len(mdcc_pool) != args.expected_mdcc_pool_rows:
        raise ValueError(
            f"Expected {args.expected_mdcc_pool_rows} MDCC pool rows, "
            f"found {len(mdcc_pool)}"
        )
    if args.mdcc_sample_count > len(mdcc_pool):
        raise ValueError(
            f"Requested {args.mdcc_sample_count} unique MDCC rows from "
            f"a pool of {len(mdcc_pool)}"
        )

    official_ids = require_unique_ids(official, "official train")
    mdcc_ids = require_unique_ids(mdcc_pool, "MDCC train")
    require_unique_ids(round2, "Round 2 manifest")
    if official_ids & mdcc_ids:
        raise ValueError("official and MDCC source ids overlap")
    validate_source_rows(
        official,
        label="official train",
        expected_source="official",
        project_root=args.project_root,
        verify_audio_exists=args.verify_audio_exists,
    )
    validate_source_rows(
        mdcc_pool,
        label="MDCC train",
        expected_source="mdcc",
        project_root=args.project_root,
        verify_audio_exists=args.verify_audio_exists,
    )

    ordered_mdcc = sorted(mdcc_pool, key=lambda row: str(row["id"]))
    random.Random(args.seed).shuffle(ordered_mdcc)
    sampled_mdcc: list[dict[str, Any]] = []
    for rank, row in enumerate(ordered_mdcc[: args.mdcc_sample_count]):
        sampled_mdcc.append(
            {
                **row,
                "sampling_seed": args.seed,
                "sampling_rank": rank,
                "sampling_policy": "sorted-id-shuffle-without-replacement",
            }
        )

    selected_ids = {str(row["id"]) for row in sampled_mdcc}
    if len(selected_ids) != args.mdcc_sample_count:
        raise RuntimeError("MDCC sampling unexpectedly produced duplicate ids")
    combined = [dict(row) for row in official] + sampled_mdcc
    require_unique_ids(combined, "combined training manifest")

    official_audio = {
        fingerprint
        for row in official
        for fingerprint in row_fingerprints(row)
        if fingerprint.startswith("audio_sha256:")
    }
    mdcc_audio = {
        fingerprint
        for row in sampled_mdcc
        for fingerprint in row_fingerprints(row)
        if fingerprint.startswith("audio_sha256:")
    }
    if official_audio & mdcc_audio:
        raise ValueError("official and selected MDCC audio hashes overlap")

    excluded_counts: dict[str, int] = {}
    protected: set[str] = set()
    for manifest in args.excluded_manifest:
        rows = read_jsonl(manifest)
        excluded_counts[str(manifest)] = len(rows)
        protected.update(collect_fingerprints(rows))
    overlaps: list[dict[str, str]] = []
    for row in combined:
        for fingerprint in sorted(row_fingerprints(row) & protected):
            overlaps.append({"id": str(row["id"]), "fingerprint": fingerprint})
    if overlaps:
        raise ValueError(
            f"Found {len(overlaps)} protected overlaps; first={overlaps[0]}"
        )

    random.Random(args.seed + 1).shuffle(combined)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = args.output_dir / "train.jsonl"
    smoke_manifest = args.output_dir / "smoke32.jsonl"
    write_jsonl(output_manifest, combined)
    write_jsonl(smoke_manifest, combined[:32])

    round2_mdcc_ids = {
        str(row["id"])
        for row in round2
        if str(row.get("source", "")).strip() == "mdcc"
    }
    source_counts = Counter(str(row["source"]) for row in combined)
    report = {
        "passed": True,
        "policy": {
            "purpose": "replace Round 2 Common Voice membership with MDCC",
            "official_rows_always_retained": True,
            "common_voice_rows": 0,
            "mdcc_sampling": "sorted-id-shuffle-without-replacement",
            "final_shuffle": "python random.Random(seed + 1)",
            "excluded_surfaces": [str(path) for path in args.excluded_manifest],
        },
        "sampling": {
            "seed": args.seed,
            "without_replacement": True,
            "mdcc_pool_rows": len(mdcc_pool),
            "mdcc_selected_rows": len(sampled_mdcc),
            "mdcc_pool_coverage": len(sampled_mdcc) / len(mdcc_pool),
            "mdcc_round2_rows": len(round2_mdcc_ids),
            "mdcc_round2_intersection": len(selected_ids & round2_mdcc_ids),
        },
        "inputs": {
            "official_train": {
                "path": str(args.official_train),
                "rows": len(official),
                "sha256": official_sha256,
            },
            "mdcc_pool": {
                "path": str(args.mdcc_train),
                "rows": len(mdcc_pool),
                "sha256": mdcc_sha256,
            },
            "round2_manifest": {
                "path": str(args.round2_manifest),
                "rows": len(round2),
                "sha256": round2_sha256,
            },
            "excluded": excluded_counts,
        },
        "output": {
            "path": str(output_manifest),
            "rows": len(combined),
            "unique_ids": len({str(row["id"]) for row in combined}),
            "sources": dict(sorted(source_counts.items())),
            "hours": round(
                sum(float(row["duration_s"]) for row in combined) / 3600,
                6,
            ),
            "sha256": sha256_file(output_manifest),
        },
        "smoke": {
            "path": str(smoke_manifest),
            "rows": len(combined[:32]),
            "sha256": sha256_file(smoke_manifest),
        },
    }
    report_path = args.output_dir / "data_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
