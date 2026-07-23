#!/usr/bin/env python3
"""Build deterministic, source-balanced external-data SFT manifests."""

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
    parser.add_argument("--official-train", type=Path, required=True)
    parser.add_argument("--common-voice", type=Path, required=True)
    parser.add_argument("--mdcc", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--external-fraction",
        type=float,
        action="append",
        default=None,
        help="External fraction of final epoch rows; may be repeated.",
    )
    parser.add_argument(
        "--nested-fractions",
        action="store_true",
        help="Smaller external fraction is a subset of larger feasible fractions.",
    )
    parser.add_argument(
        "--include-all-external",
        action="store_true",
        help="Write all_train.jsonl with every accepted publisher-train row.",
    )
    parser.add_argument(
        "--preadapt-official-multiple",
        type=float,
        default=2.0,
        help="External pre-adaptation rows as a multiple of official rows.",
    )
    return parser.parse_args()


def validate_unique_ids(rows: list[dict[str, Any]], label: str) -> None:
    ids = [str(row.get("id", "")) for row in rows]
    if not all(ids):
        raise ValueError(f"{label} contains an empty id")
    duplicates = len(ids) - len(set(ids))
    if duplicates:
        raise ValueError(f"{label} contains {duplicates} duplicate ids")


def balanced_counts(total: int, sizes: list[int]) -> list[int]:
    if total < 0:
        raise ValueError("sample count cannot be negative")
    if not sizes or any(size < 0 for size in sizes):
        raise ValueError("source sizes must be non-negative")
    counts = [0] * len(sizes)
    remaining = total
    while remaining:
        progressed = False
        for index, size in enumerate(sizes):
            if counts[index] < size and remaining:
                counts[index] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise ValueError(f"Requested {total} rows but only {sum(sizes)} exist")
    return counts


def sample_sources(
    sources: list[list[dict[str, Any]]], total: int, seed: int
) -> list[dict[str, Any]]:
    counts = balanced_counts(total, [len(source) for source in sources])
    sampled: list[dict[str, Any]] = []
    for source_index, (source, count) in enumerate(zip(sources, counts)):
        ordered = sorted(source, key=lambda row: str(row["id"]))
        rng = random.Random(seed + 1009 * source_index)
        sampled.extend(rng.sample(ordered, count))
    return sampled


def sample_sources_prefix(
    sources: list[list[dict[str, Any]]], total: int, seed: int
) -> list[dict[str, Any]]:
    counts = balanced_counts(total, [len(source) for source in sources])
    sampled: list[dict[str, Any]] = []
    for source_index, (source, count) in enumerate(zip(sources, counts)):
        ordered = sorted(source, key=lambda row: str(row["id"]))
        random.Random(seed + 1009 * source_index).shuffle(ordered)
        sampled.extend(ordered[:count])
    return sampled


def shuffled(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    result = [dict(row) for row in rows]
    random.Random(seed).shuffle(result)
    return result


def manifest_summary(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    sources = Counter(str(row.get("source") or "official") for row in rows)
    hours = sum(float(row.get("duration_s") or 0.0) for row in rows) / 3600
    return {
        "path": str(path),
        "rows": len(rows),
        "hours": round(hours, 4),
        "sources": dict(sorted(sources.items())),
        "sha256": sha256_file(path),
    }


def fraction_name(value: float) -> str:
    return f"external{round(value * 100):02d}"


def main() -> None:
    args = parse_args()
    fractions = list(dict.fromkeys(args.external_fraction or [0.25, 0.5]))
    if any(not 0 < value < 1 for value in fractions):
        raise SystemExit("--external-fraction values must be between 0 and 1")
    if args.preadapt_official_multiple <= 0:
        raise SystemExit("--preadapt-official-multiple must be positive")

    official = read_jsonl(args.official_train)
    common_voice = read_jsonl(args.common_voice)
    mdcc = read_jsonl(args.mdcc)
    validate_unique_ids(official, "official")
    validate_unique_ids(common_voice, "Common Voice")
    validate_unique_ids(mdcc, "MDCC")
    all_ids = [str(row["id"]) for row in official + common_voice + mdcc]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("IDs overlap across sources")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, dict[str, Any]] = {}

    control_path = args.output_dir / "control_official.jsonl"
    control_rows = shuffled(official, args.seed)
    write_jsonl(control_path, control_rows)
    manifests["control_official"] = manifest_summary(control_path, control_rows)

    for offset, fraction in enumerate(fractions, start=1):
        external_count = round(len(official) * fraction / (1.0 - fraction))
        if args.nested_fractions:
            external = sample_sources_prefix(
                [common_voice, mdcc], external_count, args.seed + 50_000
            )
        else:
            external = sample_sources(
                [common_voice, mdcc], external_count, args.seed + offset * 10_000
            )
        rows = shuffled(official + external, args.seed + offset)
        name = fraction_name(fraction)
        path = args.output_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        summary = manifest_summary(path, rows)
        summary["requested_external_fraction"] = fraction
        summary["actual_external_fraction"] = len(external) / len(rows)
        manifests[name] = summary

    if args.include_all_external:
        all_train_rows = shuffled(
            official + common_voice + mdcc, args.seed + 80_000
        )
        all_train_path = args.output_dir / "all_train.jsonl"
        write_jsonl(all_train_path, all_train_rows)
        summary = manifest_summary(all_train_path, all_train_rows)
        summary["requested_external_fraction"] = "all"
        summary["actual_external_fraction"] = (
            len(common_voice) + len(mdcc)
        ) / len(all_train_rows)
        manifests["all_train"] = summary

    preadapt_count = round(len(official) * args.preadapt_official_multiple)
    preadapt_rows = shuffled(
        sample_sources(
            [common_voice, mdcc], preadapt_count, args.seed + 90_000
        ),
        args.seed + 99,
    )
    preadapt_path = args.output_dir / "preadapt_external.jsonl"
    write_jsonl(preadapt_path, preadapt_rows)
    manifests["preadapt_external"] = manifest_summary(
        preadapt_path, preadapt_rows
    )

    report = {
        "seed": args.seed,
        "sampling_policy": {
            "official_rows_always_retained": True,
            "external_sources_balanced_by_row_count": True,
            "sampling_without_replacement": True,
            "external_fraction_definition": "external_rows / total_epoch_rows",
            "nested_external_fractions": args.nested_fractions,
            "all_external_rows_included": args.include_all_external,
        },
        "input_counts": {
            "official": len(official),
            "common_voice_26_zh_HK": len(common_voice),
            "mdcc": len(mdcc),
        },
        "manifests": manifests,
    }
    (args.output_dir / "mix_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
