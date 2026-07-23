#!/usr/bin/env python3
"""Join official WAV files to index.csv and build leak-resistant manifests."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cantonese_asr.io import read_test_rows, sha256_file, write_jsonl


REQUIRED_COLUMNS = ("序号", "粤语原文", "普通话翻译", "香港语言学会粤拼", "场景")
# Official WAV names use a fixed five-digit, zero-padded corpus ID. The spoken
# text may begin with another digit immediately afterwards (for example
# ``1003211号...wav`` means ID 10032 plus text beginning with "11号").
LEADING_ID = re.compile(r"^\s*(\d{5})")
INTEGERISH_ID = re.compile(r"^(\d+)(?:\.0+)?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-csv", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-probe-size", type=int, default=256)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument(
        "--exclude-test-list",
        type=Path,
        action="append",
        default=[],
        help="Public test JSONL/CSV whose leading audio IDs must not enter SFT.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def normalize_id(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    match = INTEGERISH_ID.fullmatch(text)
    if not match:
        return ""
    return str(int(match.group(1)))


def extract_audio_id(path: Path) -> str:
    match = LEADING_ID.match(unicodedata.normalize("NFKC", path.stem))
    return str(int(match.group(1))) if match else ""


def read_excluded_ids(paths: list[Path]) -> set[str]:
    excluded: dict[str, tuple[Path, int]] = {}
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Excluded test list not found: {path}")
        for line_no, row in enumerate(read_test_rows(path), start=1):
            sample_id = extract_audio_id(Path(str(row["audio_path"])))
            if not sample_id:
                raise ValueError(
                    f"Cannot derive leading numeric ID from {path} row {line_no}: "
                    f"{row['audio_path']!r}"
                )
            if sample_id in excluded:
                previous_path, previous_line = excluded[sample_id]
                raise ValueError(
                    f"Duplicate excluded ID {sample_id}: "
                    f"{previous_path}:{previous_line} and {path}:{line_no}"
                )
            excluded[sample_id] = (path, line_no)
    return set(excluded)


def path_for_manifest(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def read_index(path: Path) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    by_id: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    quarantine: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Empty index CSV: {path}")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing index.csv columns {missing}; got {reader.fieldnames}")
        for line_no, raw in enumerate(reader, start=2):
            sample_id = normalize_id(raw.get("序号"))
            row = {
                "id": sample_id,
                "text": clean_text(raw.get("粤语原文")),
                "mandarin": clean_text(raw.get("普通话翻译")),
                "jyutping": clean_text(raw.get("香港语言学会粤拼")),
                "scene": clean_text(raw.get("场景")),
            }
            if not sample_id:
                quarantine.append({**row, "reason": "invalid_csv_id", "csv_line": line_no})
                continue
            by_id[sample_id].append(row)

    unique: dict[str, dict[str, str]] = {}
    for sample_id, rows in by_id.items():
        if len(rows) != 1:
            quarantine.extend(
                {**row, "reason": "duplicate_csv_id", "duplicate_count": len(rows)}
                for row in rows
            )
        else:
            unique[sample_id] = rows[0]
    return unique, quarantine


def scan_audio(root: Path) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    by_id: defaultdict[str, list[Path]] = defaultdict(list)
    quarantine: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.suffix.lower() == ".wav"):
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        sample_id = extract_audio_id(path)
        if not sample_id:
            quarantine.append(
                {"id": "", "audio_path": str(path.resolve()), "reason": "missing_audio_id"}
            )
            continue
        by_id[sample_id].append(path)

    unique: dict[str, Path] = {}
    for sample_id, paths in by_id.items():
        if len(paths) != 1:
            quarantine.extend(
                {
                    "id": sample_id,
                    "audio_path": str(path.resolve()),
                    "reason": "duplicate_audio_id",
                    "duplicate_count": len(paths),
                }
                for path in paths
            )
        else:
            unique[sample_id] = paths[0]
    return unique, quarantine


def inspect_pair(
    sample: dict[str, str],
    wav: Path,
    project_root: Path,
    max_duration: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    base = {
        **sample,
        "audio_path": path_for_manifest(wav, project_root),
        "source": "official",
    }
    if not sample["text"]:
        return None, {**base, "reason": "empty_label"}
    try:
        info = sf.info(str(wav))
    except Exception as exc:  # libsndfile exposes multiple exception classes
        return None, {**base, "reason": "unreadable_audio", "error": str(exc)}
    duration = float(info.duration)
    if not math.isfinite(duration) or duration <= 0:
        return None, {**base, "reason": "nonpositive_duration", "duration_s": duration}
    if duration > max_duration:
        return None, {
            **base,
            "reason": "over_whisper_window",
            "duration_s": duration,
            "max_duration_s": max_duration,
        }
    return (
        {
            **base,
            "duration_s": round(duration, 6),
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "split": "",
        },
        None,
    )


def stratified_group_split(
    rows: list[dict[str, Any]], validation_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[clean_text(row["text"])].append(row)

    by_scene: defaultdict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for text_key, group_rows in grouped.items():
        scene = str(group_rows[0].get("scene") or "unknown")
        by_scene[scene].append((text_key, group_rows))

    rng = random.Random(seed)
    validation_keys: set[str] = set()
    for scene in sorted(by_scene):
        groups = sorted(by_scene[scene], key=lambda item: item[0])
        rng.shuffle(groups)
        if len(groups) <= 1:
            continue
        scene_rows = sum(len(group_rows) for _, group_rows in groups)
        target = max(1, round(scene_rows * validation_ratio))
        selected = 0
        for text_key, group_rows in groups[:-1]:
            if selected >= target:
                break
            validation_keys.add(text_key)
            selected += len(group_rows)

    if not validation_keys and len(grouped) > 1:
        validation_keys.add(sorted(grouped)[0])

    train_rows, validation_rows = [], []
    for row in rows:
        target = validation_rows if clean_text(row["text"]) in validation_keys else train_rows
        target.append({**row, "split": "validation" if target is validation_rows else "train"})
    return train_rows, validation_rows


def stratified_probe(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    if size <= 0 or not rows:
        return []
    buckets: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("scene") or "unknown")].append(row)
    rng = random.Random(seed + 1)
    for values in buckets.values():
        rng.shuffle(values)
    scenes = sorted(buckets)
    probe: list[dict[str, Any]] = []
    while len(probe) < min(size, len(rows)):
        progressed = False
        for scene in scenes:
            if buckets[scene] and len(probe) < size:
                probe.append(buckets[scene].pop())
                progressed = True
        if not progressed:
            break
    return [{**row, "split": "train_probe"} for row in probe]


def distribution(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "")) for row in rows).items()))


def duration_histogram(durations: Iterable[float]) -> dict[str, int]:
    edges = (0, 2, 4, 6, 8, 10, 15, 20, 25, 30)
    counts: Counter[str] = Counter()
    for duration in durations:
        for lower, upper in zip(edges[:-1], edges[1:]):
            if lower <= duration < upper or (upper == edges[-1] and duration <= upper):
                counts[f"{lower}-{upper}"] += 1
                break
    return {f"{lower}-{upper}": counts[f"{lower}-{upper}"] for lower, upper in zip(edges[:-1], edges[1:])}


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_ratio < 1:
        raise SystemExit("--validation-ratio must be between 0 and 1")
    if args.max_duration <= 0:
        raise SystemExit("--max-duration must be positive")
    if not args.index_csv.is_file():
        raise SystemExit(f"index.csv not found: {args.index_csv}")
    if not args.audio_root.is_dir():
        raise SystemExit(f"Audio root not found: {args.audio_root}")

    index_rows, quarantine = read_index(args.index_csv)
    audio_rows, audio_quarantine = scan_audio(args.audio_root)
    quarantine.extend(audio_quarantine)
    excluded_ids = read_excluded_ids(args.exclude_test_list)
    excluded_missing_index = sorted(
        excluded_ids - set(index_rows), key=lambda value: int(value)
    )
    excluded_missing_audio = sorted(
        excluded_ids - set(audio_rows), key=lambda value: int(value)
    )
    if excluded_missing_index or excluded_missing_audio:
        raise SystemExit(
            "Public exclusion list did not map one-to-one to official data: "
            + json.dumps(
                {
                    "missing_index_ids": excluded_missing_index[:20],
                    "missing_audio_ids": excluded_missing_audio[:20],
                },
                ensure_ascii=False,
            )
        )

    valid_rows: list[dict[str, Any]] = []
    public_excluded_rows: list[dict[str, Any]] = []
    all_ids = sorted(set(index_rows) | set(audio_rows), key=lambda value: int(value))
    for sample_id in all_ids:
        sample = index_rows.get(sample_id)
        wav = audio_rows.get(sample_id)
        if sample_id in excluded_ids:
            assert sample is not None and wav is not None
            public_excluded_rows.append(
                {
                    **sample,
                    "audio_path": path_for_manifest(wav, Path.cwd()),
                    "split": "public_excluded",
                    "source": "official",
                    "reason": "listed_in_public_test",
                }
            )
            continue
        if sample is None:
            quarantine.append(
                {
                    "id": sample_id,
                    "audio_path": str(wav.resolve()) if wav else "",
                    "reason": "audio_without_csv_row",
                }
            )
            continue
        if wav is None:
            quarantine.append({**sample, "reason": "csv_row_without_audio"})
            continue
        valid, rejected = inspect_pair(sample, wav, Path.cwd(), args.max_duration)
        if rejected:
            quarantine.append(rejected)
        elif valid:
            valid_rows.append(valid)

    if len(valid_rows) < 2:
        raise SystemExit(f"Need at least two valid pairs; found {len(valid_rows)}")

    train_rows, validation_rows = stratified_group_split(
        valid_rows, args.validation_ratio, args.seed
    )
    train_probe_rows = stratified_probe(train_rows, args.train_probe_size, args.seed)
    all_rows = sorted(train_rows + validation_rows, key=lambda row: int(row["id"]))
    quarantine = sorted(
        quarantine,
        key=lambda row: (str(row.get("id", "")), str(row.get("reason", ""))),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths = {
        "all": args.output_dir / "all.jsonl",
        "train": args.output_dir / "train.jsonl",
        "validation": args.output_dir / "validation.jsonl",
        "train_probe": args.output_dir / "train_probe.jsonl",
        "public_excluded": args.output_dir / "public_excluded.jsonl",
        "quarantine": args.output_dir / "quarantine.jsonl",
    }
    for name, rows in (
        ("all", all_rows),
        ("train", train_rows),
        ("validation", validation_rows),
        ("train_probe", train_probe_rows),
        ("public_excluded", public_excluded_rows),
        ("quarantine", quarantine),
    ):
        write_jsonl(manifest_paths[name], rows)

    durations = [float(row["duration_s"]) for row in all_rows]
    train_texts = {clean_text(row["text"]) for row in train_rows}
    validation_texts = {clean_text(row["text"]) for row in validation_rows}
    report = {
        "inputs": {
            "index_csv": str(args.index_csv.resolve()),
            "index_csv_sha256": sha256_file(args.index_csv),
            "audio_root": str(args.audio_root.resolve()),
            "exclude_test_lists": [
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
                for path in args.exclude_test_list
            ],
        },
        "parameters": {
            "validation_ratio": args.validation_ratio,
            "seed": args.seed,
            "train_probe_size": args.train_probe_size,
            "max_duration_s": args.max_duration,
        },
        "counts": {
            "csv_unique_ids": len(index_rows),
            "audio_unique_ids": len(audio_rows),
            "valid": len(all_rows),
            "train": len(train_rows),
            "validation": len(validation_rows),
            "train_probe": len(train_probe_rows),
            "public_excluded": len(public_excluded_rows),
            "quarantine": len(quarantine),
        },
        "leakage_checks": {
            "normalized_text_overlap": len(train_texts & validation_texts),
            "audio_path_overlap": len(
                {row["audio_path"] for row in train_rows}
                & {row["audio_path"] for row in validation_rows}
            ),
            "public_excluded_id_overlap": len(
                excluded_ids
                & {str(row["id"]) for row in train_rows + validation_rows}
            ),
        },
        "duration_s": {
            "total_hours": sum(durations) / 3600,
            "min": min(durations),
            "median": statistics.median(durations),
            "mean": statistics.fmean(durations),
            "max": max(durations),
        },
        "duration_histogram": duration_histogram(durations),
        "scene_distribution": {
            "train": distribution(train_rows, "scene"),
            "validation": distribution(validation_rows, "scene"),
            "train_probe": distribution(train_probe_rows, "scene"),
        },
        "sample_rates": distribution(all_rows, "sample_rate"),
        "channels": distribution(all_rows, "channels"),
        "quarantine_reasons": distribution(quarantine, "reason"),
        "manifest_sha256": {
            name: sha256_file(path) for name, path in manifest_paths.items()
        },
    }
    report_path = args.output_dir / "data_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
