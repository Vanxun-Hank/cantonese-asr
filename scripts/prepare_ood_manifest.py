#!/usr/bin/env python3
"""Build a deterministic, leak-resistant OOD panel from held-out publisher splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cantonese_asr.io import read_jsonl, sha256_file, write_jsonl
from scripts.prepare_external_manifest import (
    audio_info_bytes,
    clean_text,
    leakage_key,
    materialize_audio,
    read_cv_durations,
    relative_or_absolute,
    sha256_bytes,
    sha256_path,
    source_report,
    valid_duration,
)


SPLIT_ORDER = (
    ("common_voice_26_zh_HK", "dev"),
    ("common_voice_26_zh_HK", "test"),
    ("mdcc", "validation"),
    ("mdcc", "test"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--common-voice-root", type=Path, required=True)
    parser.add_argument("--common-voice-archive", type=Path, required=True)
    parser.add_argument("--common-voice-audio-root", type=Path, required=True)
    parser.add_argument("--mdcc-parquet-root", type=Path, required=True)
    parser.add_argument("--mdcc-audio-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--excluded-manifest",
        type=Path,
        action="append",
        default=[],
        help="Training/validation/public-test manifest forbidden from OOD diagnostics.",
    )
    parser.add_argument("--panel-per-split", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-duration", type=float, default=30.0)
    return parser.parse_args()


def resolve_manifest_audio(row: dict[str, Any], project_root: Path) -> Path | None:
    value = str(row.get("audio_path") or "").strip()
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_exclusions(
    paths: Iterable[Path], project_root: Path
) -> tuple[set[str], set[str], set[str], dict[str, int]]:
    """Return forbidden normalized texts, audio hashes and IDs."""
    text_keys: set[str] = set()
    audio_hashes: set[str] = set()
    sample_ids: set[str] = set()
    counts = Counter()
    for manifest in paths:
        for row in read_jsonl(manifest):
            counts["rows"] += 1
            sample_id = str(row.get("id") or "").strip()
            if sample_id:
                sample_ids.add(sample_id)
            text = row.get("text", row.get("ref_text", ""))
            key = leakage_key(text)
            if key:
                text_keys.add(key)
            audio_hash = str(row.get("audio_sha256") or "").strip()
            if not audio_hash:
                audio_path = resolve_manifest_audio(row, project_root)
                if audio_path and audio_path.is_file():
                    audio_hash = sha256_path(audio_path)
            if audio_hash:
                audio_hashes.add(audio_hash)
    counts["unique_text_keys"] = len(text_keys)
    counts["unique_audio_sha256"] = len(audio_hashes)
    counts["unique_ids"] = len(sample_ids)
    return text_keys, audio_hashes, sample_ids, dict(counts)


class CommonVoiceArchive:
    """Read individual Common Voice clips from the original archive safely."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.archive = tarfile.open(path, "r:gz")
        self.members: dict[str, tarfile.TarInfo] = {}
        for member in self.archive.getmembers():
            marker = "/clips/"
            if member.isfile() and marker in member.name:
                filename = member.name.rsplit(marker, 1)[1]
                if "/" not in filename:
                    self.members[filename] = member

    def read(self, filename: str) -> bytes:
        member = self.members.get(filename)
        if member is None:
            raise FileNotFoundError(f"Clip not found in {self.path}: {filename}")
        handle = self.archive.extractfile(member)
        if handle is None:
            raise OSError(f"Unable to extract {member.name}")
        return handle.read()

    def close(self) -> None:
        self.archive.close()

    def __enter__(self) -> "CommonVoiceArchive":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def reject_reason(
    sample_id: str,
    text: str,
    text_key: str,
    duration: float | None,
    seen_ids: set[str],
    seen_text: set[str],
    max_duration: float,
) -> str:
    if sample_id in seen_ids:
        return "duplicate_source_id"
    if not text or not text_key:
        return "empty_text"
    if text_key in seen_text:
        return "excluded_or_duplicate_text"
    if duration is None:
        return "missing_duration"
    if not valid_duration(duration, max_duration):
        return "invalid_duration"
    return ""


def prepare_common_voice_split(
    split: str,
    root: Path,
    archive: CommonVoiceArchive,
    audio_root: Path,
    project_root: Path,
    durations: dict[str, float],
    seen_text: set[str],
    seen_audio: set[str],
    seen_ids: set[str],
    max_duration: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    tsv = root / f"{split}.tsv"
    accepted: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    raw_rows = 0
    with tsv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {"path", "sentence"} <= set(reader.fieldnames):
            raise ValueError(f"Unexpected Common Voice schema: {reader.fieldnames}")
        for row_no, raw in enumerate(reader, start=2):
            raw_rows += 1
            filename = str(raw.get("path") or "").strip()
            sample_id = f"common_voice_26_zh_HK:{filename}"
            text_raw = str(raw.get("sentence") or "")
            text = clean_text(text_raw)
            text_key = leakage_key(text)
            duration = durations.get(filename)
            target = audio_root / split / filename
            base = {
                "id": sample_id,
                "text_raw": text_raw,
                "text": text,
                "audio_path": relative_or_absolute(target, project_root),
                "duration_s": duration,
                "source": "common_voice_26_zh_HK",
                "publisher_split": split,
                "split": "ood",
                "role": "ood_diagnostic",
                "scene": f"ood_common_voice_{split}",
                "source_row": row_no,
                "client_id": str(raw.get("client_id") or ""),
                "gender": str(raw.get("gender") or ""),
                "age": str(raw.get("age") or ""),
            }
            reason = reject_reason(
                sample_id,
                text,
                text_key,
                duration,
                seen_ids,
                seen_text,
                max_duration,
            )
            if reason:
                quarantine.append({**base, "reason": reason})
                seen_ids.add(sample_id)
                continue
            try:
                data = target.read_bytes() if target.is_file() else archive.read(filename)
                decoded_duration, sample_rate, channels = audio_info_bytes(data)
                if not valid_duration(decoded_duration, max_duration):
                    raise ValueError(f"decoded duration {decoded_duration}")
                audio_hash = sha256_bytes(data)
            except Exception as exc:
                quarantine.append({**base, "reason": "unreadable_audio", "error": str(exc)})
                seen_ids.add(sample_id)
                continue
            if audio_hash in seen_audio:
                quarantine.append(
                    {**base, "reason": "excluded_or_duplicate_audio", "audio_sha256": audio_hash}
                )
                seen_ids.add(sample_id)
                continue
            materialize_audio(target, data, audio_hash)
            seen_ids.add(sample_id)
            seen_text.add(text_key)
            seen_audio.add(audio_hash)
            accepted.append(
                {
                    **base,
                    "duration_s": round(decoded_duration, 6),
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "audio_sha256": audio_hash,
                }
            )
    return accepted, quarantine, raw_rows


def prepare_mdcc_split(
    split: str,
    parquet_root: Path,
    audio_root: Path,
    project_root: Path,
    seen_text: set[str],
    seen_audio: set[str],
    seen_ids: set[str],
    max_duration: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    files = sorted(parquet_root.glob(f"{split}-*.parquet"))
    if not files:
        raise ValueError(f"No MDCC {split} Parquet shards below {parquet_root}")
    accepted: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    raw_rows = 0
    columns = ["id", "sex", "duration", "transcript", "audio"]
    for shard in files:
        parquet = pq.ParquetFile(shard)
        row_index = 0
        for batch in parquet.iter_batches(batch_size=32, columns=columns):
            for raw in batch.to_pylist():
                raw_rows += 1
                source_id = str(raw.get("id"))
                sample_id = f"mdcc:{shard.stem}:{source_id}"
                text_raw = str(raw.get("transcript") or "")
                text = clean_text(text_raw)
                text_key = leakage_key(text)
                duration = float(raw.get("duration") or 0.0)
                audio = raw.get("audio") or {}
                audio_bytes = audio.get("bytes")
                target = audio_root / split / shard.stem / f"{row_index:08d}.wav"
                base = {
                    "id": sample_id,
                    "text_raw": text_raw,
                    "text": text,
                    "audio_path": relative_or_absolute(target, project_root),
                    "duration_s": duration,
                    "source": "mdcc",
                    "publisher_split": split,
                    "split": "ood",
                    "role": "ood_diagnostic",
                    "scene": f"ood_mdcc_{split}",
                    "source_shard": relative_or_absolute(shard, project_root),
                    "source_row": row_index,
                    "original_audio_path": str(audio.get("path") or ""),
                    "gender": str(raw.get("sex") or ""),
                }
                row_index += 1
                reason = reject_reason(
                    sample_id,
                    text,
                    text_key,
                    duration,
                    seen_ids,
                    seen_text,
                    max_duration,
                )
                if not reason and (not isinstance(audio_bytes, bytes) or not audio_bytes):
                    reason = "missing_audio_bytes"
                if reason:
                    quarantine.append({**base, "reason": reason})
                    seen_ids.add(sample_id)
                    continue
                try:
                    decoded_duration, sample_rate, channels = audio_info_bytes(audio_bytes)
                    if not valid_duration(decoded_duration, max_duration):
                        raise ValueError(f"decoded duration {decoded_duration}")
                    audio_hash = sha256_bytes(audio_bytes)
                except Exception as exc:
                    quarantine.append({**base, "reason": "unreadable_audio", "error": str(exc)})
                    seen_ids.add(sample_id)
                    continue
                if audio_hash in seen_audio:
                    quarantine.append(
                        {**base, "reason": "excluded_or_duplicate_audio", "audio_sha256": audio_hash}
                    )
                    seen_ids.add(sample_id)
                    continue
                materialize_audio(target, audio_bytes, audio_hash)
                seen_ids.add(sample_id)
                seen_text.add(text_key)
                seen_audio.add(audio_hash)
                accepted.append(
                    {
                        **base,
                        "duration_s": round(decoded_duration, 6),
                        "sample_rate": sample_rate,
                        "channels": channels,
                        "audio_sha256": audio_hash,
                    }
                )
    return accepted, quarantine, raw_rows


def stable_group_seed(seed: int, source: str, split: str) -> int:
    digest = hashlib.sha256(f"{seed}:{source}:{split}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def deterministic_panel(
    rows: list[dict[str, Any]], per_split: int, seed: int
) -> list[dict[str, Any]]:
    if per_split < 1:
        raise ValueError("per_split must be positive")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source"]), str(row["publisher_split"]))].append(row)
    selected: list[dict[str, Any]] = []
    for source, split in SPLIT_ORDER:
        candidates = sorted(grouped[(source, split)], key=lambda row: str(row["id"]))
        rng = random.Random(stable_group_seed(seed, source, split))
        if len(candidates) > per_split:
            candidates = rng.sample(candidates, per_split)
        selected.extend(sorted(candidates, key=lambda row: str(row["id"])))
    return selected


def group_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['source']}:{row['publisher_split']}"].append(row)
    return {name: source_report(group) for name, group in sorted(grouped.items())}


def main() -> None:
    args = parse_args()
    if args.panel_per_split < 1 or args.max_duration <= 0:
        raise SystemExit("--panel-per-split and --max-duration must be positive")
    excluded_text, excluded_audio, excluded_ids, exclusion_report = load_exclusions(
        args.excluded_manifest, args.project_root
    )
    seen_text = set(excluded_text)
    seen_audio = set(excluded_audio)
    seen_ids = set(excluded_ids)
    durations = read_cv_durations(args.common_voice_root / "clip_durations.tsv")
    accepted: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    raw_counts: dict[str, int] = {}

    with CommonVoiceArchive(args.common_voice_archive) as archive:
        for split in ("dev", "test"):
            rows, rejected, raw_count = prepare_common_voice_split(
                split,
                args.common_voice_root,
                archive,
                args.common_voice_audio_root,
                args.project_root,
                durations,
                seen_text,
                seen_audio,
                seen_ids,
                args.max_duration,
            )
            accepted.extend(rows)
            quarantine.extend(rejected)
            raw_counts[f"common_voice_26_zh_HK:{split}"] = raw_count
    for split in ("validation", "test"):
        rows, rejected, raw_count = prepare_mdcc_split(
            split,
            args.mdcc_parquet_root,
            args.mdcc_audio_root,
            args.project_root,
            seen_text,
            seen_audio,
            seen_ids,
            args.max_duration,
        )
        accepted.extend(rows)
        quarantine.extend(rejected)
        raw_counts[f"mdcc:{split}"] = raw_count

    accepted_text = {leakage_key(row["text"]) for row in accepted}
    accepted_hashes = {str(row["audio_sha256"]) for row in accepted}
    accepted_ids = {str(row["id"]) for row in accepted}
    if accepted_text & excluded_text:
        raise RuntimeError("Excluded text leaked into OOD manifest")
    if accepted_hashes & excluded_audio:
        raise RuntimeError("Excluded audio leaked into OOD manifest")
    if accepted_ids & excluded_ids:
        raise RuntimeError("Excluded source ID leaked into OOD manifest")
    if len(accepted_text) != len(accepted) or len(accepted_hashes) != len(accepted):
        raise RuntimeError("Duplicate text/audio leaked into OOD manifest")

    panel = deterministic_panel(accepted, args.panel_per_split, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    group_paths: dict[str, Path] = {}
    for source, split in SPLIT_ORDER:
        name = "common_voice" if source.startswith("common_voice") else "mdcc"
        path = args.output_dir / f"{name}_{split}.jsonl"
        group_paths[f"{source}:{split}"] = path
        write_jsonl(
            path,
            [
                row
                for row in accepted
                if row["source"] == source and row["publisher_split"] == split
            ],
        )
    all_path = args.output_dir / "ood_all.jsonl"
    panel_path = args.output_dir / "ood_panel.jsonl"
    quarantine_path = args.output_dir / "quarantine.jsonl"
    write_jsonl(all_path, accepted)
    write_jsonl(panel_path, panel)
    write_jsonl(quarantine_path, quarantine)
    output_paths = {"ood_all": all_path, "ood_panel": panel_path, **group_paths}
    report = {
        "purpose": "OOD diagnostics only; never used for training or model selection",
        "seed": args.seed,
        "panel_per_split": args.panel_per_split,
        "max_duration_s": args.max_duration,
        "inputs": {
            "common_voice_root": str(args.common_voice_root.resolve()),
            "common_voice_archive": str(args.common_voice_archive.resolve()),
            "mdcc_parquet_root": str(args.mdcc_parquet_root.resolve()),
            "excluded_manifests": [str(path.resolve()) for path in args.excluded_manifest],
        },
        "raw_candidate_rows": raw_counts,
        "raw_candidate_total": sum(raw_counts.values()),
        "excluded_boundary": exclusion_report,
        "accepted": {"total": source_report(accepted), "groups": group_report(accepted)},
        "panel": {"total": source_report(panel), "groups": group_report(panel)},
        "quarantine": {
            "rows": len(quarantine),
            "reasons": dict(sorted(Counter(str(row["reason"]) for row in quarantine).items())),
        },
        "leakage_checks": {
            "excluded_text_overlap": len(accepted_text & excluded_text),
            "excluded_audio_overlap": len(accepted_hashes & excluded_audio),
            "excluded_id_overlap": len(accepted_ids & excluded_ids),
            "accepted_duplicate_text": len(accepted) - len(accepted_text),
            "accepted_duplicate_audio": len(accepted) - len(accepted_hashes),
        },
        "manifest_sha256": {
            name: sha256_file(path) for name, path in sorted(output_paths.items())
        },
    }
    (args.output_dir / "data_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
