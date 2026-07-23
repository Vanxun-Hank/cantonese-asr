#!/usr/bin/env python3
"""Prepare leak-resistant Common Voice and MDCC training manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cantonese_asr.io import read_jsonl, write_jsonl
from cantonese_asr.metrics import normalize_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--common-voice-root", type=Path, required=True)
    parser.add_argument("--mdcc-parquet-root", type=Path, required=True)
    parser.add_argument("--mdcc-audio-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--protected-text-jsonl",
        type=Path,
        action="append",
        default=[],
        help="JSONL containing text or ref_text that external data must not match.",
    )
    parser.add_argument("--max-duration", type=float, default=30.0)
    return parser.parse_args()


def clean_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def leakage_key(value: Any) -> str:
    return normalize_prediction(clean_text(value))


def relative_or_absolute(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_keys(paths: Iterable[Path]) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            text = row.get("text", row.get("ref_text", ""))
            key = leakage_key(text)
            if key:
                keys.add(key)
    return keys


def audio_info_bytes(data: bytes) -> tuple[float, int, int]:
    info = sf.info(io.BytesIO(data))
    return float(info.duration), int(info.samplerate), int(info.channels)


def valid_duration(duration: float, maximum: float) -> bool:
    return math.isfinite(duration) and 0 < duration <= maximum


def read_cv_durations(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"clip", "duration[ms]"}
        if not reader.fieldnames or not expected <= set(reader.fieldnames):
            raise ValueError(f"Unexpected Common Voice durations schema: {reader.fieldnames}")
        return {
            str(row["clip"]): float(row["duration[ms]"]) / 1000.0
            for row in reader
        }


def prepare_common_voice(
    root: Path,
    project_root: Path,
    protected: set[str],
    seen_audio: set[str],
    max_duration: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_tsv = root / "train.tsv"
    durations_tsv = root / "clip_durations.tsv"
    clips = root / "clips"
    durations = read_cv_durations(durations_tsv)
    accepted: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with train_tsv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"path", "sentence"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"Unexpected Common Voice schema: {reader.fieldnames}")
        for row_no, raw in enumerate(reader, start=2):
            filename = str(raw.get("path") or "").strip()
            sample_id = f"common_voice_26_zh_HK:{filename}"
            text_raw = str(raw.get("sentence") or "")
            text = clean_text(text_raw)
            key = leakage_key(text)
            duration = durations.get(filename)
            audio_path = clips / filename
            base = {
                "id": sample_id,
                "text_raw": text_raw,
                "text": text,
                "audio_path": relative_or_absolute(audio_path, project_root),
                "duration_s": duration,
                "source": "common_voice_26_zh_HK",
                "publisher_split": "train",
                "scene": "external_common_voice",
                "source_row": row_no,
                "client_id": str(raw.get("client_id") or ""),
                "gender": str(raw.get("gender") or ""),
                "age": str(raw.get("age") or ""),
            }
            reason = ""
            if sample_id in seen_ids:
                reason = "duplicate_source_id"
            elif not text or not key:
                reason = "empty_text"
            elif key in protected:
                reason = "protected_text_overlap"
            elif duration is None:
                reason = "missing_duration"
            elif not valid_duration(duration, max_duration):
                reason = "invalid_duration"
            elif not audio_path.is_file():
                reason = "missing_audio"
            if reason:
                quarantine.append({**base, "reason": reason})
                seen_ids.add(sample_id)
                continue
            try:
                info = sf.info(str(audio_path))
                if not valid_duration(float(info.duration), max_duration):
                    raise ValueError(f"decoded duration {info.duration}")
                audio_hash = sha256_path(audio_path)
            except Exception as exc:
                quarantine.append({**base, "reason": "unreadable_audio", "error": str(exc)})
                seen_ids.add(sample_id)
                continue
            if audio_hash in seen_audio:
                quarantine.append({**base, "reason": "duplicate_audio_sha256", "audio_sha256": audio_hash})
                seen_ids.add(sample_id)
                continue
            seen_ids.add(sample_id)
            seen_audio.add(audio_hash)
            accepted.append(
                {
                    **base,
                    "duration_s": round(float(info.duration), 6),
                    "sample_rate": int(info.samplerate),
                    "channels": int(info.channels),
                    "audio_sha256": audio_hash,
                    "split": "train",
                }
            )
    return accepted, quarantine


def materialize_audio(target: Path, data: bytes, expected_sha256: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == len(data):
        if sha256_path(target) == expected_sha256:
            return
    partial = target.with_name(f"{target.name}.part")
    with partial.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(target)


def prepare_mdcc(
    parquet_root: Path,
    audio_root: Path,
    project_root: Path,
    protected: set[str],
    seen_audio: set[str],
    max_duration: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    files = sorted(parquet_root.glob("train-*.parquet"))
    if not files:
        raise ValueError(f"No MDCC train Parquet shards below {parquet_root}")
    accepted: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    columns = ["id", "sex", "duration", "transcript", "audio"]
    for shard in files:
        parquet = pq.ParquetFile(shard)
        row_index = 0
        for batch in parquet.iter_batches(batch_size=32, columns=columns):
            for raw in batch.to_pylist():
                source_id = str(raw.get("id"))
                sample_id = f"mdcc:{shard.stem}:{source_id}"
                text_raw = str(raw.get("transcript") or "")
                text = clean_text(text_raw)
                key = leakage_key(text)
                duration = float(raw.get("duration") or 0.0)
                audio = raw.get("audio") or {}
                audio_bytes = audio.get("bytes")
                target = audio_root / shard.stem / f"{row_index:08d}.wav"
                base = {
                    "id": sample_id,
                    "text_raw": text_raw,
                    "text": text,
                    "audio_path": relative_or_absolute(target, project_root),
                    "duration_s": duration,
                    "source": "mdcc",
                    "publisher_split": "train",
                    "scene": "external_mdcc",
                    "source_shard": relative_or_absolute(shard, project_root),
                    "source_row": row_index,
                    "original_audio_path": str(audio.get("path") or ""),
                    "gender": str(raw.get("sex") or ""),
                }
                row_index += 1
                reason = ""
                if sample_id in seen_ids:
                    reason = "duplicate_source_id"
                elif not text or not key:
                    reason = "empty_text"
                elif key in protected:
                    reason = "protected_text_overlap"
                elif not valid_duration(duration, max_duration):
                    reason = "invalid_duration"
                elif not isinstance(audio_bytes, bytes) or not audio_bytes:
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
                    quarantine.append({**base, "reason": "duplicate_audio_sha256", "audio_sha256": audio_hash})
                    seen_ids.add(sample_id)
                    continue
                materialize_audio(target, audio_bytes, audio_hash)
                seen_ids.add(sample_id)
                seen_audio.add(audio_hash)
                accepted.append(
                    {
                        **base,
                        "duration_s": round(decoded_duration, 6),
                        "sample_rate": sample_rate,
                        "channels": channels,
                        "audio_sha256": audio_hash,
                        "split": "train",
                    }
                )
    return accepted, quarantine


def source_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [float(row["duration_s"]) for row in rows]
    return {
        "rows": len(rows),
        "hours": round(sum(durations) / 3600, 4),
        "unique_text_keys": len({leakage_key(row["text"]) for row in rows}),
        "sample_rates": dict(sorted(Counter(str(row["sample_rate"]) for row in rows).items())),
        "channels": dict(sorted(Counter(str(row["channels"]) for row in rows).items())),
    }


def main() -> None:
    args = parse_args()
    if args.max_duration <= 0:
        raise SystemExit("--max-duration must be positive")
    protected = protected_keys(args.protected_text_jsonl)
    seen_audio: set[str] = set()
    common_voice, quarantine_cv = prepare_common_voice(
        args.common_voice_root,
        args.project_root,
        protected,
        seen_audio,
        args.max_duration,
    )
    mdcc, quarantine_mdcc = prepare_mdcc(
        args.mdcc_parquet_root,
        args.mdcc_audio_root,
        args.project_root,
        protected,
        seen_audio,
        args.max_duration,
    )
    all_rows = common_voice + mdcc
    quarantine = quarantine_cv + quarantine_mdcc
    accepted_keys = {leakage_key(row["text"]) for row in all_rows}
    if accepted_keys & protected:
        raise RuntimeError("Protected text leaked into accepted external manifest")
    hashes = [str(row["audio_sha256"]) for row in all_rows]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("Duplicate audio hash leaked into accepted external manifest")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "common_voice_train.jsonl", common_voice)
    write_jsonl(args.output_dir / "mdcc_train.jsonl", mdcc)
    write_jsonl(args.output_dir / "external_all.jsonl", all_rows)
    write_jsonl(args.output_dir / "quarantine.jsonl", quarantine)
    report = {
        "inputs": {
            "common_voice_root": str(args.common_voice_root.resolve()),
            "mdcc_parquet_root": str(args.mdcc_parquet_root.resolve()),
            "protected_text_jsonl": [
                str(path.resolve()) for path in args.protected_text_jsonl
            ],
        },
        "provenance": {
            "common_voice": {
                "dataset": "Common Voice Scripted Speech 26.0 - Chinese (Hong Kong)",
                "dataset_id": "cmqinoe3p00wonr07fumnrmtg",
                "publisher_split_used": "train",
            },
            "mdcc": {
                "dataset": "Multi-Domain Cantonese Corpus",
                "publisher_split_used": "train",
                "license_status": "user_reported_signed; proof file not stored in project",
                "download_wrapper": "ming030890/mdcc",
            },
        },
        "protected_text_keys": len(protected),
        "max_duration_s": args.max_duration,
        "sources": {
            "common_voice_26_zh_HK": source_report(common_voice),
            "mdcc": source_report(mdcc),
        },
        "total": source_report(all_rows),
        "quarantine": {
            "rows": len(quarantine),
            "reasons": dict(sorted(Counter(str(row["reason"]) for row in quarantine).items())),
        },
        "leakage_checks": {
            "protected_text_overlap": len(accepted_keys & protected),
            "duplicate_audio_sha256": len(hashes) - len(set(hashes)),
        },
    }
    (args.output_dir / "data_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
