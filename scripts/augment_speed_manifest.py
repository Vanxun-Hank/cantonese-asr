#!/usr/bin/env python3
"""Create a verified 0.9/1.0/1.1 speed-perturbation manifest.

This script intentionally builds data only. It never imports training code or
launches model work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import soundfile as sf
from scipy.signal import resample_poly

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cantonese_asr.io import read_jsonl, sha256_file, write_jsonl
from cantonese_asr.metrics import normalize_prediction

FACTORS: tuple[Fraction, ...] = (
    Fraction("0.9"),
    Fraction("1.0"),
    Fraction("1.1"),
)
TARGET_SAMPLE_RATE = 16_000
MAX_DURATION_S = 30.0
DURATION_TOLERANCE_S = 0.05


@dataclass(frozen=True)
class BuildConfig:
    train_manifest: Path
    validation_manifest: Path
    public_manifest: Path
    output_audio_root: Path
    output_manifest_dir: Path
    project_root: Path
    expected_parents: int
    workers: int
    version_tag: str


@dataclass
class ParentResult:
    parent_id: str
    rows: list[dict[str, Any]]
    error: dict[str, Any] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build official-only 0.9/1.0/1.1 speed-perturbed data."
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--output-audio-root", type=Path, required=True)
    parser.add_argument("--output-manifest-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--expected-parents", type=int, default=6292)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--version-tag", default="official-speed-v1")
    return parser.parse_args()


def normalize_id(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    if not text:
        return ""
    return text


def clean_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def leakage_key(value: Any) -> str:
    return normalize_prediction(clean_text(value))


def factor_label(factor: Fraction) -> str:
    return f"{float(factor):.1f}"


def manifest_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def resolve_audio_path(value: Any, project_root: Path) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def sha256_path(path: Path) -> str:
    return sha256_file(path)


def resample_to_rate(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    divisor = math.gcd(source_rate, target_rate)
    result = resample_poly(
        audio,
        target_rate // divisor,
        source_rate // divisor,
    )
    return np.asarray(result, dtype=np.float32)


def apply_speed(audio: np.ndarray, factor: Fraction) -> np.ndarray:
    inverse = Fraction(1, 1) / factor
    result = resample_poly(audio, inverse.numerator, inverse.denominator)
    return np.asarray(result, dtype=np.float32)


def read_mono_audio(path: Path) -> tuple[np.ndarray, int, float, int]:
    info = sf.info(str(path))
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError(f"invalid audio header: {path}")
    waveform, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if waveform.size == 0:
        raise ValueError(f"empty audio: {path}")
    mono = waveform.mean(axis=1, dtype=np.float32)
    duration = float(len(mono)) / int(sample_rate)
    return mono, int(sample_rate), duration, int(info.channels)


def write_derived_audio(
    source_audio: np.ndarray,
    source_rate: int,
    factor: Fraction,
    target: Path,
) -> tuple[float, int, int, str]:
    base_rate = resample_to_rate(source_audio, source_rate, TARGET_SAMPLE_RATE)
    perturbed = apply_speed(base_rate, factor)
    safe = np.clip(perturbed, -1.0, 1.0)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.part")
    sf.write(
        str(partial),
        safe,
        TARGET_SAMPLE_RATE,
        format="WAV",
        subtype="PCM_16",
    )
    partial.replace(target)
    info = sf.info(str(target))
    duration = float(info.duration)
    if int(info.samplerate) != TARGET_SAMPLE_RATE or int(info.channels) != 1:
        raise ValueError(
            f"derived audio is not 16 kHz mono: {target} "
            f"({info.samplerate} Hz, {info.channels} channels)"
        )
    if not math.isfinite(duration) or duration <= 0 or duration > MAX_DURATION_S:
        raise ValueError(f"derived duration out of range: {target} ({duration})")
    return duration, int(info.samplerate), int(info.channels), sha256_path(target)


def child_row(
    parent: dict[str, Any],
    *,
    parent_id: str,
    factor: Fraction,
    audio_path: str,
    duration_s: float,
    sample_rate: int,
    channels: int,
    parent_hash: str,
    audio_hash: str,
) -> dict[str, Any]:
    label = factor_label(factor)
    result = dict(parent)
    result.update(
        {
            "id": f"official:{parent_id}:speed:{label}",
            "audio_path": audio_path,
            "text": str(parent["text"]),
            "source": f"official_speed_{label}",
            "parent_id": parent_id,
            "speed_factor": float(factor),
            "augmentation": "none" if factor == 1 else "resample_poly_speed",
            "duration_s": round(duration_s, 6),
            "sample_rate": sample_rate,
            "channels": channels,
            "parent_audio_sha256": parent_hash,
            "audio_sha256": audio_hash,
            "split": "train",
        }
    )
    return result


def process_parent(
    parent: dict[str, Any],
    config: BuildConfig,
    temporary_audio_root: Path,
) -> ParentResult:
    parent_id = normalize_id(parent.get("id"))
    base = {
        "parent_id": parent_id,
        "source_audio_path": str(parent.get("audio_path", "")),
        "text": str(parent.get("text", "")),
    }
    try:
        if not parent_id:
            raise ValueError("empty parent id")
        if str(parent.get("source", "")) != "official":
            raise ValueError("parent source is not official")
        if str(parent.get("split", "")) != "train":
            raise ValueError("parent split is not train")
        if not str(parent.get("text", "")).strip():
            raise ValueError("empty parent text")
        source_path = resolve_audio_path(parent.get("audio_path"), config.project_root)
        if not source_path.is_file():
            raise FileNotFoundError(f"missing parent audio: {source_path}")
        source_audio, source_rate, source_duration, source_channels = read_mono_audio(
            source_path
        )
        if source_duration > MAX_DURATION_S:
            raise ValueError(f"parent exceeds {MAX_DURATION_S}s: {source_duration}")
        parent_hash = sha256_path(source_path)
        source_manifest_path = manifest_path(source_path, config.project_root)
        rows = [
            child_row(
                parent,
                parent_id=parent_id,
                factor=Fraction(1),
                audio_path=source_manifest_path,
                duration_s=source_duration,
                sample_rate=source_rate,
                channels=source_channels,
                parent_hash=parent_hash,
                audio_hash=parent_hash,
            )
        ]
        for factor in (Fraction("0.9"), Fraction("1.1")):
            label = factor_label(factor)
            target = temporary_audio_root / label / f"{parent_id}.wav"
            final_target = config.output_audio_root / label / f"{parent_id}.wav"
            duration, sample_rate, channels, audio_hash = write_derived_audio(
                source_audio,
                source_rate,
                factor,
                target,
            )
            expected_duration = source_duration / float(factor)
            if abs(duration - expected_duration) > DURATION_TOLERANCE_S:
                raise ValueError(
                    f"duration mismatch for {parent_id} factor {label}: "
                    f"{duration:.6f} vs expected {expected_duration:.6f}"
                )
            rows.append(
                child_row(
                    parent,
                    parent_id=parent_id,
                    factor=factor,
                    audio_path=manifest_path(final_target, config.project_root),
                    duration_s=duration,
                    sample_rate=sample_rate,
                    channels=channels,
                    parent_hash=parent_hash,
                    audio_hash=audio_hash,
                )
            )
        return ParentResult(parent_id=parent_id, rows=rows)
    except Exception as exc:
        return ParentResult(
            parent_id=parent_id,
            rows=[],
            error={**base, "reason": "augmentation_failed", "error": str(exc)},
        )


def normalized_text_set(rows: list[dict[str, Any]]) -> set[str]:
    return {key for row in rows if (key := leakage_key(row.get("text", "")))}


def validate_source_rows(rows: list[dict[str, Any]], expected_parents: int) -> None:
    if len(rows) != expected_parents:
        raise ValueError(
            f"expected {expected_parents} source rows, found {len(rows)}"
        )
    ids = [normalize_id(row.get("id")) for row in rows]
    if not all(ids):
        raise ValueError("source manifest contains an empty id")
    if len(ids) != len(set(ids)):
        raise ValueError("source manifest contains duplicate ids")
    unexpected_sources = sorted({str(row.get("source")) for row in rows} - {"official"})
    if unexpected_sources:
        raise ValueError(f"source manifest has non-official rows: {unexpected_sources}")
    unexpected_splits = sorted({str(row.get("split")) for row in rows} - {"train"})
    if unexpected_splits:
        raise ValueError(f"source manifest has non-train rows: {unexpected_splits}")


def validate_children(
    *,
    parents: list[dict[str, Any]],
    children: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    public: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_rows = len(parents) * len(FACTORS)
    if len(children) != expected_rows:
        raise ValueError(f"expected {expected_rows} output rows, found {len(children)}")
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for row in children:
        by_parent.setdefault(str(row["parent_id"]), []).append(row)
    expected_parent_ids = {normalize_id(row["id"]) for row in parents}
    if set(by_parent) != expected_parent_ids:
        raise ValueError("output parent IDs do not exactly match source parent IDs")
    factor_counts = {factor_label(factor): 0 for factor in FACTORS}
    derived_hashes: dict[str, set[str]] = {"0.9": set(), "1.1": set()}
    child_ids = set()
    for parent_id, rows in by_parent.items():
        factors = {factor_label(Fraction(str(row["speed_factor"]))) for row in rows}
        if factors != set(factor_counts):
            raise ValueError(f"parent {parent_id} does not have all three factors")
        texts = {str(row["text"]) for row in rows}
        if len(texts) != 1:
            raise ValueError(f"parent {parent_id} changed text across factors")
        for row in rows:
            row_id = str(row["id"])
            if row_id in child_ids:
                raise ValueError(f"duplicate child id: {row_id}")
            child_ids.add(row_id)
            label = factor_label(Fraction(str(row["speed_factor"])))
            factor_counts[label] += 1
            if label in derived_hashes:
                audio_hash = str(row["audio_sha256"])
                if audio_hash in derived_hashes[label]:
                    raise ValueError(f"duplicate derived audio hash at factor {label}")
                derived_hashes[label].add(audio_hash)
    if set(factor_counts.values()) != {len(parents)}:
        raise ValueError(f"unexpected factor counts: {factor_counts}")

    validation_ids = {str(row.get("id")) for row in validation}
    public_ids = {str(row.get("id")) for row in public}
    if child_ids & (validation_ids | public_ids):
        raise ValueError("augmented child IDs overlap validation/public IDs")
    child_audio = {str(row["audio_path"]) for row in children}
    validation_audio = {str(row.get("audio_path", "")) for row in validation}
    public_audio = {str(row.get("audio_path", "")) for row in public}
    if child_audio & (validation_audio | public_audio):
        raise ValueError("augmented audio paths overlap validation/public audio")

    source_keys = normalized_text_set(parents)
    validation_keys = normalized_text_set(validation)
    public_keys = normalized_text_set(public)
    child_keys = normalized_text_set(children)
    baseline = {
        "validation": sorted(source_keys & validation_keys),
        "public": sorted(source_keys & public_keys),
    }
    after = {
        "validation": sorted(child_keys & validation_keys),
        "public": sorted(child_keys & public_keys),
    }
    if baseline != after:
        raise ValueError(
            "augmentation changed normalized-text overlap against validation/public"
        )
    total_seconds = sum(float(row["duration_s"]) for row in children)
    return {
        "factor_counts": factor_counts,
        "total_seconds": total_seconds,
        "total_hours": total_seconds / 3600,
        "normalized_text_overlap": {
            "baseline": baseline,
            "augmented": after,
        },
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build(config: BuildConfig) -> dict[str, Any]:
    for path in (
        config.train_manifest,
        config.validation_manifest,
        config.public_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"manifest not found: {path}")
    if config.expected_parents <= 0:
        raise ValueError("expected parent count must be positive")
    if config.workers <= 0:
        raise ValueError("workers must be positive")
    if config.output_audio_root.exists() or config.output_manifest_dir.exists():
        raise FileExistsError(
            "refusing to overwrite existing augmentation output: "
            f"{config.output_audio_root} or {config.output_manifest_dir}"
        )
    config.output_audio_root.parent.mkdir(parents=True, exist_ok=True)
    config.output_manifest_dir.parent.mkdir(parents=True, exist_ok=True)

    parents = read_jsonl(config.train_manifest)
    validation = read_jsonl(config.validation_manifest)
    public = read_jsonl(config.public_manifest)
    validate_source_rows(parents, config.expected_parents)
    if not validation or not public:
        raise ValueError("validation and public manifests must be non-empty")

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{config.version_tag}-{uuid.uuid4().hex[:8]}-",
            dir=str(config.output_manifest_dir.parent),
        )
    )
    temporary_audio = temporary_root / "audio"
    temporary_manifest = temporary_root / "manifest"
    temporary_manifest.mkdir(parents=True)
    results: list[ParentResult] = []
    try:
        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            futures = [
                executor.submit(process_parent, parent, config, temporary_audio)
                for parent in parents
            ]
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: int(item.parent_id))
        quarantine = [result.error for result in results if result.error is not None]
        if quarantine:
            write_jsonl(temporary_manifest / "quarantine.jsonl", quarantine)
            write_json(
                temporary_manifest / "failure_report.json",
                {
                    "parents_expected": config.expected_parents,
                    "parents_failed": len(quarantine),
                    "quarantine_path": str(temporary_manifest / "quarantine.jsonl"),
                },
            )
            raise RuntimeError(
                f"{len(quarantine)} parent(s) failed augmentation; diagnostics: "
                f"{temporary_manifest}"
            )

        children = [
            row
            for result in results
            for row in sorted(result.rows, key=lambda row: float(row["speed_factor"]))
        ]
        checks = validate_children(
            parents=parents,
            children=children,
            validation=validation,
            public=public,
        )
        write_jsonl(temporary_manifest / "all.jsonl", children)
        write_jsonl(temporary_manifest / "quarantine.jsonl", [])
        report = {
            "version_tag": config.version_tag,
            "inputs": {
                "train_manifest": str(config.train_manifest.resolve()),
                "train_manifest_sha256": sha256_path(config.train_manifest),
                "validation_manifest": str(config.validation_manifest.resolve()),
                "validation_manifest_sha256": sha256_path(config.validation_manifest),
                "public_manifest": str(config.public_manifest.resolve()),
                "public_manifest_sha256": sha256_path(config.public_manifest),
            },
            "environment": {
                "python": sys.version.split()[0],
                "scipy": scipy.__version__,
                "soundfile": sf.__version__,
                "target_sample_rate": TARGET_SAMPLE_RATE,
                "max_duration_s": MAX_DURATION_S,
                "duration_tolerance_s": DURATION_TOLERANCE_S,
                "workers": config.workers,
            },
            "counts": {
                "parents": len(parents),
                "rows": len(children),
                **checks["factor_counts"],
            },
            "duration": {
                "total_seconds": round(float(checks["total_seconds"]), 6),
                "total_hours": round(float(checks["total_hours"]), 6),
            },
            "checks": {
                "normalized_text_overlap": checks["normalized_text_overlap"],
                "quarantine_rows": 0,
            },
        }
        write_json(temporary_manifest / "data_report.json", report)
        write_json(
            temporary_manifest / "receipt.json",
            {
                "version_tag": config.version_tag,
                "manifest_sha256": sha256_path(temporary_manifest / "all.jsonl"),
                "data_report_sha256": sha256_path(temporary_manifest / "data_report.json"),
                "rows": len(children),
            },
        )

        shutil.move(str(temporary_audio), str(config.output_audio_root))
        try:
            shutil.move(str(temporary_manifest), str(config.output_manifest_dir))
        except Exception:
            shutil.rmtree(config.output_audio_root, ignore_errors=True)
            raise
        shutil.rmtree(temporary_root, ignore_errors=True)
        return report
    except Exception:
        print(f"augmentation workspace retained for inspection: {temporary_root}", file=sys.stderr)
        raise


def main() -> None:
    args = parse_args()
    config = BuildConfig(
        train_manifest=args.train_manifest.resolve(),
        validation_manifest=args.validation_manifest.resolve(),
        public_manifest=args.public_manifest.resolve(),
        output_audio_root=args.output_audio_root.resolve(),
        output_manifest_dir=args.output_manifest_dir.resolve(),
        project_root=args.project_root.resolve(),
        expected_parents=int(args.expected_parents),
        workers=int(args.workers),
        version_tag=str(args.version_tag),
    )
    report = build(config)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
