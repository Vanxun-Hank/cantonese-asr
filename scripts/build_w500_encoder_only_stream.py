#!/usr/bin/env python3
"""Freeze one leakage-audited 35h Wenet v1 stream and nested prefixes.

The stream is selected without replacement from the existing filtered 500h
manifest.  Selection is deterministic, independent of training seed, and
aligned to 32 rows so both 1:1 and 2:1 source schedules checkpoint only after a
complete source cycle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import statistics
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import soundfile as sf

from cantonese_asr.io import read_jsonl
from cantonese_asr.metrics import normalize_reference


PRIMARY_TARGETS = (25, 30, 35)
ALL_TARGETS = (5, 10, 15, 20, *PRIMARY_TARGETS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--public-audio-dir",
        type=Path,
        default=Path("artifacts/data/train_raw/4856-10393"),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def row_text(row: dict[str, Any]) -> str:
    return str(
        row.get("text")
        or row.get("metric_text")
        or row.get("ref_text")
        or row.get("text_raw")
        or ""
    )


def row_duration(row: dict[str, Any]) -> float:
    return float(row.get("duration_s") or row.get("duration") or 0.0)


def row_audio(row: dict[str, Any], root: Path, public_dir: Path | None = None) -> Path:
    raw = str(row.get("audio_path") or row.get("audio") or "").strip()
    path = Path(raw)
    if public_dir is not None:
        candidate = public_dir / path.name
        if candidate.is_file():
            return candidate
    if not path.is_absolute():
        path = root / path
    return path


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "sum": sum(values),
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def validate_manifest(path: Path, expected: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    digest = sha256_file(path)
    if len(rows) != int(expected["rows"]):
        raise SystemExit(f"Row mismatch for {path}: {len(rows)} != {expected['rows']}")
    if expected.get("sha256") and digest != expected["sha256"]:
        raise SystemExit(f"SHA mismatch for {path}: {digest} != {expected['sha256']}")
    return rows


def group_key(row: dict[str, Any]) -> str:
    return str(
        row.get("program_group")
        or row.get("program")
        or row.get("speaker_id")
        or row.get("source_tar")
        or "unknown"
    )


def deterministic_round_robin(
    rows: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    """Seeded within-group shuffle followed by fair round-robin interleaving."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[group_key(row)].append(row)
    rng = random.Random(seed)
    keys = sorted(buckets)
    rng.shuffle(keys)
    queues: dict[str, deque[dict[str, Any]]] = {}
    for key in keys:
        bucket = buckets[key]
        rng.shuffle(bucket)
        queues[key] = deque(bucket)
    result: list[dict[str, Any]] = []
    active = keys
    while active:
        next_active: list[str] = []
        for key in active:
            queue = queues[key]
            if queue:
                result.append(queue.popleft())
            if queue:
                next_active.append(key)
        active = next_active
    return result


def collect_protected(
    project_root: Path,
    public_dir: Path,
    manifests: dict[str, list[dict[str, Any]]],
) -> tuple[set[str], set[str], set[str], dict[str, Any]]:
    ids: set[str] = set()
    names: set[str] = set()
    texts: set[str] = set()
    audio_hashes: set[str] = set()
    unresolved: list[str] = []
    hashed = 0
    for label, rows in manifests.items():
        for row in rows:
            identifier = str(row.get("id") or "").strip()
            if identifier:
                ids.add(identifier)
            text = normalize_reference(row_text(row))
            if text:
                texts.add(text)
            path = row_audio(
                row,
                project_root,
                public_dir if label == "public" else None,
            )
            if path.name:
                names.add(path.name)
            supplied = str(row.get("audio_sha256") or "").strip()
            if supplied:
                audio_hashes.add(supplied)
            elif path.is_file():
                audio_hashes.add(sha256_file(path))
                hashed += 1
            else:
                unresolved.append(str(path))
    return ids, names, texts, {
        "audio_sha256": audio_hashes,
        "audio_files_hashed": hashed,
        "unresolved_audio_count": len(unresolved),
        "unresolved_audio_examples": unresolved[:20],
    }


def audio_check_and_hash(path: Path, declared_duration: float) -> tuple[str, dict[str, Any]]:
    info = sf.info(path)
    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise ValueError(f"Invalid decoded audio metadata: {path}: {info}")
    decoded_duration = info.frames / info.samplerate
    if abs(decoded_duration - declared_duration) > 0.10:
        raise ValueError(
            f"Duration mismatch for {path}: decoded={decoded_duration} declared={declared_duration}"
        )
    return sha256_file(path), {
        "decoded_duration_s": decoded_duration,
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
    }


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    config_path = (root / args.config).resolve() if not args.config.is_absolute() else args.config
    output = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    public_dir = (
        (root / args.public_audio_dir).resolve()
        if not args.public_audio_dir.is_absolute()
        else args.public_audio_dir
    )
    if output.exists():
        raise SystemExit(f"Refusing to overwrite frozen output: {output}")
    staging = output.with_name(output.name + f".staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    manifests_dir = staging / "manifests"
    manifests_dir.mkdir()

    config = json.loads(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    official = validate_manifest(root / data["official_train"]["path"], data["official_train"])
    validation = validate_manifest(root / data["fixed_validation"]["path"], data["fixed_validation"])
    public = validate_manifest(
        root / data["public"]["path"],
        {"rows": data["public"]["rows"]},
    )
    ood = validate_manifest(root / data["ood"]["path"], {"rows": data["ood"]["rows"]})
    pool = validate_manifest(root / data["wenet_500h_pool"]["path"], data["wenet_500h_pool"])
    champion = validate_manifest(
        root / data["champion_wenet_exposure"]["path"],
        data["champion_wenet_exposure"],
    )
    champion_id_digest = sha256_lines([str(row["id"]) for row in champion])
    if champion_id_digest != data["champion_wenet_exposure"]["ordered_wenet_id_sha256"]:
        raise SystemExit(
            "Champion 2,240-row order changed: "
            f"{champion_id_digest} != {data['champion_wenet_exposure']['ordered_wenet_id_sha256']}"
        )

    protected_ids, protected_names, protected_texts, protected_audio = collect_protected(
        root,
        public_dir,
        {
            "champion_wenet": champion,
            "official": official,
            "validation": validation,
            "public": public,
            "ood": ood,
        },
    )
    protected_hashes: set[str] = protected_audio.pop("audio_sha256")

    rejection = Counter()
    metadata_eligible: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    seen_texts: set[str] = set()
    for row in pool:
        identifier = str(row.get("id") or "").strip()
        path = row_audio(row, root)
        normalized = normalize_reference(row_text(row))
        duration = row_duration(row)
        reason = None
        if not identifier or identifier in seen_ids:
            reason = "missing_or_duplicate_id"
        elif not normalized or normalized in seen_texts:
            reason = "empty_or_duplicate_text"
        elif not path.is_file() or str(path) in seen_paths:
            reason = "missing_or_duplicate_path"
        elif duration <= 0:
            reason = "invalid_duration"
        elif identifier in protected_ids:
            reason = "protected_id"
        elif path.name in protected_names:
            reason = "protected_audio_name"
        elif normalized in protected_texts:
            reason = "protected_text"
        elif row.get("overlap_flags"):
            reason = "upstream_overlap_flag"
        if reason:
            rejection[reason] += 1
            continue
        seen_ids.add(identifier)
        seen_paths.add(str(path))
        seen_texts.add(normalized)
        metadata_eligible.append(row)

    ordered = deterministic_round_robin(
        metadata_eligible,
        int(data["frozen_stream"]["seed"]),
    )
    alignment = int(data["frozen_stream"]["batch_alignment_rows"])
    if alignment % int(config["training"]["global_effective_batch_size"]):
        raise SystemExit("Stream alignment must be a multiple of the effective batch")

    selected: list[dict[str, Any]] = []
    selected_hashes: set[str] = set()
    fingerprints: list[dict[str, Any]] = []
    cumulative = 0.0
    boundaries: dict[int, int] = {}
    for row in ordered:
        path = row_audio(row, root)
        try:
            audio_sha, decoded = audio_check_and_hash(path, row_duration(row))
        except Exception as error:
            rejection["decode_or_duration_failure"] += 1
            fingerprints.append(
                {"id": row.get("id"), "audio_path": str(path), "error": repr(error)}
            )
            continue
        if audio_sha in protected_hashes:
            rejection["protected_audio_sha256"] += 1
            continue
        if audio_sha in selected_hashes:
            rejection["duplicate_audio_sha256"] += 1
            continue
        selected.append(row)
        selected_hashes.add(audio_sha)
        cumulative += row_duration(row)
        fingerprints.append(
            {
                "id": str(row["id"]),
                "audio_path": str(path),
                "audio_sha256": audio_sha,
                **decoded,
            }
        )
        for hours in ALL_TARGETS:
            if (
                hours not in boundaries
                and cumulative >= hours * 3600
                and len(selected) % alignment == 0
            ):
                boundaries[hours] = len(selected)
        if 35 in boundaries:
            break
    if set(boundaries) != set(ALL_TARGETS):
        raise SystemExit(f"Could not fill all duration targets: {boundaries}")

    boundary_report: dict[str, Any] = {}
    manifest_paths: dict[int, Path] = {}
    for hours in PRIMARY_TARGETS:
        prefix = selected[: boundaries[hours]]
        path = manifests_dir / f"wenet_new_{hours}h.jsonl"
        write_jsonl(path, prefix)
        manifest_paths[hours] = path
        seconds = sum(row_duration(row) for row in prefix)
        boundary_report[str(hours)] = {
            "path": str((output / "manifests" / path.name).resolve()),
            "rows": len(prefix),
            "wenet_optimizer_steps": len(prefix) // 16,
            "target_seconds": hours * 3600,
            "actual_seconds": seconds,
            "overshoot_seconds": seconds - hours * 3600,
            "hours": seconds / 3600,
            "average_duration_seconds": seconds / len(prefix),
            "sha256": sha256_file(path),
            "ordered_id_sha256": sha256_lines([str(row["id"]) for row in prefix]),
        }
    if read_jsonl(manifest_paths[30])[: boundaries[25]] != read_jsonl(manifest_paths[25]):
        raise SystemExit("30h is not an exact 25h prefix extension")
    if read_jsonl(manifest_paths[35])[: boundaries[30]] != read_jsonl(manifest_paths[30]):
        raise SystemExit("35h is not an exact 30h prefix extension")

    selected35 = selected[: boundaries[35]]
    selected35_ids = {str(row["id"]) for row in selected35}
    stats = {
        "duration_seconds": describe([row_duration(row) for row in selected35]),
        "text_characters": describe(
            [float(len(normalize_reference(row_text(row)))) for row in selected35]
        ),
        "program_groups": Counter(group_key(row) for row in selected35).most_common(),
        "speakers": Counter(str(row.get("speaker_id") or "unknown") for row in selected35).most_common(),
        "source_tars": Counter(str(row.get("source_tar") or "unknown") for row in selected35),
    }
    fingerprints_path = staging / "selected_audio_fingerprints.jsonl"
    write_jsonl(
        fingerprints_path,
        [row for row in fingerprints if row.get("id") in selected35_ids],
    )
    report = {
        "passed": True,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "seed": data["frozen_stream"]["seed"],
        "alignment_rows": alignment,
        "source_pool": {
            "path": str((root / data["wenet_500h_pool"]["path"]).resolve()),
            "rows": len(pool),
            "sha256": data["wenet_500h_pool"]["sha256"],
        },
        "champion_exposure": {
            "path": str((root / data["champion_wenet_exposure"]["path"]).resolve()),
            "rows": len(champion),
            "sha256": data["champion_wenet_exposure"]["sha256"],
            "ordered_id_sha256": champion_id_digest,
        },
        "protected": {
            "id_count": len(protected_ids),
            "audio_name_count": len(protected_names),
            "normalized_text_count": len(protected_texts),
            "audio_sha256_count": len(protected_hashes),
            **protected_audio,
        },
        "metadata_eligible_rows": len(metadata_eligible),
        "rejections": dict(rejection),
        "boundaries": boundary_report,
        "safety_boundary_rows": {str(k): boundaries[k] for k in (5, 10, 15, 20)},
        "selected_35h": {
            "unique_ids": len(selected35_ids),
            "unique_paths": len({str(row_audio(row, root)) for row in selected35}),
            "unique_audio_sha256": len(selected_hashes),
            "champion_id_overlap": len({str(row["id"]) for row in selected35} & {str(row["id"]) for row in champion}),
            "champion_text_overlap": len(
                {normalize_reference(row_text(row)) for row in selected35}
                & {normalize_reference(row_text(row)) for row in champion}
            ),
            "protected_audio_sha256_overlap": len(selected_hashes & protected_hashes),
            "stats": stats,
        },
        "fingerprints": {
            "path": str((output / fingerprints_path.name).resolve()),
            "sha256": sha256_file(fingerprints_path),
        },
    }
    (staging / "stream_build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (staging / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  manifests/{path.name}\n"
            for path in manifest_paths.values()
        )
        + f"{sha256_file(fingerprints_path)}  {fingerprints_path.name}\n",
        encoding="utf-8",
    )
    (staging / "MANIFESTS_FROZEN").write_text("PASS\n", encoding="utf-8")
    staging.rename(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
