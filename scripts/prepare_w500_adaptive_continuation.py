#!/usr/bin/env python3
"""Audit and freeze distribution-preserving W500 continuation manifests."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from cantonese_asr.io import read_jsonl
from cantonese_asr.metrics import normalize_reference
from cantonese_asr.w500_adaptive import (
    assert_exact_prefix,
    bucket_key,
    middle_loss_rows,
    nearest_aligned_boundary,
)
from scripts.build_w500_encoder_only_stream import (
    audio_check_and_hash,
    collect_protected,
    row_audio,
    row_duration,
    row_text,
    sha256_file,
    sha256_lines,
)


_CHAR_REPEAT = re.compile(r"(.)\1{7,}")


def abnormal_unicode(text: str) -> bool:
    if "\ufffd" in text:
        return True
    return any(
        unicodedata.category(char) in {"Cc", "Cs", "Co"}
        and char not in "\t\n\r"
        for char in text
    )


def repeated_phrase(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if _CHAR_REPEAT.search(compact):
        return True
    for width in range(2, 5):
        for start in range(max(0, len(compact) - 120)):
            unit = compact[start : start + width]
            if unit and unit * 4 in compact[start:]:
                return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("audit", "freeze"), required=True)
    parser.add_argument("--candidate-losses", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    _write_json(temporary, value)
    os.replace(temporary, path)


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _protected_manifests(root: Path, cfg: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    data = cfg["data"]
    paths = {
        "consumed_35h": data["consumed_35h"],
        "official": data["official"],
        "validation": data["validation"],
        "public": data["public"],
        "ood": data["ood"],
    }
    if data.get("champion_w20"):
        paths["champion_w20"] = data["champion_w20"]
    return {label: read_jsonl(_resolve(root, path)) for label, path in paths.items()}


def audit_candidates(
    root: Path, cfg: Mapping[str, Any], output: Path
) -> dict[str, Any]:
    """Audit the pool and atomically publish the exact loss-scoring manifest."""

    if output.exists():
        raise SystemExit(f"Refusing to overwrite adaptive audit: {output}")
    staging = output.with_name(output.name + f".staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    data = cfg["data"]
    pool_path = _resolve(root, data["w500_pool"])
    pool = read_jsonl(pool_path)
    protected_manifests = _protected_manifests(root, cfg)
    public_dir = root / "artifacts/data/train_raw/4856-10393"
    protected_ids, protected_names, protected_texts, protected_audio = collect_protected(
        root, public_dir, protected_manifests
    )
    protected_hashes: set[str] = protected_audio.pop("audio_sha256")

    eligible: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    seen_texts: set[str] = set()
    seen_hashes: set[str] = set()
    for source_row in pool:
        row = dict(source_row)
        identifier = str(row.get("id") or "").strip()
        text = row_text(row).strip()
        normalized = normalize_reference(text)
        audio = row_audio(row, root)
        duration = row_duration(row)
        reason: str | None = None
        decoded: dict[str, Any] = {}
        audio_sha = ""
        resolved_audio = str(audio.resolve())
        if not identifier:
            reason = "missing_id"
        elif identifier in seen_ids:
            reason = "duplicate_id"
        elif not text or not normalized:
            reason = "empty_text"
        elif abnormal_unicode(text):
            reason = "abnormal_unicode"
        elif repeated_phrase(text):
            reason = "obvious_repetition"
        elif not math.isfinite(duration) or duration <= 0:
            reason = "invalid_duration"
        elif not audio.is_file():
            reason = "missing_audio"
        elif resolved_audio in seen_paths:
            reason = "duplicate_audio_path"
        elif normalized in seen_texts:
            reason = "duplicate_normalized_text"
        elif identifier in protected_ids:
            reason = "protected_id"
        elif audio.name in protected_names:
            reason = "protected_audio_name"
        elif normalized in protected_texts:
            reason = "protected_normalized_text"
        elif row.get("overlap_flags") not in (None, "", [], {}):
            reason = "upstream_overlap_flag"
        if reason is None:
            try:
                audio_sha, decoded = audio_check_and_hash(audio, duration)
            except Exception as error:  # the audit receipt preserves exact evidence
                reason = "decode_or_duration_failure"
                decoded = {"error": repr(error)}
        if reason is None and audio_sha in protected_hashes:
            reason = "protected_audio_sha256"
        if reason is None and audio_sha in seen_hashes:
            reason = "duplicate_audio_sha256"
        if reason is not None:
            rejections[reason] += 1
            audit_rows.append(
                {
                    "id": identifier or None,
                    "status": "rejected",
                    "reason": reason,
                    "audio_path": str(audio),
                    **decoded,
                }
            )
            continue

        seen_ids.add(identifier)
        seen_paths.add(resolved_audio)
        seen_texts.add(normalized)
        seen_hashes.add(audio_sha)
        row["metric_text"] = normalized
        row["audio_sha256"] = audio_sha
        row["adaptive_audio_check"] = decoded
        eligible.append(row)
        audit_rows.append(
            {
                "id": identifier,
                "status": "eligible",
                "reason": None,
                "audio_path": str(audio),
                "audio_sha256": audio_sha,
                **decoded,
            }
        )

    eligible_path = staging / "eligible_candidates.jsonl"
    audit_path = staging / "candidate_audit.jsonl"
    _write_jsonl(eligible_path, eligible)
    _write_jsonl(audit_path, audit_rows)
    eligible_ids = {str(row["id"]) for row in eligible}
    eligible_texts = {str(row["metric_text"]) for row in eligible}
    report = {
        "passed": bool(eligible),
        "pool": {
            "path": str(pool_path.resolve()),
            "rows": len(pool),
            "sha256": sha256_file(pool_path),
        },
        "eligible": {
            "path": str((output / eligible_path.name).resolve()),
            "rows": len(eligible),
            "sha256": sha256_file(eligible_path),
            "ordered_id_sha256": sha256_lines([str(row["id"]) for row in eligible]),
        },
        "candidate_audit": {
            "path": str((output / audit_path.name).resolve()),
            "rows": len(audit_rows),
            "sha256": sha256_file(audit_path),
        },
        "rejections": dict(sorted(rejections.items())),
        "protected": {
            "id_count": len(protected_ids),
            "audio_name_count": len(protected_names),
            "normalized_text_count": len(protected_texts),
            "audio_sha256_count": len(protected_hashes),
            **protected_audio,
        },
        "overlaps": {
            "protected": len(eligible_ids & protected_ids)
            + len(eligible_texts & protected_texts)
            + len(seen_hashes & protected_hashes),
            "duplicate_ids": len(eligible) - len(eligible_ids),
            "duplicate_texts": len(eligible) - len(eligible_texts),
            "duplicate_audio_hashes": len(eligible) - len(seen_hashes),
        },
    }
    if not report["passed"] or any(report["overlaps"].values()):
        report["passed"] = False
    _write_json(staging / "candidate_audit_summary.json", report)
    staging.rename(output)
    return report


def _stable_digest(seed: int, namespace: str, identifier: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{identifier}".encode("utf-8")).hexdigest()


def _metadata_quality(row: Mapping[str, Any]) -> float:
    values: list[float] = []
    for name in ("quality_score", "confidence", "jyutping_confidence", "dnsmos", "snr"):
        value = row.get(name)
        if value in (None, ""):
            continue
        number = float(value)
        if math.isfinite(number):
            values.append(number)
    return sum(values) / len(values) if values else 0.0


def _bucket_cell(key: tuple[str, str, str, str, str]) -> tuple[int, int]:
    duration_labels = (
        "dur_0_3", "dur_3_5", "dur_5_8", "dur_8_12", "dur_12_20",
        "dur_20_30", "dur_30_plus",
    )
    text_labels = (
        "txt_1_11", "txt_11_16", "txt_16_21", "txt_21_31", "txt_31_46",
        "txt_46_80", "txt_80_plus",
    )
    try:
        return duration_labels.index(key[0]), text_labels.index(key[1])
    except ValueError as error:
        raise ValueError(f"Unsupported adaptive bucket: {key[:2]}") from error


def balanced_candidate_order(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    seed: int,
    initial_band: tuple[float, float],
    relaxed_band: tuple[float, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Order candidates by frozen distribution and documented loss relaxations."""

    reference_counts: Counter[tuple[str, str, str, str, str]] = Counter(
        bucket_key(row) for row in reference_rows
    )
    if not reference_counts:
        raise ValueError("The consumed-35h reference stream is empty")
    by_actual: defaultdict[
        tuple[str, str, str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in candidate_rows:
        by_actual[bucket_key(row)].append(row)

    initial_ids: set[str] = set()
    relaxed_ids: set[str] = set()
    for rows in by_actual.values():
        initial_ids.update(
            str(row["id"])
            for row in middle_loss_rows(rows, low=initial_band[0], high=initial_band[1])
        )
        relaxed_ids.update(
            str(row["id"])
            for row in middle_loss_rows(rows, low=relaxed_band[0], high=relaxed_band[1])
        )

    references_by_cell: defaultdict[
        tuple[int, int], list[tuple[str, str, str, str, str]]
    ] = defaultdict(list)
    for key in reference_counts:
        references_by_cell[_bucket_cell(key)].append(key)
    reference_cells = sorted(references_by_cell)

    assigned: defaultdict[
        tuple[str, str, str, str, str], list[tuple[int, str, float, dict[str, Any], str]]
    ] = defaultdict(list)
    for row in candidate_rows:
        identifier = str(row["id"])
        actual = bucket_key(row)
        if actual in reference_counts:
            target = actual
            if identifier in initial_ids:
                stage_index, stage = 0, "initial_10_90"
            elif identifier in relaxed_ids:
                stage_index, stage = 1, "relaxed_5_95"
            else:
                stage_index, stage = 2, "same_bucket_all"
        else:
            actual_cell = _bucket_cell(actual)
            nearest_cell = min(
                reference_cells,
                key=lambda cell: (
                    abs(cell[0] - actual_cell[0]) + abs(cell[1] - actual_cell[1]),
                    cell,
                ),
            )
            possible = references_by_cell[nearest_cell]
            target = min(
                possible,
                key=lambda key: (
                    sum(left != right for left, right in zip(actual[2:], key[2:])),
                    -reference_counts[key],
                    key,
                ),
            )
            stage_index, stage = 3, "nearest_duration_text_bucket"
        assigned[target].append(
            (
                stage_index,
                _stable_digest(seed, "balanced-selection", identifier),
                -_metadata_quality(row),
                row,
                stage,
            )
        )

    queues: dict[
        tuple[str, str, str, str, str], list[tuple[int, str, float, dict[str, Any], str]]
    ] = {}
    for key, items in assigned.items():
        queues[key] = sorted(items, key=lambda item: (item[0], item[1], item[2], str(item[3]["id"])))

    positions: Counter[tuple[str, str, str, str, str]] = Counter()
    heap: list[tuple[float, tuple[str, str, str, str, str]]] = []
    for key, queue in queues.items():
        if queue:
            heapq.heappush(heap, (1.0 / reference_counts[key], key))
    ordered: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    while heap:
        _, key = heapq.heappop(heap)
        item = queues[key][positions[key]]
        _, _, _, row, stage = item
        ordered.append(row)
        selections.append(
            {
                "id": str(row["id"]),
                "stage": stage,
                "actual_bucket": list(bucket_key(row)),
                "target_bucket": list(key),
            }
        )
        positions[key] += 1
        if positions[key] < len(queues[key]):
            heapq.heappush(
                heap,
                ((positions[key] + 1.0) / reference_counts[key], key),
            )
    if len(ordered) != len(candidate_rows):
        raise RuntimeError("Balanced order lost candidate rows")
    return ordered, selections


def _read_loss_sidecar(path: Path, eligible_sha256: str) -> dict[str, float]:
    rows = read_jsonl(path)
    receipts: set[str] = set()
    loss_rows: list[dict[str, Any]] = []
    for row in rows:
        receipt_sha = row.get("eligible_manifest_sha256") or row.get("manifest_sha256")
        if receipt_sha:
            receipts.add(str(receipt_sha))
        if row.get("id") is not None:
            loss_rows.append(row)
    companion_candidates = (
        path.with_suffix(path.suffix + ".receipt.json"),
        path.with_suffix(".receipt.json"),
        path.parent / "loss_receipt.json",
    )
    for companion in companion_candidates:
        if companion.is_file():
            receipt = json.loads(companion.read_text(encoding="utf-8"))
            receipt_sha = receipt.get("eligible_manifest_sha256") or receipt.get("manifest_sha256")
            if receipt_sha:
                receipts.add(str(receipt_sha))
    if receipts != {eligible_sha256}:
        raise SystemExit(
            "Candidate-loss receipt does not identify the exact eligible manifest: "
            f"expected={eligible_sha256} observed={sorted(receipts)}"
        )
    losses: dict[str, float] = {}
    for row in loss_rows:
        identifier = str(row["id"])
        if identifier in losses:
            raise SystemExit(f"Duplicate candidate loss for {identifier}")
        try:
            value = float(row["normalized_loss"])
        except (KeyError, TypeError, ValueError) as error:
            raise SystemExit(f"Invalid candidate loss for {identifier}") from error
        if not math.isfinite(value):
            raise SystemExit(f"Non-finite candidate loss for {identifier}")
        losses[identifier] = value
    return losses


def _manifest_receipt(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    target_hours: float,
    target_seconds: float,
    prefix_of_50h: bool,
) -> dict[str, Any]:
    actual_seconds = sum(row_duration(row) for row in rows)
    return {
        "path": str(path.resolve()),
        "target_hours": target_hours,
        "target_continuation_seconds": target_seconds,
        "rows": len(rows),
        "wenet_optimizer_steps": len(rows) // 16,
        "actual_seconds": actual_seconds,
        "actual_continuation_hours": actual_seconds / 3600.0,
        "deviation_seconds": actual_seconds - target_seconds,
        "ordered_id_sha256": sha256_lines([str(row["id"]) for row in rows]),
        "sha256": sha256_file(path),
        "prefix_of_50h": prefix_of_50h,
    }


def _historical_delta(
    consumed: list[dict[str, Any]], historical: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    assert_exact_prefix(consumed, historical)
    delta = historical[len(consumed) :]
    if not delta:
        raise SystemExit("Historical 40h manifest has no post-35h continuation")
    if {str(row["id"]) for row in consumed} & {str(row["id"]) for row in delta}:
        raise SystemExit("Historical continuation overlaps consumed-35h IDs")
    return delta


def freeze_manifests(
    root: Path,
    cfg: Mapping[str, Any],
    output: Path,
    candidate_losses: Path,
) -> dict[str, Any]:
    """Freeze balanced nested prefixes after validating the exact loss sidecar."""

    marker = output / "MANIFESTS_FROZEN"
    if marker.exists():
        raise SystemExit(f"Refusing to overwrite frozen manifests: {output}")
    summary_path = output / "candidate_audit_summary.json"
    eligible_path = output / "eligible_candidates.jsonl"
    if not summary_path.is_file() or not eligible_path.is_file():
        raise SystemExit("Audit mode must pass before freeze mode")
    audit = json.loads(summary_path.read_text(encoding="utf-8"))
    eligible_sha = sha256_file(eligible_path)
    if not audit.get("passed") or audit["eligible"]["sha256"] != eligible_sha:
        raise SystemExit("Eligible-candidate audit receipt is missing, failed, or stale")

    eligible = read_jsonl(eligible_path)
    losses = _read_loss_sidecar(candidate_losses, eligible_sha)
    eligible_ids = [str(row["id"]) for row in eligible]
    missing = sorted(set(eligible_ids) - set(losses))
    extra = sorted(set(losses) - set(eligible_ids))
    if missing or extra or len(losses) != len(eligible):
        raise SystemExit(
            "Candidate losses must cover every eligible ID exactly once: "
            f"missing={missing[:20]} extra={extra[:20]}"
        )
    candidates = [dict(row, normalized_loss=losses[str(row["id"])]) for row in eligible]

    data = cfg["data"]
    consumed = read_jsonl(_resolve(root, data["consumed_35h"]))
    historical = read_jsonl(_resolve(root, data["historical_40h"]))
    initial = tuple(float(value) for value in data["loss_percentile_initial"])
    relaxed = tuple(float(value) for value in data["loss_percentile_relaxed"])
    ordered, selections = balanced_candidate_order(
        consumed,
        candidates,
        seed=int(data["selection_seed"]),
        initial_band=(initial[0], initial[1]),
        relaxed_band=(relaxed[0], relaxed[1]),
    )
    alignment = int(data["alignment_rows"])
    if alignment != int(cfg["training"]["global_effective_batch_size"]):
        raise SystemExit("Manifest alignment must equal the 16-sample W500 optimizer step")
    start_hours = float(cfg["start"]["actual_wenet_hours"])
    targets = [float(value) for value in data["targets_hours"]]
    target_seconds = {
        target: (target - start_hours) * 3600.0 for target in targets
    }
    if sum(row_duration(row) for row in ordered) < target_seconds[max(targets)]:
        raise SystemExit("Eligible candidates cannot fill the 50h continuation target")
    boundaries = {
        target: nearest_aligned_boundary(
            ordered, target_seconds=target_seconds[target], alignment=alignment
        )
        for target in targets
    }
    ordered_boundaries = [boundaries[target] for target in targets]
    if ordered_boundaries != sorted(set(ordered_boundaries)):
        raise SystemExit(f"Continuation boundaries are not strictly nested: {boundaries}")

    manifests = output / "manifests"
    staging_manifests = output / f"manifests.staging-{os.getpid()}"
    if manifests.exists() or staging_manifests.exists():
        raise SystemExit(f"Refusing to overwrite manifest directory: {manifests}")
    staging_manifests.mkdir()
    manifest_rows: dict[float, list[dict[str, Any]]] = {}
    staging_paths: dict[float, Path] = {}
    for target in targets:
        rows = ordered[: boundaries[target]]
        name = f"wenet_balanced_after35_to_{str(target).replace('.', 'p')}h.jsonl"
        path = staging_manifests / name
        _write_jsonl(path, rows)
        manifest_rows[target] = rows
        staging_paths[target] = path
    assert_exact_prefix(manifest_rows[40.0], manifest_rows[45.0])
    assert_exact_prefix(manifest_rows[45.0], manifest_rows[50.0])

    historical_delta = _historical_delta(consumed, historical)
    historical_boundary = nearest_aligned_boundary(
        historical_delta,
        target_seconds=target_seconds[40.0],
        alignment=alignment,
    )
    historical_rows = historical_delta[:historical_boundary]
    historical_staging = staging_manifests / "wenet_historical_after35_to_40p0h.jsonl"
    _write_jsonl(historical_staging, historical_rows)
    staging_manifests.rename(manifests)

    boundary_receipts: dict[str, Any] = {}
    for target in targets:
        path = manifests / staging_paths[target].name
        boundary_receipts[str(target)] = _manifest_receipt(
            path,
            manifest_rows[target],
            target_hours=target,
            target_seconds=target_seconds[target],
            prefix_of_50h=(
                [str(row["id"]) for row in manifest_rows[target]]
                == [str(row["id"]) for row in manifest_rows[50.0][: len(manifest_rows[target])]]
            ),
        )
    historical_path = manifests / historical_staging.name
    historical_receipt = _manifest_receipt(
        historical_path,
        historical_rows,
        target_hours=40.0,
        target_seconds=target_seconds[40.0],
        prefix_of_50h=False,
    )

    selected50 = selections[: boundaries[50.0]]
    reference_counts = Counter(bucket_key(row) for row in consumed)
    selected_counts = Counter(tuple(item["target_bucket"]) for item in selected50)
    quota_deviations = []
    total_reference = sum(reference_counts.values())
    for key in sorted(reference_counts):
        expected = len(selected50) * reference_counts[key] / total_reference
        actual = selected_counts[key]
        quota_deviations.append(
            {
                "type": "quota_deviation",
                "target_bucket": list(key),
                "expected_rows": expected,
                "actual_rows": actual,
                "deviation_rows": actual - expected,
            }
        )
    relaxation_rows = [
        {"type": "loss_band_relaxation", **item}
        for item in selected50
        if item["stage"] != "initial_10_90"
    ]
    relaxations_path = output / "selection_relaxations.jsonl"
    temporary_relaxations = relaxations_path.with_name(
        relaxations_path.name + f".tmp-{os.getpid()}"
    )
    _write_jsonl(temporary_relaxations, relaxation_rows + quota_deviations)
    os.replace(temporary_relaxations, relaxations_path)

    stage40_arms = []
    for index, arm in enumerate(cfg["stage40"]["arms"]):
        stream_path = historical_path if arm["stream"] == "historical" else (
            manifests / staging_paths[40.0].name
        )
        stage40_arms.append(
            {
                "name": arm["name"],
                "gpu": index,
                "stream": arm["stream"],
                "start_checkpoint": str(_resolve(root, cfg["start"]["checkpoint"]).resolve()),
                "start_hours": round(start_hours, 10),
                "target_hours": 40.0,
                "wenet_manifest": str(stream_path.resolve()),
                "wenet_continuation_cursor": 0,
                "official_sample_cursor": int(cfg["start"]["official_sample_cursor"]),
                "wenet_lr": float(arm["wenet_lr"]),
                "official_lr": float(arm["official_lr"]),
            }
        )
    arms_payload = {
        "passed": True,
        "config": str(_resolve(root, "configs/rounds/w500_adaptive_continuation.json").resolve()),
        "arms": stage40_arms,
    }
    _atomic_write_json(output / "stage40_arms.json", arms_payload)

    preparation = {
        "passed": True,
        "selection_seed": int(data["selection_seed"]),
        "loss_bands": {"initial": list(initial), "relaxed": list(relaxed)},
        "eligible": {
            "path": str(eligible_path.resolve()),
            "rows": len(eligible),
            "sha256": eligible_sha,
        },
        "candidate_losses": {
            "path": str(candidate_losses.resolve()),
            "sha256": sha256_file(candidate_losses),
            "rows": len(losses),
            "eligible_manifest_sha256": eligible_sha,
        },
        "consumed_35h": {
            "path": str(_resolve(root, data["consumed_35h"]).resolve()),
            "rows": len(consumed),
            "sha256": sha256_file(_resolve(root, data["consumed_35h"])),
        },
        "boundaries": boundary_receipts,
        "historical_40h": historical_receipt,
        "overlaps": audit["overlaps"],
        "relaxations": {
            "path": str(relaxations_path.resolve()),
            "rows": len(relaxation_rows),
            "quota_deviation_rows": len(quota_deviations),
            "sha256": sha256_file(relaxations_path),
        },
        "stage40_arms": str((output / "stage40_arms.json").resolve()),
        "nested_prefixes": True,
        "automatic_platform_upload": False,
    }
    _atomic_write_json(output / "preparation.json", preparation)

    checksum_paths = [
        eligible_path,
        output / "candidate_audit.jsonl",
        output / "candidate_audit_summary.json",
        relaxations_path,
        output / "stage40_arms.json",
        output / "preparation.json",
        historical_path,
        *(manifests / staging_paths[target].name for target in targets),
    ]
    checksum_text = "".join(
        f"{sha256_file(path)}  {path.relative_to(output)}\n" for path in checksum_paths
    )
    _atomic_write_text(output / "SHA256SUMS", checksum_text)
    _atomic_write_text(marker, "PASS\n")
    return preparation


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    config_path = _resolve(root, args.config)
    output = _resolve(root, args.output_dir)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if args.mode == "audit":
        report = audit_candidates(root, cfg, output)
    else:
        if args.candidate_losses is None:
            raise SystemExit("--candidate-losses is required in freeze mode")
        report = freeze_manifests(
            root,
            cfg,
            output,
            _resolve(root, args.candidate_losses),
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
