"""Pure policy helpers for the adaptive W500 continuation tournament.

This module deliberately contains no model or filesystem code.  Keeping the
ranking, guardrails, bucketing, aligned-boundary and branching policy pure
makes the frozen experiment registry independently testable.
"""

from __future__ import annotations

import math
from functools import cmp_to_key
from typing import Any, Iterable, Mapping


DEFAULT_GUARDRAILS: dict[str, float | int] = {
    "validation_tol2_min": 0.836880,
    "public_tol2_min": 0.869211,
    "public_insertions_max": 200,
    "public_severe_max": 25,
    "public_repeated_runaway_max": 0,
    "public_replacement_max": 10,
    "public_effective_max_length_max": 0,
    "ood_tol2_min": 0.325,
    "ood_cer_max": 0.346632,
}


def validation_proxy(*, cer: float, tol2: float) -> float:
    """Return the approved fixed-validation platform-score proxy."""

    return 70.0 * (1.0 - float(cer)) + 20.0 * float(tol2)


def guardrail_failures(
    row: Mapping[str, Any],
    *,
    thresholds: Mapping[str, float | int] | None = None,
) -> list[str]:
    """Return catastrophic vetoes without using public/OOD for ranking."""

    limits = {**DEFAULT_GUARDRAILS, **dict(thresholds or {})}
    try:
        validation = row["validation"]
        public = row["public"]
        ood = row["ood"]
        checks = (
            (
                float(validation["tol2"]) < float(limits["validation_tol2_min"]),
                "validation_tol2",
            ),
            (float(public["tol2"]) < float(limits["public_tol2_min"]), "public_tol2"),
            (
                int(public["insertions"]) > int(limits["public_insertions_max"]),
                "public_insertions",
            ),
            (int(public["severe"]) > int(limits["public_severe_max"]), "public_severe"),
            (
                int(public["repeated_runaway"])
                > int(limits["public_repeated_runaway_max"]),
                "public_repeated_runaway",
            ),
            (
                int(public["replacement"]) > int(limits["public_replacement_max"]),
                "public_replacement",
            ),
            (
                int(public["effective_max"])
                > int(limits["public_effective_max_length_max"]),
                "public_effective_max",
            ),
            (float(ood["tol2"]) < float(limits["ood_tol2_min"]), "ood_tol2"),
            (float(ood["cer"]) > float(limits["ood_cer_max"]), "ood_cer"),
            (not bool(row["complete"]), "incomplete"),
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return ["incomplete"]
    return [label for failed, label in checks if failed]


def _compare(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    left_validation = left["validation"]
    right_validation = right["validation"]
    left_proxy = validation_proxy(
        cer=left_validation["cer"], tol2=left_validation["tol2"]
    )
    right_proxy = validation_proxy(
        cer=right_validation["cer"], tol2=right_validation["tol2"]
    )
    if abs(left_proxy - right_proxy) >= 0.05:
        return -1 if left_proxy > right_proxy else 1
    keys = (
        (float(left_validation["cer"]), float(right_validation["cer"]), False),
        (float(left_validation["tol2"]), float(right_validation["tol2"]), True),
        (int(left_validation["severe"]), int(right_validation["severe"]), False),
        (float(left["hours"]), float(right["hours"]), False),
    )
    for left_value, right_value, descending in keys:
        if left_value == right_value:
            continue
        if descending:
            return -1 if left_value > right_value else 1
        return -1 if left_value < right_value else 1
    left_label = str(left["label"])
    right_label = str(right["label"])
    if left_label == right_label:
        return 0
    return -1 if left_label < right_label else 1


def rank_rows(
    rows: Iterable[dict[str, Any]],
    *,
    thresholds: Mapping[str, float | int] | None = None,
) -> list[dict[str, Any]]:
    """Rank eligible rows using fixed validation and nothing else."""

    eligible = [
        row for row in rows if not guardrail_failures(row, thresholds=thresholds)
    ]
    return sorted(eligible, key=cmp_to_key(_compare))


def _bin(value: float, edges: tuple[float, ...], prefix: str) -> str:
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {prefix} value: {value}")
    for left, right in zip(edges, edges[1:]):
        if left <= value < right:
            return f"{prefix}_{left:g}_{right:g}"
    if value < edges[0]:
        return f"{prefix}_below_{edges[0]:g}"
    return f"{prefix}_{edges[-1]:g}_plus"


def bucket_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    """Build the approved duration/text/speaker/program/source bucket."""

    duration = float(row.get("duration_s") or row.get("duration") or 0.0)
    text = str(row.get("text") or row.get("metric_text") or "")
    return (
        _bin(duration, (0, 3, 5, 8, 12, 20, 30), "dur"),
        _bin(float(len(text)), (1, 11, 16, 21, 31, 46, 80), "txt"),
        str(row.get("speaker_id") or "unknown"),
        str(row.get("program_group") or row.get("program") or "unknown"),
        str(row.get("source_tar") or row.get("source") or "unknown"),
    )


def middle_loss_rows(
    rows: list[dict[str, Any]], *, low: float, high: float
) -> list[dict[str, Any]]:
    """Return the rank-percentile loss band with deterministic tie ordering."""

    if not 0.0 <= low < high <= 1.0:
        raise ValueError(f"Invalid loss percentile band: {(low, high)}")
    ordered = sorted(
        rows,
        key=lambda row: (float(row["normalized_loss"]), str(row["id"])),
    )
    if any(not math.isfinite(float(row["normalized_loss"])) for row in ordered):
        raise ValueError("normalized_loss must be finite")
    start = math.ceil(len(ordered) * low)
    stop = math.ceil(len(ordered) * high)
    return ordered[start:stop]


def nearest_aligned_boundary(
    rows: list[dict[str, Any]], *, target_seconds: float, alignment: int
) -> int:
    """Find the complete-optimizer-step prefix nearest a duration target."""

    if alignment <= 0:
        raise ValueError("alignment must be positive")
    if not math.isfinite(float(target_seconds)) or target_seconds <= 0:
        raise ValueError("target_seconds must be finite and positive")
    candidates: list[tuple[float, int]] = []
    cumulative = 0.0
    for index, row in enumerate(rows, start=1):
        duration = float(row.get("duration_s") or row.get("duration") or 0.0)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid duration at row {index}: {duration}")
        cumulative += duration
        if index % alignment == 0:
            candidates.append((abs(cumulative - target_seconds), index))
    if not candidates:
        raise ValueError("No aligned manifest boundary")
    return min(candidates)[1]


def assert_exact_prefix(
    shorter: list[dict[str, Any]], longer: list[dict[str, Any]]
) -> None:
    """Raise when the ordered IDs in ``shorter`` are not an exact prefix."""

    if len(shorter) > len(longer):
        raise ValueError("Nested manifest prefix is longer than its parent")
    short_ids = [str(row["id"]) for row in shorter]
    long_ids = [str(row["id"]) for row in longer[: len(shorter)]]
    if short_ids != long_ids:
        raise ValueError("Nested manifest prefix mismatch")


def branch_top_two(
    parents: list[dict[str, Any]], *, target_hours: float, manifest: str
) -> list[dict[str, Any]]:
    """Create constant/half-LR branches for exactly two ranked parents."""

    if len(parents) != 2:
        raise ValueError("Exactly two parents are required")
    arms: list[dict[str, Any]] = []
    for rank, parent in enumerate(parents, start=1):
        for mode, scale in (("constant", 1.0), ("half", 0.5)):
            arms.append(
                {
                    "name": f"TOP{rank}_{mode.upper()}_TO_{target_hours:g}H",
                    "parent_label": parent["label"],
                    "start_checkpoint": parent["checkpoint"],
                    "start_hours": float(parent["hours"]),
                    "target_hours": float(target_hours),
                    "wenet_manifest": manifest,
                    "wenet_continuation_cursor": int(parent["wenet_cursor"]),
                    "official_sample_cursor": int(parent["official_cursor"]),
                    "wenet_lr": float(parent["wenet_lr"]) * scale,
                    "official_lr": float(parent["official_lr"]) * scale,
                    "lr_mode": mode,
                }
            )
    return arms
