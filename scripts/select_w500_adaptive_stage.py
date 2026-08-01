#!/usr/bin/env python3
"""Select adaptive W500 checkpoints using fixed validation only.

Public and OOD diagnostics are read solely to veto catastrophic candidates.
They are deliberately absent from every ordering key.  Stage 40 and 45 emit
four immutable next-stage arms (constant and half LR for each Top-2 parent);
stage 50 emits the global eligible acoustic winner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

from cantonese_asr.w500_adaptive import (
    branch_top_two,
    guardrail_failures,
    rank_rows,
    validation_proxy,
)


EXPECTED_ROWS = {"validation": 702, "public": 1900, "ood": 2000}
METADATA_FILES = (
    "candidate.json",
    "evaluation_candidate.json",
    "evaluation_task.json",
    "task.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument(
        "--target-hours", type=float, choices=(40.0, 45.0, 50.0), required=True
    )
    parser.add_argument("--balanced-next-manifest", type=Path)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--next-arms-output", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_jsonl_rows(path: Path) -> int:
    """Count and parse every non-empty JSONL row; malformed rows are incomplete."""

    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL {path}:{line_number}: {error}") from error
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _duration_seconds(row: dict[str, Any]) -> float:
    raw = row.get("duration_s", row.get("duration", row.get("seconds")))
    if raw is None:
        raise ValueError(f"Manifest row lacks duration: {row.get('id', '<unknown>')}")
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"Invalid manifest duration: {raw!r}")
    return value


def aligned_cursor_for_continuation(
    manifest: Path, *, target_seconds: float, alignment: int
) -> dict[str, float | int]:
    """Map exposure to the nearest complete-batch prefix in a new stream."""

    if alignment <= 0:
        raise ValueError("alignment must be positive")
    if not math.isfinite(target_seconds) or target_seconds < 0.0:
        raise ValueError("target_seconds must be finite and non-negative")
    rows = read_jsonl(manifest)
    candidates: list[tuple[float, int, float]] = [(abs(target_seconds), 0, 0.0)]
    cumulative = 0.0
    for index, row in enumerate(rows, start=1):
        cumulative += _duration_seconds(row)
        if index % alignment == 0:
            candidates.append((abs(cumulative - target_seconds), index, cumulative))
    _, cursor, actual = min(candidates, key=lambda item: (item[0], item[1]))
    return {
        "cursor": cursor,
        "actual_prefix_seconds": actual,
        "target_prefix_seconds": target_seconds,
        "deviation_seconds": actual - target_seconds,
        "manifest_sha256": sha256_file(manifest),
    }


def _number(
    mappings: Iterable[dict[str, Any]],
    keys: tuple[str, ...],
    *,
    default: float,
) -> float:
    for mapping in mappings:
        for key in keys:
            if key in mapping and mapping[key] is not None:
                value = float(mapping[key])
                if math.isfinite(value):
                    return value
    return default


def _integer(
    mappings: Iterable[dict[str, Any]],
    keys: tuple[str, ...],
    *,
    default: int,
) -> int:
    value = _number(mappings, keys, default=float(default))
    return int(value)


def load_surface(surface_dir: Path, surface: str) -> dict[str, Any]:
    """Normalize current evaluator output and materialize completeness evidence."""

    errors: list[str] = []
    summary_path = surface_dir / "summary.json"
    metrics_path = surface_dir / "metrics.json"
    summary: dict[str, Any] = {}
    metrics_file: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            summary = read_json(summary_path)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            errors.append(f"invalid_summary:{error}")
    if metrics_path.is_file():
        try:
            metrics_file = read_json(metrics_path)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            errors.append(f"invalid_metrics:{error}")
    if not summary and not metrics_file:
        errors.append("missing_metrics")

    metrics = summary.get("metrics", metrics_file)
    diagnostics = metrics_file.get("diagnostics", {})
    operations = summary.get("operations", diagnostics)
    generation = summary.get("generation", {})
    generation_path = surface_dir / "generation.json"
    if not generation and generation_path.is_file():
        try:
            generation = read_json(generation_path)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            errors.append(f"invalid_generation:{error}")
    mappings = [metrics, summary, diagnostics]
    cer = _number(mappings, ("cer",), default=float("inf"))
    tol2 = _number(
        mappings,
        ("sentence_accuracy_tol2", "tol2"),
        default=float("-inf"),
    )
    severe = _integer(
        [summary, diagnostics, metrics],
        ("severe_error_count", "severe"),
        default=10**12,
    )
    normalized: dict[str, Any] = {
        "cer": cer,
        "tol2": tol2,
        "severe": severe,
        "path": str(surface_dir.resolve()),
    }
    if surface == "public":
        normalized.update(
            {
                "insertions": _integer(
                    [operations, diagnostics], ("insertions",), default=10**12
                ),
                "repeated_runaway": _integer(
                    [generation], ("repeated_runaway_count", "repeated_runaway"), default=10**12
                ),
                "replacement": _integer(
                    [generation],
                    ("replacement_character_count", "replacement"),
                    default=10**12,
                ),
                "effective_max": _integer(
                    [generation],
                    ("effective_max_length_count", "effective_max"),
                    default=10**12,
                ),
            }
        )

    for required in (
        "metrics.json",
        "top_confusions.csv",
        "error_examples.json",
        "generation.json",
    ):
        if not (surface_dir / required).is_file():
            errors.append(f"missing_{required}")

    observed_rows: dict[str, int | None] = {}
    for name in ("predictions.jsonl", "generation_tokens.jsonl"):
        path = surface_dir / name
        if not path.is_file():
            errors.append(f"missing_{name}")
            observed_rows[name] = None
            continue
        try:
            observed_rows[name] = count_jsonl_rows(path)
        except ValueError as error:
            errors.append(str(error))
            observed_rows[name] = None
            continue
        if observed_rows[name] != EXPECTED_ROWS[surface]:
            errors.append(
                f"{name}_rows:{observed_rows[name]}!={EXPECTED_ROWS[surface]}"
            )
    normalized["completeness"] = {
        "expected_rows": EXPECTED_ROWS[surface],
        "observed_rows": observed_rows,
        "errors": errors,
        "complete": not errors,
    }
    return normalized


def _metadata_candidates(surface_root: Path, evaluation_root: Path) -> list[Path]:
    candidates: list[Path] = []
    current = surface_root
    while True:
        candidates.extend(current / name for name in METADATA_FILES)
        if current == evaluation_root or evaluation_root not in current.parents:
            break
        current = current.parent
    return candidates


def load_candidate_metadata(surface_root: Path, evaluation_root: Path) -> dict[str, Any]:
    for path in _metadata_candidates(surface_root, evaluation_root):
        if path.is_file():
            value = read_json(path)
            if "candidate" in value and isinstance(value["candidate"], dict):
                value = value["candidate"]
            if "task" in value and isinstance(value["task"], dict):
                value = value["task"]
            result = dict(value)
            result["metadata_path"] = str(path.resolve())
            result["metadata_sha256"] = sha256_file(path)
            return result
    return {}


def _find_adaptive_state(metadata: dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    checkpoint_value = metadata.get("checkpoint", metadata.get("model_dir"))
    if not checkpoint_value:
        return {}, None
    checkpoint = Path(str(checkpoint_value)).resolve()
    path = checkpoint / "adaptive_state.json"
    if not path.is_file():
        return {}, None
    return read_json(path), path


def normalize_candidate(surface_root: Path, evaluation_root: Path) -> dict[str, Any]:
    metadata = load_candidate_metadata(surface_root, evaluation_root)
    adaptive, adaptive_path = _find_adaptive_state(metadata)
    validation = load_surface(surface_root / "validation", "validation")
    public = load_surface(surface_root / "public", "public")
    ood = load_surface(surface_root / "ood", "ood")
    complete = all(
        surface["completeness"]["complete"]
        for surface in (validation, public, ood)
    )
    label = str(metadata.get("label") or surface_root.relative_to(evaluation_root))
    checkpoint = metadata.get("checkpoint", metadata.get("model_dir"))
    if checkpoint is None and adaptive_path is not None:
        checkpoint = str(adaptive_path.parent)
    hours_value = _number(
        [metadata, adaptive],
        ("hours", "actual_wenet_hours", "wenet_hours_target"),
        default=float("nan"),
    )
    wenet_lr_value = _number([metadata, adaptive], ("wenet_lr",), default=float("nan"))
    official_lr_value = _number(
        [metadata, adaptive], ("official_lr",), default=float("nan")
    )
    wenet_cursor = _integer(
        [metadata, adaptive],
        ("wenet_cursor", "wenet_sample_cursor", "wenet_continuation_cursor"),
        default=-1,
    )
    official_cursor = _integer(
        [metadata, adaptive],
        ("official_cursor", "official_sample_cursor"),
        default=-1,
    )
    model_hash = metadata.get(
        "model_safetensors_sha256",
        metadata.get("weight_sha256", adaptive.get("model_safetensors_sha256")),
    )
    hours = hours_value if math.isfinite(hours_value) else None
    wenet_lr = wenet_lr_value if math.isfinite(wenet_lr_value) else None
    official_lr = official_lr_value if math.isfinite(official_lr_value) else None
    identity = adaptive.get("identity", {}) if isinstance(adaptive.get("identity"), dict) else {}
    wenet_manifest = metadata.get(
        "wenet_manifest",
        metadata.get("wenet_manifest_path", identity.get("wenet_manifest")),
    )
    cumulative_wenet_seconds = _number(
        [metadata, adaptive],
        ("cumulative_wenet_seconds",),
        default=(hours * 3600.0 if hours is not None else float("nan")),
    )
    if not math.isfinite(cumulative_wenet_seconds):
        cumulative_wenet_seconds = None
    metadata_errors: list[str] = []
    for key, value in (
        ("checkpoint", checkpoint),
        ("model_safetensors_sha256", model_hash),
        ("hours", hours),
        ("wenet_lr", wenet_lr),
        ("official_lr", official_lr),
    ):
        if value is None:
            metadata_errors.append(f"missing_{key}")
    if wenet_cursor < 0:
        metadata_errors.append("missing_wenet_cursor")
    if official_cursor < 0:
        metadata_errors.append("missing_official_cursor")
    complete = complete and not metadata_errors
    row: dict[str, Any] = {
        "label": label,
        "checkpoint": str(Path(str(checkpoint)).resolve()) if checkpoint else None,
        "outputs_pre": str(surface_root.resolve()),
        "model_safetensors_sha256": model_hash,
        "hours": hours,
        "wenet_lr": wenet_lr,
        "official_lr": official_lr,
        "wenet_cursor": wenet_cursor,
        "official_cursor": official_cursor,
        "wenet_manifest": (
            str(Path(str(wenet_manifest)).resolve()) if wenet_manifest else None
        ),
        "cumulative_wenet_seconds": cumulative_wenet_seconds,
        "validation": {
            "cer": validation["cer"],
            "tol2": validation["tol2"],
            "severe": validation["severe"],
        },
        "public": {
            "cer": public["cer"],
            "tol2": public["tol2"],
            "severe": public["severe"],
            "insertions": public["insertions"],
            "repeated_runaway": public["repeated_runaway"],
            "replacement": public["replacement"],
            "effective_max": public["effective_max"],
        },
        "ood": {"cer": ood["cer"], "tol2": ood["tol2"], "severe": ood["severe"]},
        "complete": complete,
        "diagnostic_evidence": {
            "validation": validation["completeness"],
            "public": public["completeness"],
            "ood": ood["completeness"],
            "metadata_path": metadata.get("metadata_path"),
            "metadata_sha256": metadata.get("metadata_sha256"),
            "adaptive_state_path": str(adaptive_path) if adaptive_path else None,
            "adaptive_state_sha256": sha256_file(adaptive_path) if adaptive_path else None,
            "metadata_errors": metadata_errors,
        },
    }
    row["validation_proxy"] = (
        validation_proxy(cer=row["validation"]["cer"], tol2=row["validation"]["tol2"])
        if math.isfinite(row["validation"]["cer"])
        and math.isfinite(row["validation"]["tol2"])
        else None
    )
    return row


def discover_candidates(evaluation_root: Path) -> list[dict[str, Any]]:
    roots: list[Path] = []
    for validation_dir in evaluation_root.rglob("validation"):
        if not validation_dir.is_dir():
            continue
        surface_root = validation_dir.parent
        if (surface_root / "public").is_dir() and (surface_root / "ood").is_dir():
            roots.append(surface_root)
    unique_roots = sorted(set(roots), key=lambda path: str(path.relative_to(evaluation_root)))
    if not unique_roots:
        raise SystemExit(f"No complete validation/public/OOD directory triplets: {evaluation_root}")
    rows = [normalize_candidate(path, evaluation_root) for path in unique_roots]
    labels = [row["label"] for row in rows]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise SystemExit(f"Duplicate candidate labels: {duplicates}")
    return rows


def _branch_parent(row: dict[str, Any]) -> dict[str, Any]:
    required = {
        "checkpoint": row.get("checkpoint"),
        "model_safetensors_sha256": row.get("model_safetensors_sha256"),
        "hours": row.get("hours"),
        "wenet_lr": row.get("wenet_lr"),
        "official_lr": row.get("official_lr"),
        "wenet_cursor": row.get("wenet_cursor"),
        "official_cursor": row.get("official_cursor"),
        "cumulative_wenet_seconds": row.get("cumulative_wenet_seconds"),
    }
    missing = [
        key
        for key, value in required.items()
        if value is None
        or (isinstance(value, float) and not math.isfinite(value))
        or (key.endswith("cursor") and int(value) < 0)
    ]
    if missing:
        raise SystemExit(f"Selected parent {row['label']} lacks branch fields: {missing}")
    return {
        "label": row["label"],
        "checkpoint": row["checkpoint"],
        "model_safetensors_sha256": row["model_safetensors_sha256"],
        "hours": float(row["hours"]),
        "wenet_lr": float(row["wenet_lr"]),
        "official_lr": float(row["official_lr"]),
        "wenet_cursor": int(row["wenet_cursor"]),
        "official_cursor": int(row["official_cursor"]),
        "wenet_manifest": row.get("wenet_manifest"),
        "cumulative_wenet_seconds": float(row["cumulative_wenet_seconds"]),
    }


def remap_branch_cursor(
    parent: dict[str, Any],
    *,
    balanced_manifest: Path,
    root_wenet_seconds: float,
    alignment: int,
) -> dict[str, Any]:
    """Return a same-stream reuse or audited oldstream -> balanced remap."""

    source_manifest = parent.get("wenet_manifest")
    same_stream = bool(source_manifest) and Path(str(source_manifest)).resolve() == balanced_manifest.resolve()
    if same_stream:
        rows = read_jsonl(balanced_manifest)
        cursor = int(parent["wenet_cursor"])
        if cursor < 0 or cursor > len(rows) or cursor % alignment:
            raise SystemExit(f"Selected parent has invalid balanced cursor: {cursor}")
        prefix_seconds = sum(_duration_seconds(row) for row in rows[:cursor])
        return {
            "mode": "reuse_same_balanced_stream",
            "source_manifest": str(Path(str(source_manifest)).resolve()),
            "source_cursor": cursor,
            "target_manifest": str(balanced_manifest.resolve()),
            "target_cursor": cursor,
            "target_prefix_seconds": prefix_seconds,
            "actual_prefix_seconds": prefix_seconds,
            "deviation_seconds": 0.0,
            "manifest_sha256": sha256_file(balanced_manifest),
        }
    parent_seconds = float(parent["cumulative_wenet_seconds"])
    requested = max(0.0, parent_seconds - float(root_wenet_seconds))
    remap = aligned_cursor_for_continuation(
        balanced_manifest, target_seconds=requested, alignment=alignment
    )
    return {
        "mode": "remap_to_balanced_stream",
        "source_manifest": source_manifest,
        "source_cursor": int(parent["wenet_cursor"]),
        "target_manifest": str(balanced_manifest.resolve()),
        "target_cursor": int(remap["cursor"]),
        "root_wenet_seconds": float(root_wenet_seconds),
        "parent_cumulative_wenet_seconds": parent_seconds,
        **remap,
    }


def build_selection(
    rows: list[dict[str, Any]],
    *,
    target_hours: float,
    balanced_next_manifest: Path | None,
    root_wenet_seconds: float = 35.101072222222214 * 3600.0,
    alignment: int = 16,
    thresholds: dict[str, float | int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        row["guardrail_failures"] = guardrail_failures(
            row, thresholds=thresholds
        )
        row["eligible"] = not row["guardrail_failures"]
        candidates.append(row)
    ranked = rank_rows(candidates, thresholds=thresholds)
    if not ranked:
        raise SystemExit("No eligible adaptive W500 checkpoint")
    ineligible = [row for row in candidates if not row["eligible"]]
    top2 = ranked[:2]
    branches: list[dict[str, Any]] = []
    if target_hours in {40.0, 45.0}:
        if len(top2) != 2:
            raise SystemExit("Two eligible parents are required for the next tournament stage")
        if balanced_next_manifest is None:
            raise SystemExit("--balanced-next-manifest is required before stage 50")
        next_target = target_hours + 5.0
        parents = [_branch_parent(row) for row in top2]
        branches = branch_top_two(
            parents,
            target_hours=next_target,
            manifest=str(balanced_next_manifest.resolve()),
        )
        parent_by_label = {row["label"]: row for row in parents}
        for branch in branches:
            parent = parent_by_label[branch["parent_label"]]
            remap = remap_branch_cursor(
                parent,
                balanced_manifest=balanced_next_manifest,
                root_wenet_seconds=root_wenet_seconds,
                alignment=alignment,
            )
            branch["wenet_continuation_cursor"] = int(remap["target_cursor"])
            branch["continuation_wenet_seconds"] = float(
                remap["actual_prefix_seconds"]
            )
            branch["cursor_remap"] = remap
            branch["parent_model_sha256"] = parent["model_safetensors_sha256"]
            branch["ranking_parent"] = {
                "label": parent["label"],
                "model_safetensors_sha256": parent["model_safetensors_sha256"],
            }
    payload = {
        "schema_version": 1,
        "target_hours": target_hours,
        "ranking_surface": "fixed_validation_only",
        "public_ood_role": "catastrophic_veto_only",
        "public_ood_used_for_ranking": False,
        "candidates": candidates,
        "eligible": ranked,
        "ineligible": ineligible,
        "ranking": [row["label"] for row in ranked],
        "selected": {
            "top2": top2,
            "winner": ranked[0] if target_hours == 50.0 else None,
        },
        # Compatibility aliases for the approved implementation plan.
        "top2": top2,
        "winner": ranked[0] if target_hours == 50.0 else None,
        "branches": branches,
        "next_stage_arms": branches,
        "automatic_platform_upload": False,
    }
    return payload, branches


def write_json_exclusive(path: Path, value: Any) -> None:
    if path.exists():
        raise SystemExit(f"Refusing to overwrite immutable selection output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + f".tmp-{os.getpid()}")
    if staging.exists():
        raise SystemExit(f"Stale selection staging path: {staging}")
    staging.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    staging.rename(path)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    evaluation_root = args.evaluation_root.resolve()
    if not config_path.is_file():
        raise SystemExit(f"Missing config: {config_path}")
    if not evaluation_root.is_dir():
        raise SystemExit(f"Missing evaluation root: {evaluation_root}")
    config = read_json(config_path)
    ranking_surface = config.get("selection", {}).get("ranking_surface")
    if ranking_surface not in (None, "fixed_validation_only"):
        raise SystemExit(f"Forbidden ranking surface: {ranking_surface}")
    if args.target_hours in {40.0, 45.0} and args.next_arms_output is None:
        raise SystemExit("--next-arms-output is required for stage 40 and 45")
    if args.target_hours == 50.0 and args.next_arms_output is not None:
        raise SystemExit("Stage 50 selects a winner and must not emit next arms")

    rows = discover_candidates(evaluation_root)
    payload, branches = build_selection(
        rows,
        target_hours=float(args.target_hours),
        balanced_next_manifest=(
            args.balanced_next_manifest.resolve()
            if args.balanced_next_manifest is not None
            else None
        ),
        root_wenet_seconds=float(config["start"]["actual_wenet_hours"]) * 3600.0,
        alignment=int(config.get("data", {}).get("alignment_rows", 16)),
        thresholds=config.get("selection", {}).get("guardrails"),
    )
    payload["inputs"] = {
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "evaluation_root": str(evaluation_root),
    }
    write_json_exclusive(args.selection_output.resolve(), payload)
    if args.next_arms_output is not None:
        arm_payload = {
            "schema_version": 1,
            "ranking_surface": "fixed_validation_only",
            "public_ood_used_for_ranking": False,
            "selection": str(args.selection_output.resolve()),
            "selection_sha256": sha256_file(args.selection_output.resolve()),
            "arms": branches,
        }
        write_json_exclusive(args.next_arms_output.resolve(), arm_payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
