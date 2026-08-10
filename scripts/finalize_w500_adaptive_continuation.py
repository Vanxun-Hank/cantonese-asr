#!/usr/bin/env python3
"""Finalize W500 adaptive continuation with fixed-validation-only ranking."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import tempfile
from functools import cmp_to_key
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SURFACES = ("validation", "public", "ood")
EXPECTED_ROWS = {"validation": 702, "public": 1900, "ood": 2000}
REQUIRED_EVALUATION_FILES = (
    "metrics.json",
    "top_confusions.csv",
    "error_examples.json",
    "generation.json",
    "predictions.jsonl",
    "generation_tokens.jsonl",
)
PACKAGE_LABELS = ("RAW_WINNER", "MIX15", "MIX30", "MIX45", "MIX60")
MIX_WEIGHTS = {"MIX15": 0.15, "MIX30": 0.30, "MIX45": 0.45, "MIX60": 0.60}
DEFAULT_GUARDRAILS = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--interpolation-root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--curves", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Required JSON file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            count += 1
    return count


def _metric_value(container: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in container:
            return container[name]
    raise ValueError(f"Missing metric; expected one of {names}")


def compact_surface(surface: dict[str, Any], name: str) -> dict[str, Any]:
    """Normalize direct or existing-report surface metrics to one schema."""

    metrics = surface.get("metrics", surface)
    operations = surface.get("operations", surface)
    generation = surface.get("generation", surface)
    compact = {
        "cer": float(_metric_value(metrics, "cer")),
        "tol2": float(
            _metric_value(metrics, "tol2", "sentence_accuracy_tol2", "accuracy_tol2")
        ),
        "severe": int(
            surface.get(
                "severe",
                surface.get("severe_error_count", metrics.get("severe", 0)),
            )
        ),
    }
    if name == "public":
        compact.update(
            {
                "insertions": int(_metric_value(operations, "insertions")),
                "repeated_runaway": int(
                    _metric_value(
                        generation, "repeated_runaway", "repeated_runaway_count"
                    )
                ),
                "replacement": int(
                    _metric_value(
                        generation, "replacement", "replacement_character_count"
                    )
                ),
                "effective_max": int(
                    _metric_value(
                        generation,
                        "effective_max",
                        "effective_max_length_count",
                    )
                ),
            }
        )
    return compact


def normalize_row(row: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    label = str(row.get("label") or row.get("name") or row.get("arm") or "").strip()
    if not label:
        raise ValueError(f"Candidate row has no label in {source_path}")
    source_surfaces = row.get("surfaces", row)
    surfaces: dict[str, dict[str, Any]] = {}
    for surface in SURFACES:
        value = source_surfaces.get(surface)
        if not isinstance(value, dict):
            raise ValueError(f"{label} has no {surface} metrics in {source_path}")
        surfaces[surface] = compact_surface(value, surface)
    checkpoint_value = row.get("checkpoint", row.get("model_dir", row.get("path")))
    if isinstance(checkpoint_value, dict):
        checkpoint_value = checkpoint_value.get("path") or checkpoint_value.get("model_dir")
    weight_hash = (
        row.get("weight_sha256")
        or row.get("model_safetensors_sha256")
        or (
            row.get("checkpoint", {}).get("model_safetensors_sha256")
            if isinstance(row.get("checkpoint"), dict)
            else None
        )
    )
    return {
        "label": label,
        "hours": float(
            row.get(
                "hours",
                row.get(
                    "actual_wenet_hours",
                    row.get("target_wenet_hours", row.get("wenet_hours", 0.0)),
                ),
            )
        ),
        "checkpoint": str(checkpoint_value) if checkpoint_value is not None else None,
        "weight_sha256": str(weight_hash) if weight_hash else None,
        "outputs_pre": row.get("outputs_pre") or row.get("outputs_pre_dir"),
        "complete": bool(row.get("complete", True)),
        "validation": surfaces["validation"],
        "public": surfaces["public"],
        "ood": surfaces["ood"],
        "source_path": str(source_path.resolve()),
        "raw": row,
    }


def validation_proxy(row: dict[str, Any], config: dict[str, Any]) -> float:
    selection = config.get("selection", {})
    return float(selection.get("proxy_cer_weight", 70.0)) * (
        1.0 - float(row["validation"]["cer"])
    ) + float(selection.get("proxy_tol2_weight", 20.0)) * float(
        row["validation"]["tol2"]
    )


def guardrail_failures(row: dict[str, Any], config: dict[str, Any]) -> list[str]:
    thresholds = dict(DEFAULT_GUARDRAILS)
    thresholds.update(config.get("selection", {}).get("guardrails", {}))
    validation, public, ood = row["validation"], row["public"], row["ood"]
    checks = (
        (validation["tol2"] < thresholds["validation_tol2_min"], "validation_tol2"),
        (public["tol2"] < thresholds["public_tol2_min"], "public_tol2"),
        (public["insertions"] > thresholds["public_insertions_max"], "public_insertions"),
        (public["severe"] > thresholds["public_severe_max"], "public_severe"),
        (
            public["repeated_runaway"]
            > thresholds["public_repeated_runaway_max"],
            "public_repeated_runaway",
        ),
        (public["replacement"] > thresholds["public_replacement_max"], "public_replacement"),
        (
            public["effective_max"]
            > thresholds["public_effective_max_length_max"],
            "public_effective_max",
        ),
        (ood["tol2"] < thresholds["ood_tol2_min"], "ood_tol2"),
        (ood["cer"] > thresholds["ood_cer_max"], "ood_cer"),
        (not row["complete"], "incomplete"),
    )
    return [label for failed, label in checks if failed]


def _compare_rows(
    left: dict[str, Any], right: dict[str, Any], config: dict[str, Any]
) -> int:
    left_proxy = validation_proxy(left, config)
    right_proxy = validation_proxy(right, config)
    tie = float(config.get("selection", {}).get("proxy_tie", 0.05))
    if abs(left_proxy - right_proxy) >= tie:
        return -1 if left_proxy > right_proxy else 1
    keys = (
        (float(left["validation"]["cer"]), float(right["validation"]["cer"]), False),
        (float(left["validation"]["tol2"]), float(right["validation"]["tol2"]), True),
        (int(left["validation"]["severe"]), int(right["validation"]["severe"]), False),
        (float(left["hours"]), float(right["hours"]), False),
    )
    for left_value, right_value, descending in keys:
        if left_value != right_value:
            if descending:
                return -1 if left_value > right_value else 1
            return -1 if left_value < right_value else 1
    return -1 if left["label"] < right["label"] else int(left["label"] != right["label"])


def rank_rows(
    rows: Iterable[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    eligible = [row for row in rows if not guardrail_failures(row, config)]
    return sorted(eligible, key=cmp_to_key(lambda a, b: _compare_rows(a, b, config)))


def _candidate_objects(value: Any) -> list[dict[str, Any]]:
    """Extract rows from the selector schemas used across adaptive stages."""

    output: list[dict[str, Any]] = []
    if isinstance(value, list):
        for item in value:
            output.extend(_candidate_objects(item))
        return output
    if not isinstance(value, dict):
        return output
    if ("label" in value or "name" in value) and (
        "validation" in value or "surfaces" in value
    ):
        return [value]
    for key in (
        "eligible",
        "ineligible",
        "rows",
        "candidates",
        "selected",
        "top2",
        "all_rows",
        "checkpoints",
    ):
        if key in value:
            output.extend(_candidate_objects(value[key]))
    return output


def _selection_files(stage_root: Path) -> list[Path]:
    files = [
        path
        for path in stage_root.rglob("*.json")
        if "select" in path.name.lower() and path.is_file()
    ]
    if not files:
        files = [
            path
            for path in stage_root.rglob("*.json")
            if path.name.lower() in {"candidates.json", "selected.json", "results.json"}
        ]
    return sorted(set(files))


def load_stage_rows(stage_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selection_files = _selection_files(stage_root)
    if not selection_files:
        raise ValueError(f"No stage selection JSON found under {stage_root}")
    rows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for path in selection_files:
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Stage selection must be an object: {path}")
        ranking_surface = payload.get("ranking_surface", payload.get("selection_surface"))
        if ranking_surface != "fixed_validation_only":
            raise ValueError(f"Stage selection is not fixed-validation-only: {path}")
        if payload.get("public_ood_used_for_ranking", False) is not False:
            raise ValueError(f"Stage selection used Public/OOD for ranking: {path}")
        extracted = _candidate_objects(payload)
        if not extracted:
            raise ValueError(f"Stage selection contains no checkpoint rows: {path}")
        evaluation_root_value = payload.get("inputs", {}).get("evaluation_root")
        evaluation_root = (
            Path(str(evaluation_root_value)).resolve()
            if evaluation_root_value
            else None
        )
        for candidate in extracted:
            candidate = dict(candidate)
            if not candidate.get("outputs_pre") and evaluation_root is not None:
                label = str(
                    candidate.get("label")
                    or candidate.get("name")
                    or candidate.get("arm")
                    or ""
                )
                candidate["outputs_pre"] = str((evaluation_root / label).resolve())
            rows.append(normalize_row(candidate, source_path=path))
        receipts.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "ranking_surface": ranking_surface,
                "public_ood_used_for_ranking": False,
                "row_count": len(extracted),
            }
        )

    deduplicated: dict[str, dict[str, Any]] = {}
    for row in rows:
        previous = deduplicated.get(row["label"])
        if previous is not None:
            left_hash, right_hash = previous["weight_sha256"], row["weight_sha256"]
            if left_hash and right_hash and left_hash != right_hash:
                raise ValueError(f"Conflicting duplicate stage row: {row['label']}")
            continue
        deduplicated[row["label"]] = row
    return list(deduplicated.values()), receipts


def _canonical_package_label(label: str) -> str | None:
    upper = label.upper().replace("-", "_")
    for name in PACKAGE_LABELS:
        if upper == name or upper.endswith("/" + name) or upper.endswith("_" + name):
            return name
    if upper in {"RAW", "ACOUSTIC_WINNER", "RAW_ACOUSTIC_WINNER"}:
        return "RAW_WINNER"
    return None


def load_interpolation_rows(interpolation_root: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    ignored = set(REQUIRED_EVALUATION_FILES) | {
        "interpolation_receipt.json",
        "config.json",
    }
    for path in sorted(interpolation_root.rglob("*.json")):
        if path.name in ignored:
            continue
        payload = read_json(path)
        for candidate in _candidate_objects(payload):
            canonical = _canonical_package_label(
                str(candidate.get("label") or candidate.get("name") or "")
            )
            if canonical is None:
                continue
            normalized = normalize_row(candidate, source_path=path)
            normalized["label"] = canonical
            previous = rows.get(canonical)
            if previous is not None and previous["weight_sha256"] != normalized["weight_sha256"]:
                raise ValueError(f"Conflicting final candidate rows for {canonical}")
            rows[canonical] = normalized
    missing = [label for label in PACKAGE_LABELS if label not in rows]
    if missing:
        raise ValueError(
            f"Interpolation results must contain raw plus four mixes; missing={missing}"
        )
    return rows


def _resolve_path(value: str | Path, source_path: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = source_path.parent / path
    return path.resolve()


def verify_model(row: dict[str, Any]) -> dict[str, Any]:
    source_path = Path(row["source_path"])
    if not row.get("checkpoint"):
        raise ValueError(f"Candidate has no model checkpoint: {row['label']}")
    checkpoint = _resolve_path(row["checkpoint"], source_path)
    weight = checkpoint / "model.safetensors"
    if not weight.is_file():
        raise ValueError(f"Missing model.safetensors for {row['label']}: {weight}")
    if not row.get("weight_sha256"):
        raise ValueError(f"Candidate has no recorded model SHA-256: {row['label']}")
    actual = sha256_file(weight)
    if actual != row["weight_sha256"]:
        raise ValueError(
            f"Model SHA-256 mismatch for {row['label']}: "
            f"recorded={row['weight_sha256']} actual={actual}"
        )
    row["checkpoint"] = str(checkpoint)
    return {
        "checkpoint": str(checkpoint),
        "model_safetensors_sha256": actual,
    }


def _metric_declared_rows(metrics: Any) -> int | None:
    if not isinstance(metrics, dict):
        return None
    for key in ("rows", "row_count", "num_rows", "n_samples", "count"):
        value = metrics.get(key)
        if isinstance(value, int):
            return value
    nested = metrics.get("metrics")
    return _metric_declared_rows(nested) if isinstance(nested, dict) else None


def validate_outputs_pre(outputs_pre: Path) -> dict[str, Any]:
    """Require the complete three-surface ``outputs_pre`` evidence bundle.

    An explicit ``outputs_pre`` field may reference an evaluator root named
    ``historical_current``.  The field and the required contents define the
    contract; the referenced directory itself need not literally be named
    ``outputs_pre``.
    """

    if not outputs_pre.is_dir():
        raise ValueError(f"Mandatory outputs_pre evidence root is missing: {outputs_pre}")
    surfaces: dict[str, Any] = {}
    for surface in SURFACES:
        directory = outputs_pre / surface
        if not directory.is_dir():
            raise ValueError(f"Missing outputs_pre surface directory: {directory}")
        missing = [name for name in REQUIRED_EVALUATION_FILES if not (directory / name).is_file()]
        if missing:
            raise ValueError(f"Missing evaluation files in {directory}: {missing}")
        expected = EXPECTED_ROWS[surface]
        prediction_rows = count_jsonl(directory / "predictions.jsonl")
        generation_rows = count_jsonl(directory / "generation_tokens.jsonl")
        if prediction_rows != expected or generation_rows != expected:
            raise ValueError(
                f"Incomplete {surface} rows in {directory}: expected={expected}, "
                f"predictions={prediction_rows}, generation_tokens={generation_rows}"
            )
        metrics = read_json(directory / "metrics.json")
        declared = _metric_declared_rows(metrics)
        if declared is not None and declared != expected:
            raise ValueError(
                f"metrics.json row count mismatch in {directory}: "
                f"expected={expected}, declared={declared}"
            )
        # Parse mandatory JSON evidence before report completion.
        read_json(directory / "error_examples.json")
        read_json(directory / "generation.json")
        surfaces[surface] = {
            "directory": str(directory.resolve()),
            "expected_rows": expected,
            "prediction_rows": prediction_rows,
            "generation_token_rows": generation_rows,
            "file_sha256": {
                name: sha256_file(directory / name) for name in REQUIRED_EVALUATION_FILES
            },
        }
    return {"passed": True, "outputs_pre": str(outputs_pre.resolve()), "surfaces": surfaces}


def _outputs_pre_for(row: dict[str, Any]) -> Path:
    source_path = Path(row["source_path"])
    explicit = row.get("outputs_pre")
    if isinstance(explicit, str):
        return _resolve_path(explicit, source_path)
    if isinstance(explicit, dict):
        roots = {
            _resolve_path(value, source_path).parent
            for value in explicit.values()
            if isinstance(value, str)
        }
        if len(roots) == 1:
            return roots.pop()
        raise ValueError(f"Ambiguous outputs_pre mapping for {row['label']}: {explicit}")
    if row.get("checkpoint"):
        return Path(row["checkpoint"]) / "outputs_pre"
    raise ValueError(f"No outputs_pre location for {row['label']}")


def _verify_interpolation_receipts(
    final_rows: dict[str, dict[str, Any]], raw_weight_sha256: str
) -> dict[str, Any]:
    receipts: dict[str, Any] = {}
    encoder_digest: str | None = None
    immutable_hashes: dict[str, str] | None = None
    for label in PACKAGE_LABELS:
        row = final_rows[label]
        if label == "RAW_WINNER":
            if row["weight_sha256"] != raw_weight_sha256:
                raise ValueError(
                    "RAW_WINNER model hash does not match the fixed-validation acoustic winner"
                )
            continue
        receipt_path = Path(row["checkpoint"]) / "interpolation_receipt.json"
        receipt = read_json(receipt_path)
        if receipt.get("passed") is not True or receipt.get("name") != label:
            raise ValueError(f"Invalid interpolation receipt: {receipt_path}")
        alpha = receipt.get("alpha")
        stable_weight = receipt.get("stable_weight")
        if (
            alpha is None
            or stable_weight is None
            or abs(float(alpha) - MIX_WEIGHTS[label]) > 1e-12
            or abs(float(stable_weight) - MIX_WEIGHTS[label]) > 1e-12
        ):
            raise ValueError(f"Wrong interpolation alpha in {receipt_path}")
        if receipt.get("output_model_sha256") != row["weight_sha256"]:
            raise ValueError(f"Interpolation output hash mismatch in {receipt_path}")
        if (
            receipt.get("endpoints", {}).get("acoustic_model_sha256")
            != raw_weight_sha256
        ):
            raise ValueError(
                f"Interpolation acoustic endpoint does not match RAW_WINNER in {receipt_path}"
            )
        digest = receipt.get("encoder_tensor_sha256")
        if not digest:
            raise ValueError(f"Missing Encoder digest in {receipt_path}")
        if encoder_digest is None:
            encoder_digest = str(digest)
        elif digest != encoder_digest:
            raise ValueError("MIX15/30/45/60 Encoder tensor digests are not identical")
        checks = receipt.get("persisted_tensor_audit", {}).get("checks", {})
        digests = receipt.get("persisted_tensor_audit", {}).get("digests", {})
        if (
            digests.get("acoustic_encoder") != digest
            or digests.get("mixed_encoder") != digest
        ):
            raise ValueError(
                f"RAW/MIX Encoder digest parity is absent or failed in {receipt_path}"
            )
        required_checks = (
            "encoder_exact",
            "alpha_0_reproduces_acoustic_endpoint",
            "alpha_1_reproduces_stable_decoder_with_acoustic_encoder",
            "sampled_formula_checks_passed",
        )
        if not all(checks.get(name) is True for name in required_checks):
            raise ValueError(f"Incomplete interpolation tensor audit in {receipt_path}")
        receipt_immutable = receipt.get("immutable_file_hashes")
        if not isinstance(receipt_immutable, dict) or set(receipt_immutable) != {
            "config.json",
            "generation_config.json",
            "predict.py",
        }:
            raise ValueError(f"Missing immutable inference hashes in {receipt_path}")
        if immutable_hashes is None:
            immutable_hashes = receipt_immutable
        elif receipt_immutable != immutable_hashes:
            raise ValueError("Config/generation/predict hashes differ between mixes")
        receipts[label] = {
            "path": str(receipt_path.resolve()),
            "sha256": sha256_file(receipt_path),
            "stable_weight": receipt["stable_weight"],
            "encoder_tensor_sha256": digest,
        }
    return {
        "passed": True,
        "raw_and_mix_encoder_tensor_sha256": encoder_digest,
        "immutable_file_hashes": immutable_hashes,
        "receipts": receipts,
    }


def _curve_rows(rows: Iterable[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (item["hours"], item["label"])):
        failures = guardrail_failures(row, config)
        for surface in SURFACES:
            values = row[surface]
            output.append(
                {
                    "label": row["label"],
                    "hours": row["hours"],
                    "surface": surface,
                    "tol2": values["tol2"],
                    "cer": values["cer"],
                    "severe": values["severe"],
                    "validation_proxy": validation_proxy(row, config),
                    "eligible": not failures,
                    "guardrail_failures": ";".join(failures),
                }
            )
    return output


def _csv_text(rows: list[dict[str, Any]]) -> str:
    handle = io.StringIO(newline="")
    fields = (
        "label",
        "hours",
        "surface",
        "tol2",
        "cer",
        "severe",
        "validation_proxy",
        "eligible",
        "guardrail_failures",
    )
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def _baseline_context(
    config: dict[str, Any], preparation: dict[str, Any], stage_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    configured = config.get("baselines") or preparation.get("baselines")
    if isinstance(configured, list) and configured:
        return configured
    legacy = next(
        (row for row in stage_rows if "LEGACY" in row["label"].upper()), None
    )
    return [
        {
            "label": "BASELINE_35H",
            "checkpoint": config.get("start", {}).get("checkpoint"),
            "weight_sha256": config.get("start", {}).get("weight_sha256"),
            "actual_wenet_hours": config.get("start", {}).get("actual_wenet_hours"),
            "ranking_role": "historical_context_only",
        },
        (
            {
                "label": "LEGACY40",
                "checkpoint": legacy["checkpoint"],
                "weight_sha256": legacy["weight_sha256"],
                "ranking_role": "historical_context_only",
            }
            if legacy
            else {
                "label": "LEGACY40",
                "ranking_role": "historical_context_only",
                "evidence_status": "reference_not_present_in_stage_root",
            }
        ),
    ]


def _report_text(matrix: dict[str, Any]) -> str:
    lines = [
        "# W500 adaptive continuation and Decoder interpolation",
        "",
        "Checkpoint ranking uses fixed official validation only. Public and OOD are catastrophic vetoes and never rerank candidates.",
        "",
        "## Acoustic winner",
        "",
        f"- `{matrix['raw_acoustic_winner']['label']}` at {matrix['raw_acoustic_winner']['hours']:.6f} W500 hours",
        f"- Model SHA-256: `{matrix['raw_acoustic_winner']['weight_sha256']}`",
        "",
        "## Raw and interpolation candidates",
        "",
        "| fixed rank | package label | validation proxy | CER | tol2 | Public/OOD veto |",
        "|---:|---|---:|---:|---:|---|",
    ]
    ranks = {row["label"]: index for index, row in enumerate(matrix["final_ranking"], 1)}
    for row in matrix["package_candidates"]:
        lines.append(
            f"| {ranks[row['label']]} | {row['label']} | {row['validation_proxy']:.6f} | "
            f"{row['validation']['cer']:.6f} | {row['validation']['tol2']:.6f} | PASS |"
        )
    lines.extend(
        [
            "",
            "## Evidence gates",
            "",
            f"- Stage checkpoint curves: {len(matrix['all_checkpoint_curves'])} candidates.",
            "- Required outputs_pre files and exact 702/1900/2000 prediction/token row counts: PASS.",
            "- Decoder interpolation alpha endpoints, sampled formulas, and identical Encoder digests: PASS.",
            "- Automatic platform upload: disabled.",
            "",
        ]
    )
    return "\n".join(lines)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _write_atomic_new(path: Path, data: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite finalizer output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    outputs = (args.matrix, args.curves, args.tasks, args.report)
    sums_path = args.matrix.parent / "SHA256SUMS"
    complete_path = args.matrix.parent / "EXPERIMENT_COMPLETE"
    for path in (*outputs, sums_path, complete_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite finalizer output: {path}")

    config = read_json(args.config)
    preparation = read_json(args.preparation)
    if preparation.get("passed") is not True:
        raise ValueError("Preparation receipt is absent or did not pass")
    if config.get("selection", {}).get("ranking_surface") != "fixed_validation_only":
        raise ValueError("Registry does not require fixed-validation-only ranking")

    stage_rows, selection_receipts = load_stage_rows(args.stage_root)
    if len(stage_rows) < 4:
        raise ValueError("At least four adaptive checkpoint curve rows are required")
    stage_verification: dict[str, Any] = {}
    outputs_cache: dict[str, dict[str, Any]] = {}
    for row in stage_rows:
        model_audit = verify_model(row)
        outputs_pre = _outputs_pre_for(row)
        cache_key = str(outputs_pre.resolve())
        if cache_key not in outputs_cache:
            outputs_cache[cache_key] = validate_outputs_pre(outputs_pre)
        output_audit = outputs_cache[cache_key]
        row["outputs_pre"] = str(outputs_pre.resolve())
        stage_verification[row["label"]] = {
            "model": model_audit,
            "evaluation": output_audit,
        }
    acoustic_ranking = rank_rows(stage_rows, config)
    if not acoustic_ranking:
        raise ValueError("No eligible acoustic checkpoint remains after Public/OOD vetoes")
    raw_acoustic_winner = acoustic_ranking[0]

    final_rows = load_interpolation_rows(args.interpolation_root)
    final_verification: dict[str, Any] = {}
    final_failures: dict[str, list[str]] = {}
    for label in PACKAGE_LABELS:
        row = final_rows[label]
        model_audit = verify_model(row)
        outputs_pre = _outputs_pre_for(row)
        cache_key = str(outputs_pre.resolve())
        if cache_key not in outputs_cache:
            outputs_cache[cache_key] = validate_outputs_pre(outputs_pre)
        output_audit = outputs_cache[cache_key]
        row["outputs_pre"] = str(outputs_pre.resolve())
        failures = guardrail_failures(row, config)
        final_failures[label] = failures
        final_verification[label] = {
            "model": model_audit,
            "evaluation": output_audit,
            "guardrail_failures": failures,
            "eligible_for_packaging": not failures,
        }

    interpolation_audit = _verify_interpolation_receipts(
        final_rows, raw_acoustic_winner["weight_sha256"]
    )
    final_ranking = rank_rows(final_rows.values(), config)
    if not final_ranking:
        raise ValueError("No final candidate passes packaging guardrails")

    package_rows: list[dict[str, Any]] = []
    fixed_ranks = {row["label"]: index for index, row in enumerate(final_ranking, 1)}
    for order, label in enumerate(PACKAGE_LABELS, start=1):
        row = final_rows[label]
        if final_failures[label]:
            continue
        package_rows.append(
            {
                "label": label,
                "package_order": order,
                "fixed_validation_rank": fixed_ranks[label],
                "model_dir": row["checkpoint"],
                "processor_dir": row["checkpoint"],
                "weight_sha256": row["weight_sha256"],
                "outputs_pre": row["outputs_pre"],
                "validation": row["validation"],
                "public": row["public"],
                "ood": row["ood"],
                "validation_proxy": validation_proxy(row, config),
                "public_ood_veto_passed": True,
                "automatic_platform_upload": False,
            }
        )

    curves = _curve_rows(stage_rows, config)
    matrix = {
        "experiment": config.get("experiment", "W500 adaptive continuation"),
        "ranking_surface": "fixed_validation_only",
        "public_ood_used_for_ranking": False,
        "public_ood_role": "catastrophic_veto_only",
        "preparation": {
            "path": str(args.preparation.resolve()),
            "sha256": sha256_file(args.preparation),
            "passed": True,
        },
        "stage_selections": selection_receipts,
        "historical_baselines": _baseline_context(config, preparation, stage_rows),
        "all_checkpoint_curves": [
            {
                "label": row["label"],
                "hours": row["hours"],
                "weight_sha256": row["weight_sha256"],
                "validation": row["validation"],
                "public": row["public"],
                "ood": row["ood"],
                "validation_proxy": validation_proxy(row, config),
                "guardrail_failures": guardrail_failures(row, config),
            }
            for row in sorted(stage_rows, key=lambda item: (item["hours"], item["label"]))
        ],
        "raw_acoustic_winner": {
            "label": raw_acoustic_winner["label"],
            "hours": raw_acoustic_winner["hours"],
            "checkpoint": raw_acoustic_winner["checkpoint"],
            "weight_sha256": raw_acoustic_winner["weight_sha256"],
            "validation_proxy": validation_proxy(raw_acoustic_winner, config),
        },
        "final_ranking": [
            {
                "label": row["label"],
                "validation_proxy": validation_proxy(row, config),
                "validation": row["validation"],
                "guardrail_failures": [],
            }
            for row in final_ranking
        ],
        "all_final_candidates": [
            {
                "label": label,
                "validation": final_rows[label]["validation"],
                "public": final_rows[label]["public"],
                "ood": final_rows[label]["ood"],
                "validation_proxy": validation_proxy(final_rows[label], config),
                "guardrail_failures": final_failures[label],
                "eligible_for_packaging": not final_failures[label],
            }
            for label in PACKAGE_LABELS
        ],
        "package_candidates": package_rows,
        "verification": {
            "stage": stage_verification,
            "final": final_verification,
            "interpolation": interpolation_audit,
            "required_outputs_pre_files": list(REQUIRED_EVALUATION_FILES),
            "expected_rows": EXPECTED_ROWS,
        },
        "automatic_platform_upload": False,
    }
    tasks = {
        "ranking_surface": "fixed_validation_only",
        "public_ood_used_for_ranking": False,
        "automatic_platform_upload": False,
        "tasks": package_rows,
    }
    report = _report_text(matrix)
    payloads = {
        args.matrix: _json_text(matrix),
        args.curves: _csv_text(curves),
        args.tasks: _json_text(tasks),
        args.report: report,
    }
    for path, data in payloads.items():
        _write_atomic_new(path, data)

    sums = "".join(f"{sha256_file(path)}  {path.resolve()}\n" for path in outputs)
    _write_atomic_new(sums_path, sums)
    completion = {
        "passed": True,
        "ranking_surface": "fixed_validation_only",
        "public_ood_used_for_ranking": False,
        "package_candidate_count": len(package_rows),
        "artifacts": {str(path.resolve()): sha256_file(path) for path in outputs},
        "sha256sums": str(sums_path.resolve()),
        "automatic_platform_upload": False,
    }
    _write_atomic_new(complete_path, _json_text(completion))
    return completion


def main() -> None:
    args = parse_args()
    try:
        completion = finalize(args)
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(completion, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
