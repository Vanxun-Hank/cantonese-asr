#!/usr/bin/env python3
"""Select a checkpoint using fixed internal-validation guardrails and tie-breaks."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metrics-jsonl",
        type=Path,
        help="Training metrics stream (defaults to RUN_DIR/metrics.jsonl)",
    )
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        help="Optional zero-shot metrics retained in the selection report",
    )
    parser.add_argument("--max-cer", type=float, default=0.1163)
    parser.add_argument("--min-sentence-accuracy", type=float, default=0.8219)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_step(name_or_path: str) -> int | None:
    match = CHECKPOINT_PATTERN.search(Path(name_or_path).name)
    return int(match.group(1)) if match else None


def best_model_step(run_dir: Path) -> int | None:
    state_path = run_dir / "trainer_state.json"
    if not state_path.is_file():
        return None
    best_path = read_json(state_path).get("best_model_checkpoint")
    return checkpoint_step(str(best_path)) if best_path else None


def validation_losses(path: Path) -> dict[int, float]:
    """Map checkpoint global step to the true validation loss logged in training."""
    losses: dict[int, float] = {}
    if not path.is_file():
        return losses
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "eval_validation_loss" not in row or "global_step" not in row:
                continue
            step = int(row["global_step"])
            loss = float(row["eval_validation_loss"])
            if not math.isfinite(loss):
                continue
            previous = losses.get(step)
            if previous is not None and not math.isclose(
                previous, loss, rel_tol=1e-9, abs_tol=1e-12
            ):
                raise ValueError(
                    f"Conflicting validation losses for step {step} at {path}:{line_no}: "
                    f"{previous} vs {loss}"
                )
            losses[step] = loss
    return losses


def candidate_step(name: str, resolved_best_step: int | None) -> int | None:
    if name == "best_model":
        return resolved_best_step
    return checkpoint_step(name)


def build_candidates(
    run_dir: Path,
    losses: dict[int, float],
    max_cer: float,
    min_sentence_accuracy: float,
) -> list[dict[str, Any]]:
    resolved_best_step = best_model_step(run_dir)
    candidates_by_step: dict[int, dict[str, Any]] = {}
    unresolved: list[dict[str, Any]] = []

    for path in sorted(run_dir.glob("diagnostics/*/validation/metrics.json")):
        metrics = read_json(path)
        name = path.parents[1].name
        model_dir = run_dir / name
        if not model_dir.is_dir():
            continue
        step = candidate_step(name, resolved_best_step)
        accuracy = float(metrics["sentence_accuracy_tol2"])
        cer = float(metrics["cer"])
        validation_loss = losses.get(step) if step is not None else None
        rejection_reasons = []
        if step is None:
            rejection_reasons.append("checkpoint_step_unresolved")
        if validation_loss is None:
            rejection_reasons.append("validation_loss_missing")
        if accuracy < min_sentence_accuracy:
            rejection_reasons.append("sentence_accuracy_below_guardrail")
        if cer > max_cer:
            rejection_reasons.append("cer_above_guardrail")
        row = {
            "checkpoint": name,
            "checkpoint_step": step,
            "model_dir": str(model_dir),
            "sentence_accuracy_tol2": accuracy,
            "cer": cer,
            "validation_loss": validation_loss,
            "eligible": not rejection_reasons,
            "rejection_reasons": rejection_reasons,
        }
        if step is None:
            unresolved.append(row)
            continue
        previous = candidates_by_step.get(step)
        # A real checkpoint directory is the canonical name when best_model is a copy.
        if previous is None or (
            previous["checkpoint"] == "best_model" and name.startswith("checkpoint-")
        ):
            candidates_by_step[step] = row
    return list(candidates_by_step.values()) + unresolved


def rank_key(row: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        float(row["sentence_accuracy_tol2"]),
        -float(row["cer"]),
        -float(row["validation_loss"]),
        -int(row["checkpoint_step"]),
    )


def main() -> None:
    args = parse_args()
    if not (0 <= args.max_cer <= 1 and 0 <= args.min_sentence_accuracy <= 1):
        raise SystemExit("Guardrails must be within [0, 1]")
    metrics_path = args.metrics_jsonl or args.run_dir / "metrics.jsonl"
    losses = validation_losses(metrics_path)
    candidates = build_candidates(
        args.run_dir,
        losses,
        max_cer=args.max_cer,
        min_sentence_accuracy=args.min_sentence_accuracy,
    )
    eligible = [row for row in candidates if row["eligible"]]
    selected = max(eligible, key=rank_key) if eligible else None
    report: dict[str, Any] = {
        "selection_surface": "internal_validation_only",
        "guardrails": {
            "max_cer": args.max_cer,
            "min_sentence_accuracy_tol2": args.min_sentence_accuracy,
        },
        "tie_break_order": [
            "sentence_accuracy_tol2_desc",
            "cer_asc",
            "validation_loss_asc",
            "checkpoint_step_asc",
        ],
        "metrics_jsonl": str(metrics_path),
        "selected": selected,
        "candidates": sorted(
            candidates,
            key=lambda row: (
                not row["eligible"],
                -row["sentence_accuracy_tol2"],
                row["cer"],
                row["validation_loss"]
                if row["validation_loss"] is not None
                else math.inf,
                row["checkpoint_step"]
                if row["checkpoint_step"] is not None
                else math.inf,
            ),
        ),
    }
    if args.baseline_metrics:
        report["zero_shot_baseline"] = read_json(args.baseline_metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if selected is None:
        raise SystemExit(
            "No independently evaluated checkpoint passes both fixed validation guardrails"
        )


if __name__ == "__main__":
    main()
