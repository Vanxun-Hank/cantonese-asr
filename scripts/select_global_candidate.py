#!/usr/bin/env python3
"""Choose one final model from guarded per-run selections using validation only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.select_best_checkpoint import rank_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_candidates(paths: list[Path]) -> list[dict[str, Any]]:
    candidates = []
    seen_models: set[str] = set()
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("selection_surface") != "internal_validation_only":
            raise ValueError(f"Selection is not validation-only: {path}")
        selected = report.get("selected")
        if not selected or selected.get("eligible") is not True:
            raise ValueError(f"Selection has no eligible checkpoint: {path}")
        model_dir = str(Path(selected["model_dir"]).resolve())
        if model_dir in seen_models:
            continue
        seen_models.add(model_dir)
        candidates.append(
            {
                **selected,
                "model_dir": model_dir,
                "run_dir": str(path.parent.resolve()),
                "selection_report": str(path.resolve()),
            }
        )
    return candidates


def main() -> None:
    args = parse_args()
    candidates = load_candidates(args.selection)
    if not candidates:
        raise SystemExit("No eligible unique candidates")
    selected = max(candidates, key=rank_key)
    report = {
        "selection_surface": "internal_validation_only",
        "ood_used_for_selection": False,
        "tie_break_order": [
            "sentence_accuracy_tol2_desc",
            "cer_asc",
            "validation_loss_asc",
            "checkpoint_step_asc",
        ],
        "selected": selected,
        "candidates": sorted(candidates, key=rank_key, reverse=True),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
