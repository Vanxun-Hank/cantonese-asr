#!/usr/bin/env python3
"""Materialize per-epoch validation/train-probe predictions and error reports."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def named_manifest(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "diagnostic manifest must use NAME=/path/to/manifest.jsonl"
        )
    name, path = value.split("=", 1)
    name = name.strip()
    if not name or not name.replace("_", "").replace("-", "").isalnum():
        raise argparse.ArgumentTypeError(
            f"invalid diagnostic split name: {name!r}"
        )
    if name in {"validation", "train_probe", "ood_panel"}:
        raise argparse.ArgumentTypeError(f"reserved diagnostic split name: {name}")
    if not path.strip():
        raise argparse.ArgumentTypeError("diagnostic manifest path is empty")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--train-probe-manifest", type=Path, required=True)
    parser.add_argument("--ood-panel-manifest", type=Path)
    parser.add_argument(
        "--diagnostic-manifest",
        type=named_manifest,
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Additional diagnostic-only split; may be repeated.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--generation-max-length", type=int, default=225)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    if path.name == "best_model":
        return 10**18
    return int(path.name.rsplit("-", 1)[-1])


def resolve_best_checkpoint_name(run_dir: Path) -> str | None:
    state_path = run_dir / "trainer_state.json"
    if not state_path.is_file():
        return None
    state = json.loads(state_path.read_text(encoding="utf-8"))
    value = state.get("best_model_checkpoint")
    return Path(str(value)).name if value else None


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def prediction_command(
    python: Path,
    checkpoint: Path,
    processor_dir: Path,
    audio_dir: Path,
    manifest: Path,
    output: Path,
    batch_size: int,
    generation_max_length: int,
) -> list[str]:
    return [
        str(python),
        "predict.py",
        "--model_dir",
        str(checkpoint),
        "--processor_dir",
        str(processor_dir),
        "--audio_dir",
        str(audio_dir),
        "--test_list",
        str(manifest),
        "--output_jsonl",
        str(output),
        "--batch_size",
        str(batch_size),
        "--generation-max-length",
        str(generation_max_length),
    ]


def main() -> None:
    args = parse_args()
    checkpoints = sorted(
        [path for path in args.run_dir.glob("checkpoint-*") if path.is_dir()],
        key=checkpoint_step,
    )
    best_model = args.run_dir / "best_model"
    existing_names = {path.name for path in checkpoints}
    resolved_best = resolve_best_checkpoint_name(args.run_dir)
    if best_model.is_dir() and resolved_best not in existing_names:
        checkpoints.append(best_model)
    if not checkpoints:
        raise SystemExit(f"No checkpoint-* or best_model below {args.run_dir}")

    generated = []
    for checkpoint in checkpoints:
        checkpoint_name = checkpoint.name
        splits = [
            ("validation", args.validation_manifest),
            ("train_probe", args.train_probe_manifest),
        ]
        if args.ood_panel_manifest:
            splits.append(("ood_panel", args.ood_panel_manifest))
        splits.extend(args.diagnostic_manifest)
        for split, manifest in splits:
            report_dir = args.run_dir / "diagnostics" / checkpoint_name / split
            prediction_path = report_dir / "predictions.jsonl"
            loss_path = report_dir / "teacher_forced_loss.json"
            report_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = report_dir / "metrics.json"
            if not args.skip_existing or not (prediction_path.is_file() and metrics_path.is_file()):
                run(
                    prediction_command(
                        args.python,
                        checkpoint,
                        args.run_dir,
                        args.audio_dir,
                        manifest,
                        prediction_path,
                        args.batch_size,
                        args.generation_max_length,
                    )
                )
                run(
                    [
                        str(args.python),
                        "scripts/evaluate_predictions.py",
                        "--pred-jsonl",
                        str(prediction_path),
                        "--reference",
                        str(manifest),
                        "--reference-field",
                        "text",
                        "--report-dir",
                        str(report_dir),
                    ]
                    + (["--write-symmetric-t2s"] if split == "ood_panel" else [])
                )
            if not args.skip_existing or not loss_path.is_file():
                run(
                    [
                        str(args.python),
                        "scripts/evaluate_manifest_loss.py",
                        "--model-dir",
                        str(checkpoint),
                        "--processor-dir",
                        str(args.run_dir),
                        "--manifest",
                        str(manifest),
                        "--project-root",
                        str(PROJECT_ROOT),
                        "--output",
                        str(loss_path),
                        "--batch-size",
                        str(args.batch_size),
                        "--num-workers",
                        str(args.num_workers),
                    ]
                )
            generated.append(
                {
                    "checkpoint": checkpoint_name,
                    "split": split,
                    "report_dir": str(report_dir),
                    "generation_max_length": args.generation_max_length,
                    "teacher_forced_loss": str(loss_path),
                }
            )
    (args.run_dir / "diagnostics" / "index.json").write_text(
        json.dumps(generated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
