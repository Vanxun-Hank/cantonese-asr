#!/usr/bin/env python3
"""Build and validate a flat, single-weight offline submission ZIP."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cantonese_asr.io import sha256_file


WEIGHT_NAMES = {
    "model.safetensors",
    "pytorch_model.bin",
    "model.pt",
    "model.pth",
    "model.ckpt",
}
REQUIRED = {
    "predict.py",
    "config.json",
    "preprocessor_config.json",
    "generation_config.json",
}
TRAINING_ONLY = {
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "training_args.bin",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--predict-py", type=Path, default=Path("predict.py"))
    parser.add_argument("--requirements", type=Path)
    parser.add_argument("--output-zip", type=Path, required=True)
    return parser.parse_args()


def validate_names(names: list[str]) -> None:
    if any("/" in name or "\\" in name for name in names):
        raise ValueError("Submission ZIP must be flat; nested paths found")
    missing = sorted(REQUIRED - set(names))
    if missing:
        raise ValueError(f"Missing required submission files: {missing}")
    weights = [name for name in names if name in WEIGHT_NAMES]
    if len(weights) != 1:
        raise ValueError(f"Expected exactly one ASR weight, found: {weights}")
    if weights[0] != "model.safetensors":
        raise ValueError("Goal 1 packaging requires model.safetensors")


def main() -> None:
    args = parse_args()
    if not args.model_dir.is_dir():
        raise SystemExit(f"Model directory not found: {args.model_dir}")
    if not args.predict_py.is_file():
        raise SystemExit(f"predict.py not found: {args.predict_py}")

    model_files = [
        path
        for path in args.model_dir.iterdir()
        if path.is_file() and path.name not in TRAINING_ONLY
    ]
    weight_files = [path for path in model_files if path.name in WEIGHT_NAMES]
    if len(weight_files) != 1 or weight_files[0].name != "model.safetensors":
        raise SystemExit(
            f"Expected only model.safetensors in model directory; found {[p.name for p in weight_files]}"
        )

    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cantonese-asr-submission-") as temp_name:
        staging = Path(temp_name)
        for source in model_files:
            shutil.copy2(source, staging / source.name)
        shutil.copy2(args.predict_py, staging / "predict.py")
        if args.requirements:
            shutil.copy2(args.requirements, staging / "requirements.txt")

        names = sorted(path.name for path in staging.iterdir() if path.is_file())
        validate_names(names)
        manifest = {
            "files": {
                name: {
                    "size_bytes": (staging / name).stat().st_size,
                    "sha256": sha256_file(staging / name),
                }
                for name in names
            },
            "single_weight": "model.safetensors",
            "offline": True,
        }
        (staging / "submission_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        names.append("submission_manifest.json")
        with zipfile.ZipFile(
            args.output_zip, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        ) as archive:
            for name in sorted(names):
                archive.write(staging / name, arcname=name)

    with zipfile.ZipFile(args.output_zip) as archive:
        archived_names = archive.namelist()
        validate_names(archived_names)
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Corrupt file in submission ZIP: {bad}")
    print(
        json.dumps(
            {
                "submission": str(args.output_zip.resolve()),
                "size_bytes": args.output_zip.stat().st_size,
                "sha256": sha256_file(args.output_zip),
                "files": archived_names,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
