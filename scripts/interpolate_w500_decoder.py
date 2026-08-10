#!/usr/bin/env python3
"""Create an immutable, audited W500 Decoder-interpolated checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import load_file, save_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cantonese_asr.w500_decoder_interpolation import (  # noqa: E402
    audit_interpolation,
    interpolate_decoder_state,
)


MIX_WEIGHTS = {"MIX15": 0.15, "MIX30": 0.30, "MIX45": 0.45, "MIX60": 0.60}
PROCESSOR_FILES = (
    "added_tokens.json",
    "merges.txt",
    "normalizer.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
REQUIRED_ACOUSTIC_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
)
ARCHITECTURE_FIELDS = (
    "model_type",
    "vocab_size",
    "num_mel_bins",
    "d_model",
    "encoder_layers",
    "encoder_attention_heads",
    "encoder_ffn_dim",
    "decoder_layers",
    "decoder_attention_heads",
    "decoder_ffn_dim",
    "max_source_positions",
    "max_target_positions",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acoustic-checkpoint", type=Path, required=True)
    parser.add_argument("--stable-decoder-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--name", choices=("MIX15", "MIX30", "MIX45", "MIX60"), required=True
    )
    parser.add_argument("--stable-weight", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/rounds/w500_adaptive_continuation.json",
        help="Approved immutable experiment registry.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root containing the canonical predict.py.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Required JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}: {exc}") from exc


def _expected_hashes(registry: dict[str, Any]) -> dict[str, str]:
    start = registry.get("start", {})
    aliases = {
        "config.json": ("config_sha256",),
        "generation_config.json": ("generation_config_sha256",),
        "predict.py": ("predict_py_sha256", "predict_sha256"),
    }
    expected: dict[str, str] = {}
    for filename, keys in aliases.items():
        for key in keys:
            value = start.get(key)
            if value:
                expected[filename] = str(value)
                break
        if filename not in expected:
            raise ValueError(f"Registry start section is missing the hash for {filename}")
    return expected


def _verify_architecture_configs(acoustic: Path, stable: Path) -> dict[str, Any]:
    acoustic_config = read_json(acoustic / "config.json")
    stable_config = read_json(stable / "config.json")
    if set(acoustic_config) != set(stable_config):
        raise ValueError(
            "Endpoint config key mismatch: "
            f"acoustic_only={sorted(set(acoustic_config) - set(stable_config))[:20]} "
            f"stable_only={sorted(set(stable_config) - set(acoustic_config))[:20]}"
        )
    checked: dict[str, Any] = {}
    for key in ARCHITECTURE_FIELDS:
        if key in acoustic_config:
            if acoustic_config[key] != stable_config[key]:
                raise ValueError(
                    f"Endpoint architecture mismatch for {key}: "
                    f"{acoustic_config[key]!r} != {stable_config[key]!r}"
                )
            checked[key] = acoustic_config[key]
    return {"config_key_count": len(acoustic_config), "architecture_fields": checked}


def _safetensors_metadata(path: Path) -> dict[str, str]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    return dict(metadata or {"format": "pt"})


def _verify_required_inputs(
    acoustic_checkpoint: Path,
    stable_decoder_checkpoint: Path,
    canonical_predict: Path,
) -> None:
    for checkpoint, label in (
        (acoustic_checkpoint, "acoustic"),
        (stable_decoder_checkpoint, "stable Decoder"),
    ):
        if not checkpoint.is_dir():
            raise ValueError(f"{label} checkpoint is not a directory: {checkpoint}")
        if not (checkpoint / "model.safetensors").is_file():
            raise ValueError(f"{label} model.safetensors is missing: {checkpoint}")
        if not (checkpoint / "config.json").is_file():
            raise ValueError(f"{label} config.json is missing: {checkpoint}")
    for filename in REQUIRED_ACOUSTIC_FILES:
        if not (acoustic_checkpoint / filename).is_file():
            raise ValueError(f"Acoustic checkpoint is missing {filename}")
    if not canonical_predict.is_file():
        raise ValueError(f"Canonical predict.py is missing: {canonical_predict}")


def interpolate_checkpoint(
    *,
    acoustic_checkpoint: Path,
    stable_decoder_checkpoint: Path,
    name: str,
    stable_weight: float,
    output_dir: Path,
    registry_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Interpolate, audit, and atomically publish a single candidate."""

    expected_weight = MIX_WEIGHTS.get(name)
    if expected_weight is None:
        raise ValueError(f"Unknown interpolation candidate: {name}")
    if abs(float(stable_weight) - expected_weight) > 1e-12:
        raise ValueError(
            f"{name} requires stable weight {expected_weight}, got {stable_weight}"
        )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite interpolation output: {output_dir}")

    canonical_predict = project_root / "predict.py"
    _verify_required_inputs(
        acoustic_checkpoint, stable_decoder_checkpoint, canonical_predict
    )
    registry = read_json(registry_path)
    expected_hashes = _expected_hashes(registry)
    architecture_audit = _verify_architecture_configs(
        acoustic_checkpoint, stable_decoder_checkpoint
    )

    immutable_sources = {
        "config.json": acoustic_checkpoint / "config.json",
        "generation_config.json": acoustic_checkpoint / "generation_config.json",
        "predict.py": canonical_predict,
    }
    immutable_hashes = {name_: sha256_file(path) for name_, path in immutable_sources.items()}
    mismatches = {
        name_: {"expected": expected_hashes[name_], "actual": actual}
        for name_, actual in immutable_hashes.items()
        if actual != expected_hashes[name_]
    }
    if mismatches:
        raise ValueError(f"Immutable inference-file hash mismatch: {mismatches}")

    acoustic_weight = acoustic_checkpoint / "model.safetensors"
    stable_weight_path = stable_decoder_checkpoint / "model.safetensors"
    acoustic_state = load_file(acoustic_weight, device="cpu")
    stable_state = load_file(stable_weight_path, device="cpu")
    projection_stored = "proj_out.weight" in acoustic_state
    if projection_stored != ("proj_out.weight" in stable_state):
        raise ValueError("Only one endpoint stores proj_out.weight")
    if not projection_stored:
        tied_name = "model.decoder.embed_tokens.weight"
        acoustic_config = read_json(acoustic_checkpoint / "config.json")
        if tied_name not in acoustic_state or not bool(
            acoustic_config.get("tie_word_embeddings", True)
        ):
            raise ValueError(
                "proj_out.weight is absent and cannot be audited through the tied "
                "model.decoder.embed_tokens.weight"
            )
    mixed_state = interpolate_decoder_state(
        acoustic_state, stable_state, stable_weight=stable_weight
    )
    tensor_audit = audit_interpolation(
        acoustic_state, stable_state, mixed_state, stable_weight=stable_weight
    )
    if (
        tensor_audit["digests"]["acoustic_encoder"]
        != tensor_audit["digests"]["mixed_encoder"]
    ):
        raise ValueError("Encoder digest changed during interpolation")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        metadata = _safetensors_metadata(acoustic_weight)
        save_file(mixed_state, staging / "model.safetensors", metadata=metadata)
        for filename in (*REQUIRED_ACOUSTIC_FILES, *PROCESSOR_FILES):
            source = acoustic_checkpoint / filename
            target = staging / filename
            if source.is_file() and not target.exists():
                shutil.copy2(source, target)
        shutil.copy2(canonical_predict, staging / "predict.py")

        copied_asset_hashes = {
            path.name: sha256_file(path)
            for path in staging.iterdir()
            if path.is_file() and path.name != "model.safetensors"
        }
        for filename, actual in copied_asset_hashes.items():
            source = (
                canonical_predict
                if filename == "predict.py"
                else acoustic_checkpoint / filename
            )
            if sha256_file(source) != actual:
                raise ValueError(f"Copied processor/inference file changed: {filename}")
        copied_hashes = {
            name_: sha256_file(staging / name_) for name_ in immutable_sources
        }
        if copied_hashes != immutable_hashes:
            raise ValueError(
                f"Immutable files changed while copying: source={immutable_hashes}, "
                f"copied={copied_hashes}"
            )
        reloaded = load_file(staging / "model.safetensors", device="cpu")
        persisted_audit = audit_interpolation(
            acoustic_state, stable_state, reloaded, stable_weight=stable_weight
        )
        receipt = {
            "passed": True,
            "name": name,
            "formula": "(1 - stable_weight) * acoustic + stable_weight * stable_decoder",
            "alpha": float(stable_weight),
            "stable_weight": float(stable_weight),
            "acoustic_weight": 1.0 - float(stable_weight),
            "output_dir": str(output_dir.resolve()),
            "endpoints": {
                "acoustic_checkpoint": str(acoustic_checkpoint.resolve()),
                "acoustic_model_sha256": sha256_file(acoustic_weight),
                "stable_decoder_checkpoint": str(stable_decoder_checkpoint.resolve()),
                "stable_decoder_model_sha256": sha256_file(stable_weight_path),
            },
            "output_model_sha256": sha256_file(staging / "model.safetensors"),
            "architecture_audit": architecture_audit,
            "projection_audit": {
                "proj_out_stored_as_separate_tensor": projection_stored,
                "effective_projection_tensor": (
                    "proj_out.weight"
                    if projection_stored
                    else "model.decoder.embed_tokens.weight (tied proj_out)"
                ),
            },
            "tensor_audit": tensor_audit,
            "persisted_tensor_audit": persisted_audit,
            "encoder_tensor_sha256": persisted_audit["digests"]["mixed_encoder"],
            "immutable_file_hashes": copied_hashes,
            "copied_asset_hashes": copied_asset_hashes,
            "registry_sha256": sha256_file(registry_path),
            "automatic_platform_upload": False,
        }
        (staging / "interpolation_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.rename(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return receipt


def main() -> None:
    args = parse_args()
    try:
        receipt = interpolate_checkpoint(
            acoustic_checkpoint=args.acoustic_checkpoint,
            stable_decoder_checkpoint=args.stable_decoder_checkpoint,
            name=args.name,
            stable_weight=args.stable_weight,
            output_dir=args.output_dir,
            registry_path=args.config,
            project_root=args.project_root,
        )
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
