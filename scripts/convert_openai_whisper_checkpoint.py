#!/usr/bin/env python3
"""Convert an original OpenAI Whisper checkpoint using frozen local HF metadata.

The conversion is deliberately offline: tokenizer/config/generation metadata must
already exist in ``--output-dir`` and only the model weights are materialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import GenerationConfig, WhisperConfig, WhisperForConditionalGeneration


WHISPER_MAPPING = {
    "blocks": "layers",
    "mlp.0": "fc1",
    "mlp.2": "fc2",
    "mlp_ln": "final_layer_norm",
    ".attn.query": ".self_attn.q_proj",
    ".attn.key": ".self_attn.k_proj",
    ".attn.value": ".self_attn.v_proj",
    ".attn_ln": ".self_attn_layer_norm",
    ".attn.out": ".self_attn.out_proj",
    ".cross_attn.query": ".encoder_attn.q_proj",
    ".cross_attn.key": ".encoder_attn.k_proj",
    ".cross_attn.value": ".encoder_attn.v_proj",
    ".cross_attn_ln": ".encoder_attn_layer_norm",
    ".cross_attn.out": ".encoder_attn.out_proj",
    "decoder.ln.": "decoder.layer_norm.",
    "encoder.ln.": "encoder.layer_norm.",
    "token_embedding": "embed_tokens",
    "encoder.positional_embedding": "encoder.embed_positions.weight",
    "decoder.positional_embedding": "decoder.embed_positions.weight",
    "ln_post": "layer_norm",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_weight_files(output_dir: Path) -> list[Path]:
    """Return the deterministic HF weight surface, including sharded models."""
    single = output_dir / "model.safetensors"
    if single.is_file():
        return [single]
    index = output_dir / "model.safetensors.index.json"
    if not index.is_file():
        return []
    payload = json.loads(index.read_text(encoding="utf-8"))
    shards = sorted({str(value) for value in payload.get("weight_map", {}).values()})
    files = [index, *(output_dir / shard for shard in shards)]
    if not shards or any(not path.is_file() for path in files):
        raise FileNotFoundError("incomplete sharded safetensors surface")
    return files


def weight_surface_receipt(output_dir: Path) -> dict[str, object]:
    files = model_weight_files(output_dir)
    if not files:
        raise FileNotFoundError(f"no safetensors weights under {output_dir}")
    records = [
        {
            "path": path.name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in files
    ]
    canonical = "\n".join(
        f'{item["path"]}\t{item["sha256"]}\t{item["bytes"]}' for item in records
    )
    return {
        "files": records,
        "surface_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "total_bytes": sum(int(item["bytes"]) for item in records),
        "sharded": len(files) > 1,
    }


def renamed_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key in {"layers", "blocks"}:
            continue
        new_key = key
        for source, target in WHISPER_MAPPING.items():
            new_key = new_key.replace(source, target)
        if new_key in converted:
            raise ValueError(f"duplicate converted key: {new_key}")
        converted[new_key] = value
    return converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument(
        "--receipt-only",
        action="store_true",
        help="write a receipt for an already-converted, complete weight surface",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actual_sha = sha256_file(args.checkpoint)
    if actual_sha != args.expected_sha256:
        raise ValueError(f"OpenAI checkpoint SHA mismatch: {actual_sha}")
    for required in ("config.json", "generation_config.json", "tokenizer.json"):
        if not (args.output_dir / required).is_file():
            raise FileNotFoundError(args.output_dir / required)

    allowed_missing = {
        "encoder.embed_positions.weights",
        "decoder.embed_positions.weights",
    }
    if args.receipt_only:
        missing = sorted(allowed_missing)
        unexpected: list[str] = []
    else:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        raw_state = checkpoint["model_state_dict"]
        projection = raw_state["decoder.token_embedding.weight"]
        state_dict = renamed_state_dict(raw_state)

        config = WhisperConfig.from_pretrained(args.output_dir, local_files_only=True)
        model = WhisperForConditionalGeneration(config)
        missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
        if set(missing) - allowed_missing or unexpected:
            raise ValueError({"missing": missing, "unexpected": unexpected})
        model.proj_out.weight.data.copy_(projection)
        model.generation_config = GenerationConfig.from_pretrained(
            args.output_dir, local_files_only=True
        )
        model.save_pretrained(args.output_dir, safe_serialization=True)
    weight_surface = weight_surface_receipt(args.output_dir)
    receipt = {
        "source": str(args.checkpoint.resolve()),
        "source_sha256": actual_sha,
        "output": str(args.output_dir.resolve()),
        "weight_surface": weight_surface,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "receipt_only": args.receipt_only,
    }
    (args.output_dir / "openai_conversion_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
