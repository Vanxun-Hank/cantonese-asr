#!/usr/bin/env python3
"""Score every row in an immutable W500 candidate shard.

The adaptive manifest builder needs a per-sample difficulty sidecar.  The
existing ``evaluate_manifest_loss.py`` intentionally reports only one aggregate
loss, so this command keeps that behavior untouched and writes one JSONL row per
candidate instead.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from cantonese_asr.io import read_jsonl, sha256_file
from cantonese_asr.metrics import normalize_reference
from train import ManifestDataset, SpeechSeq2SeqCollator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eligible-manifest-sha256", required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    return parser.parse_args()


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 scoring requires CUDA")
    return dtype


def per_sample_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    character_counts: list[int],
) -> list[dict[str, float | int]]:
    """Return token mean and character-normalized loss for each batch row."""

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError(
            f"Unexpected logits/labels shapes: {tuple(logits.shape)}, {tuple(labels.shape)}"
        )
    if len(character_counts) != labels.shape[0]:
        raise ValueError("character_counts must match the batch size")
    token_losses = F.cross_entropy(
        logits.float().transpose(1, 2),
        labels,
        reduction="none",
        ignore_index=-100,
    )
    valid = labels.ne(-100)
    rows: list[dict[str, float | int]] = []
    for index, characters in enumerate(character_counts):
        token_count = int(valid[index].sum().item())
        if token_count <= 0 or characters <= 0:
            raise ValueError(
                f"Invalid normalization counts at batch row {index}: "
                f"tokens={token_count}, characters={characters}"
            )
        loss_sum = float(token_losses[index][valid[index]].sum().item())
        token_mean = loss_sum / token_count
        normalized = loss_sum / characters
        if not math.isfinite(token_mean) or not math.isfinite(normalized):
            raise ValueError(f"Non-finite sample loss at batch row {index}")
        rows.append(
            {
                "loss_sum": loss_sum,
                "target_tokens": token_count,
                "normalized_characters": characters,
                "token_mean_loss": token_mean,
                "normalized_loss": normalized,
            }
        )
    return rows


def row_text(row: dict[str, Any]) -> str:
    for name in ("text", "sentence", "transcript", "metric_text"):
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    raise ValueError(f"Candidate row has no text: {row.get('id')}")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise SystemExit("--batch-size must be positive and --num-workers non-negative")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype, device)
    candidates = read_jsonl(args.manifest)
    if len({str(row.get("id")) for row in candidates}) != len(candidates):
        raise SystemExit("Candidate shard IDs must be unique")

    processor = WhisperProcessor.from_pretrained(args.processor_dir, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_dir,
        local_files_only=True,
        use_safetensors=True,
        dtype=dtype,
    ).to(device)
    model.eval()
    dataset = ManifestDataset(args.manifest, processor, args.project_root)
    if len(dataset) != len(candidates):
        raise SystemExit("ManifestDataset changed the immutable candidate row count")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=SpeechSeq2SeqCollator(processor),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cursor = 0
    with args.output.open("x", encoding="utf-8") as handle, torch.inference_mode():
        for batch in loader:
            batch_size = int(batch["labels"].shape[0])
            source_rows = candidates[cursor : cursor + batch_size]
            labels = batch["labels"].to(device)
            model_inputs = {
                "input_features": batch["input_features"].to(device=device, dtype=dtype),
                "labels": labels,
            }
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                model_inputs["attention_mask"] = attention_mask.to(device)
            outputs = model(**model_inputs)
            counts = [max(1, len(normalize_reference(row_text(row)))) for row in source_rows]
            losses = per_sample_losses(outputs.logits, labels, counts)
            for source, loss in zip(source_rows, losses):
                record = {
                    "id": str(source["id"]),
                    "eligible_manifest_sha256": args.eligible_manifest_sha256,
                    **loss,
                }
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            cursor += batch_size
    if cursor != len(candidates):
        raise SystemExit(f"Scored {cursor} rows, expected {len(candidates)}")
    receipt = {
        "passed": True,
        "model_dir": str(args.model_dir.resolve()),
        "processor_dir": str(args.processor_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "eligible_manifest_sha256": args.eligible_manifest_sha256,
        "rows": cursor,
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "dtype": str(dtype).removeprefix("torch."),
        "device": str(device),
        "elapsed_seconds": time.monotonic() - started,
    }
    if torch.cuda.is_available():
        receipt["gpu_peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
    args.receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
