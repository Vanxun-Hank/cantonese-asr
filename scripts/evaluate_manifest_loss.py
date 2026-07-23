#!/usr/bin/env python3
"""Compute token-weighted teacher-forced loss for one immutable manifest."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from cantonese_asr.io import read_jsonl, sha256_file
from train import ManifestDataset, SpeechSeq2SeqCollator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto"
    )
    return parser.parse_args()


def weighted_mean_loss(weighted_losses: Iterable[tuple[float, int]]) -> tuple[float, int]:
    loss_sum = 0.0
    token_count = 0
    for loss, count in weighted_losses:
        if not math.isfinite(loss) or count < 0:
            raise ValueError(f"Invalid loss/count pair: {(loss, count)}")
        loss_sum += float(loss) * int(count)
        token_count += int(count)
    if token_count == 0:
        raise ValueError("Cannot compute loss with zero target tokens")
    return loss_sum / token_count, token_count


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 evaluation requires CUDA")
    return dtype


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise SystemExit("--batch-size must be positive and --num-workers non-negative")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype, device)
    processor = WhisperProcessor.from_pretrained(args.processor_dir, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=dtype,
        use_safetensors=True,
    ).to(device)
    model.eval()
    dataset = ManifestDataset(args.manifest, processor, args.project_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=SpeechSeq2SeqCollator(processor),
    )
    started = time.monotonic()
    weighted: list[tuple[float, int]] = []
    with torch.inference_mode():
        for batch in loader:
            labels = batch["labels"].to(device)
            target_tokens = int(labels.ne(-100).sum().item())
            model_inputs = {
                "input_features": batch["input_features"].to(device=device, dtype=dtype),
                "attention_mask": batch.get("attention_mask"),
                "labels": labels,
            }
            if model_inputs["attention_mask"] is not None:
                model_inputs["attention_mask"] = model_inputs["attention_mask"].to(device)
            outputs = model(**model_inputs)
            weighted.append((float(outputs.loss.item()), target_tokens))
    mean_loss, token_count = weighted_mean_loss(weighted)
    rows = read_jsonl(args.manifest)
    report = {
        "metric": "teacher_forced_token_cross_entropy",
        "loss": mean_loss,
        "target_tokens": token_count,
        "samples": len(rows),
        "model_dir": str(args.model_dir.resolve()),
        "processor_dir": str(args.processor_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "dtype": str(dtype).removeprefix("torch."),
        "device": str(device),
        "elapsed_seconds": time.monotonic() - started,
    }
    if torch.cuda.is_available():
        report["gpu_peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
