#!/usr/bin/env python3
"""Offline Whisper inference entry compatible with the official protocol."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--test_list", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--processor_dir", type=Path)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_beams", type=int, default=1)
    return parser.parse_args()


def read_test_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not str(row.get("audio_path", "")).strip():
                    raise ValueError(f"Missing audio_path at {path}:{line_no}")
                rows.append(row)
        return rows

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Empty test list: {path}")
        audio_key = next(
            (
                key
                for key in reader.fieldnames
                if key.strip().lower() in {"audio_path", "audio:file"}
            ),
            None,
        )
        if audio_key is None:
            raise ValueError(f"No audio_path column in {path}: {reader.fieldnames}")
        rows = []
        for line_no, row in enumerate(reader, start=2):
            audio_path = str(row.get(audio_key, "")).strip()
            if not audio_path:
                raise ValueError(f"Missing audio_path at {path}:{line_no}")
            rows.append({**row, "audio_path": audio_path})
        return rows


def resolve_audio(audio_dir: Path, listed_path: str) -> Path:
    candidate = Path(listed_path)
    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    candidates.extend((audio_dir / candidate, audio_dir / candidate.name))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Audio not found for {listed_path!r}: {candidates}")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_beams < 1:
        raise SystemExit("--batch_size and --num_beams must be positive")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(
        args.processor_dir or args.model_dir, local_files_only=True
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=dtype,
        use_safetensors=True,
    ).to(device)
    model.generation_config.language = "zh"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.eval()

    rows = read_test_rows(args.test_list)
    listed_paths = [str(row["audio_path"]).strip() for row in rows]
    duplicates = [path for path in dict.fromkeys(listed_paths) if listed_paths.count(path) > 1]
    if duplicates:
        raise ValueError(f"Duplicate audio_path in test list: {duplicates[:5]}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for start in range(0, len(listed_paths), args.batch_size):
            batch_paths = listed_paths[start : start + args.batch_size]
            audio_arrays = [
                librosa.load(resolve_audio(args.audio_dir, item), sr=16_000, mono=True)[0]
                for item in batch_paths
            ]
            processed = processor.feature_extractor(
                audio_arrays,
                sampling_rate=16_000,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_features = processed.input_features.to(device=device, dtype=dtype)
            attention_mask = processed.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            with torch.inference_mode():
                predicted_ids = model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    language="zh",
                    task="transcribe",
                    num_beams=args.num_beams,
                )
            texts = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            for listed_path, text in zip(batch_paths, texts):
                output.write(
                    json.dumps(
                        {"audio_path": listed_path, "pred_text": text.strip()},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    temporary.replace(args.output_jsonl)


if __name__ == "__main__":
    main()
