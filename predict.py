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


def load_whisper_model(
    model_dir: Path,
    *,
    dtype: torch.dtype | None = None,
) -> WhisperForConditionalGeneration:
    """Load the submitted full Whisper model without repository-local imports."""
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "use_safetensors": True,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    return WhisperForConditionalGeneration.from_pretrained(model_dir, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--test_list", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument(
        "--diagnostics-jsonl",
        type=Path,
        help=(
            "Optional local-only token sidecar for generation diagnostics. "
            "The official output protocol is unchanged when omitted."
        ),
    )
    parser.add_argument("--model_dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--processor_dir", type=Path)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument(
        "--generation-max-length",
        type=int,
        default=225,
        help="Maximum total number of generated tokens (must match training evaluation)",
    )
    return parser.parse_args()


def validate_inference_args(args: argparse.Namespace) -> None:
    """Reject invalid decoding settings before loading any model assets."""
    if args.batch_size < 1 or args.num_beams < 1 or args.generation_max_length < 1:
        raise SystemExit(
            "--batch_size, --num_beams and --generation-max-length must be positive"
        )


def generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Return the single canonical decoding configuration used by offline inference."""
    return {
        "language": "zh",
        "task": "transcribe",
        "num_beams": args.num_beams,
        "max_length": args.generation_max_length,
    }


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
    validate_inference_args(args)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(
        args.processor_dir or args.model_dir, local_files_only=True
    )
    model = load_whisper_model(args.model_dir, dtype=dtype).to(device)
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
    diagnostic_temporary = None
    if args.diagnostics_jsonl is not None:
        args.diagnostics_jsonl.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_temporary = args.diagnostics_jsonl.with_suffix(
            args.diagnostics_jsonl.suffix + ".tmp"
        )
    diagnostic_handle = (
        diagnostic_temporary.open("w", encoding="utf-8")
        if diagnostic_temporary is not None
        else None
    )
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
                    **generation_kwargs(args),
                )
            texts = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            token_rows = predicted_ids.detach().cpu().tolist()
            for listed_path, text, token_ids in zip(
                batch_paths,
                texts,
                token_rows,
            ):
                cleaned = text.strip()
                output.write(
                    json.dumps(
                        {"audio_path": listed_path, "pred_text": cleaned},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if diagnostic_handle is not None:
                    diagnostic_handle.write(
                        json.dumps(
                            {
                                "audio_path": listed_path,
                                "pred_text": cleaned,
                                "token_ids": [int(token) for token in token_ids],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    if diagnostic_handle is not None:
        diagnostic_handle.close()
        assert diagnostic_temporary is not None
        assert args.diagnostics_jsonl is not None
        diagnostic_temporary.replace(args.diagnostics_jsonl)
    temporary.replace(args.output_jsonl)


if __name__ == "__main__":
    main()
