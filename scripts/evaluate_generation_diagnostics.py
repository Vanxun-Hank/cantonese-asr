#!/usr/bin/env python3
"""Measure Whisper generation failure modes that CER alone can hide."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import librosa
import torch
from transformers import WhisperProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cantonese_asr.io import read_jsonl
from cantonese_asr.model_loading import load_whisper_model

CANTONESE_COLLOQUIAL_MARKERS = (
    "唔",
    "冇",
    "喺",
    "嘅",
    "啲",
    "咁",
    "咩",
    "喎",
    "㗎",
    "啫",
    "佢",
    "哋",
    "呢个",
    "嗰个",
)
FORMAL_MANDARIN_MARKERS = (
    "没有",
    "不要",
    "这个",
    "那个",
    "什么",
    "怎么",
    "是否",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--audio-dir",
        type=Path,
        help="Optional root used for public/test manifests with relocated audio.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--generation-max-length", type=int, default=225)
    return parser.parse_args()


def resolve_audio(
    project_root: Path,
    value: str,
    audio_dir: Path | None = None,
) -> Path:
    listed = Path(value)
    candidates = [listed] if listed.is_absolute() else [project_root / listed]
    if audio_dir is not None:
        candidates.extend((audio_dir / listed, audio_dir / listed.name))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Audio not found for {value!r}: {candidates}")


def repeated_runaway(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if re.search(r"(.)\1{7,}", compact):
        return True
    for width in range(2, 13):
        for start in range(max(0, len(compact) - 120), len(compact) - width * 3 + 1):
            unit = compact[start : start + width]
            if unit and unit * 4 in compact[start:]:
                return True
    return False


def abnormal_character_count(text: str) -> int:
    return sum(
        character == "\ufffd"
        or (ord(character) < 32 and character not in {"\n", "\t", "\r"})
        for character in text
    )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.generation_max_length < 1:
        raise SystemExit("Batch size and generation length must be positive")
    rows = read_jsonl(args.manifest)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if not rows:
        raise SystemExit("Diagnostic manifest is empty")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(
        args.processor_dir,
        language="zh",
        task="transcribe",
        local_files_only=True,
    )
    model = load_whisper_model(args.model_dir, dtype=dtype).to(device)
    model.generation_config.language = "zh"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.eval()

    eos_ids = model.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_set = {eos_ids}
    else:
        eos_set = {int(item) for item in eos_ids}
    details: list[dict[str, Any]] = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        audio = [
            librosa.load(
                resolve_audio(
                    args.project_root,
                    str(row["audio_path"]),
                    args.audio_dir,
                ),
                sr=16_000,
                mono=True,
            )[0]
            for row in batch
        ]
        processed = processor.feature_extractor(
            audio,
            sampling_rate=16_000,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
        )
        with torch.inference_mode():
            token_ids = model.generate(
                processed.input_features.to(device=device, dtype=dtype),
                attention_mask=processed.attention_mask.to(device),
                language="zh",
                task="transcribe",
                num_beams=1,
                max_length=args.generation_max_length,
            )
        token_rows = token_ids.detach().cpu().tolist()
        texts = processor.batch_decode(token_rows, skip_special_tokens=True)
        for row, ids, text in zip(batch, token_rows, texts):
            cleaned = text.strip()
            actual_ids = list(ids)
            for token_index, token in enumerate(ids[1:], start=1):
                if int(token) in eos_set:
                    actual_ids = ids[: token_index + 1]
                    break
            details.append(
                {
                    "id": str(row.get("id", "")),
                    "audio_path": str(row["audio_path"]),
                    "reference": str(row.get("text", "")),
                    "prediction": cleaned,
                    "token_length": len(actual_ids),
                    "character_length": len(cleaned),
                    "has_eos": bool(actual_ids and actual_ids[-1] in eos_set),
                    "empty": not cleaned,
                    "hit_generation_max_length": len(actual_ids)
                    >= args.generation_max_length,
                    "repeated_runaway": repeated_runaway(cleaned),
                    "abnormal_character_count": abnormal_character_count(cleaned),
                    "contains_latin": bool(re.search(r"[A-Za-z]", cleaned)),
                    "contains_digit": bool(re.search(r"\d", cleaned)),
                    "code_switching": bool(re.search(r"[A-Za-z]", cleaned))
                    and bool(re.search(r"[\u3400-\u9fff]", cleaned)),
                    "contains_cantonese_colloquial": any(
                        marker in cleaned
                        for marker in CANTONESE_COLLOQUIAL_MARKERS
                    ),
                    "contains_formal_mandarin_marker": any(
                        marker in cleaned for marker in FORMAL_MANDARIN_MARKERS
                    ),
                }
            )

    token_lengths = [row["token_length"] for row in details]
    character_lengths = [row["character_length"] for row in details]
    total_characters = sum(character_lengths)
    abnormal_characters = sum(row["abnormal_character_count"] for row in details)
    summary = {
        "num_samples": len(details),
        "generation_max_length": args.generation_max_length,
        "eos_rate": mean(row["has_eos"] for row in details),
        "empty_prediction_rate": mean(row["empty"] for row in details),
        "hit_generation_max_length_count": sum(
            row["hit_generation_max_length"] for row in details
        ),
        "repeated_runaway_count": sum(row["repeated_runaway"] for row in details),
        "abnormal_character_rate": (
            abnormal_characters / total_characters if total_characters else None
        ),
        "mean_output_token_length": mean(token_lengths),
        "max_output_token_length": max(token_lengths),
        "mean_output_character_length": mean(character_lengths),
        "max_output_character_length": max(character_lengths),
        "contains_latin_rate": mean(row["contains_latin"] for row in details),
        "contains_digit_rate": mean(row["contains_digit"] for row in details),
        "code_switching_rate": mean(row["code_switching"] for row in details),
        "cantonese_colloquial_marker_rate": mean(
            row["contains_cantonese_colloquial"] for row in details
        ),
        "formal_mandarin_marker_rate": mean(
            row["contains_formal_mandarin_marker"] for row in details
        ),
    }
    payload = {"summary": summary, "samples": details}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
