#!/usr/bin/env python3
"""Evaluate one frozen RAW_WINNER decoding arm without modifying model assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cantonese_asr.metrics import (  # noqa: E402
    build_error_analysis,
    compute_diagnostic_metrics,
    compute_official_metrics,
    levenshtein_ops,
    normalize_prediction,
    normalize_reference,
)
from predict import read_test_rows, resolve_audio  # noqa: E402
from scripts.evaluate_predictions import read_references  # noqa: E402


ARM_CONFIGS: dict[str, dict[str, Any]] = {
    "D0_CURRENT": {
        "num_beams": 2, "no_repeat_ngram_size": 4,
        "repetition_penalty": 1.05, "max_length": 225,
        "early_stopping": True,
    },
    "D1_BEAM1": {
        "num_beams": 1, "no_repeat_ngram_size": 4,
        "repetition_penalty": 1.05, "max_length": 225,
        "early_stopping": False,
    },
    "D2_BEAM4": {
        "num_beams": 4, "no_repeat_ngram_size": 4,
        "repetition_penalty": 1.05, "max_length": 225,
        "early_stopping": True,
    },
    "D3_BEAM5": {
        "num_beams": 5, "no_repeat_ngram_size": 4,
        "repetition_penalty": 1.05, "max_length": 225,
        "early_stopping": True,
    },
    "D4_MAX128": {
        "num_beams": 2, "no_repeat_ngram_size": 4,
        "repetition_penalty": 1.05, "max_length": 128,
        "early_stopping": True,
    },
    "D5_RP100": {
        "num_beams": 2, "no_repeat_ngram_size": 4,
        "repetition_penalty": 1.0, "max_length": 225,
        "early_stopping": True,
    },
    "D6_RP110": {
        "num_beams": 2, "no_repeat_ngram_size": 4,
        "repetition_penalty": 1.10, "max_length": 225,
        "early_stopping": True,
    },
    "D7_MBR_B5": {
        "num_beams": 5, "no_repeat_ngram_size": 4,
        "repetition_penalty": 1.05, "max_length": 225,
        "early_stopping": True, "num_return_sequences": 5,
        "selection": "minimum_bayes_risk_character_consensus",
    },
}

KNOWN_REPEAT_IDS = ("06762", "06872")
LONG_CHARACTER_REPEAT = re.compile(r"(.)\1{7,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARM_CONFIGS), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--ood-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--generation-max-length", type=int, default=225)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--mbr-anchor-root",
        type=Path,
        help=(
            "D3_BEAM5 output root used to anchor D7 candidate rank 0. "
            "Required for D7_MBR_B5 so the two arms share the exact first candidate."
        ),
    )
    parser.add_argument(
        "--surfaces",
        nargs="+",
        choices=("validation", "public", "ood"),
        default=("validation", "public", "ood"),
        help="Evaluate only the listed surfaces; the default preserves legacy behavior.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def percentile(values: list[int] | list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def actual_tokens(tokens: list[int], eos_ids: set[int]) -> list[int]:
    for index, token in enumerate(tokens, start=0):
        if int(token) in eos_ids:
            return [int(value) for value in tokens[: index + 1]]
    return [int(value) for value in tokens]


def trailing_token_cycle(tokens: list[int], special_ids: set[int]) -> dict[str, Any]:
    content = [int(token) for token in tokens if int(token) not in special_ids]
    best: dict[str, Any] = {
        "triggered": False,
        "width": None,
        "repetitions": 0,
        "coverage": 0,
    }
    for width in range(1, 5):
        if len(content) < width * 4:
            continue
        unit = content[-width:]
        repetitions = 1
        cursor = len(content) - width * 2
        while cursor >= 0 and content[cursor : cursor + width] == unit:
            repetitions += 1
            cursor -= width
        coverage = repetitions * width
        if repetitions >= 4 and coverage >= 8 and coverage > best["coverage"]:
            best = {
                "triggered": True,
                "width": width,
                "repetitions": repetitions,
                "coverage": coverage,
                "unit": unit,
            }
    return best


def long_text_cycle(text: str) -> dict[str, Any]:
    compact = re.sub(r"\s+", "", text)
    if LONG_CHARACTER_REPEAT.search(compact):
        return {"triggered": True, "kind": "single_character"}
    for width in range(2, 5):
        for start in range(max(0, len(compact) - 160), len(compact) - width * 4 + 1):
            unit = compact[start : start + width]
            if unit and unit * 4 in compact[start:]:
                return {
                    "triggered": True,
                    "kind": "text_ngram",
                    "width": width,
                    "unit": unit,
                }
    return {"triggered": False}


def operation_counts(reference: str, prediction: str) -> dict[str, int]:
    _, operations = levenshtein_ops(
        list(normalize_reference(reference)),
        list(normalize_prediction(prediction)),
    )
    counts = {"substitutions": 0, "deletions": 0, "insertions": 0}
    for source, target in operations:
        if source == "<ins>":
            counts["insertions"] += 1
        elif target == "<del>":
            counts["deletions"] += 1
        elif source != target:
            counts["substitutions"] += 1
    return counts


def generation_kwargs(arm: str, max_length: int | None = None) -> dict[str, Any]:
    config = ARM_CONFIGS[arm]
    effective_max_length = int(config.get("max_length", max_length or 225))
    result: dict[str, Any] = {
        "language": "zh",
        "task": "transcribe",
        "return_timestamps": False,
        "do_sample": False,
        "num_beams": int(config["num_beams"]),
        "no_repeat_ngram_size": int(config["no_repeat_ngram_size"]),
        "repetition_penalty": float(config["repetition_penalty"]),
        "length_penalty": 1.0,
        "max_length": effective_max_length,
    }
    if int(config["num_beams"]) > 1:
        result["early_stopping"] = bool(config["early_stopping"])
    return result


def normalized_character_distance(left: str, right: str) -> float:
    """Return symmetric normalized character edit distance for MBR selection."""
    left_norm = normalize_prediction(left)
    right_norm = normalize_prediction(right)
    distance, _ = levenshtein_ops(list(left_norm), list(right_norm))
    return float(distance) / float(max(len(left_norm), len(right_norm), 1))


def select_mbr_candidate(
    texts: list[str], sequence_scores: list[float]
) -> tuple[int, list[float]]:
    """Select the consensus candidate; model score and rank break exact ties."""
    if not texts or len(texts) != len(sequence_scores):
        raise ValueError("MBR texts and sequence scores must be non-empty and aligned")
    risks: list[float] = []
    for index, text in enumerate(texts):
        distances = [
            normalized_character_distance(text, other)
            for other_index, other in enumerate(texts)
            if other_index != index
        ]
        risks.append(sum(distances) / len(distances) if distances else 0.0)
    winner = min(
        range(len(texts)),
        key=lambda index: (risks[index], -sequence_scores[index], index),
    )
    return winner, risks


def anchor_sequence_score(
    anchor_text: str,
    candidate_texts: list[str],
    candidate_scores: list[float],
) -> float:
    """Reuse the best returned beam score matching the D3 anchor text.

    Cross-device beam ties can change the raw top string.  When no returned
    candidate normalizes to the anchor, the top returned beam score is the
    deterministic conservative proxy used only for exact MBR-risk ties.
    """
    if not candidate_texts or len(candidate_texts) != len(candidate_scores):
        raise ValueError("Candidate texts and scores must be non-empty and aligned")
    anchor = normalize_prediction(anchor_text)
    matches = [
        float(score)
        for text, score in zip(candidate_texts, candidate_scores)
        if normalize_prediction(text) == anchor
    ]
    return max(matches) if matches else float(candidate_scores[0])


def resolve_surface_audio(project_root: Path, listed_path: str) -> Path:
    try:
        return resolve_audio(project_root, listed_path)
    except FileNotFoundError:
        name = Path(listed_path).name
        candidates = (
            project_root / "artifacts" / "data" / "train_raw" / "4856-10393" / name,
            project_root / "artifacts" / "data" / "test_audio" / name,
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise


def evaluate_surface(
    *,
    surface: str,
    manifest: Path,
    reference_field: str,
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    arm: str,
    project_root: Path,
    output_dir: Path,
    batch_size: int,
    generation_max_length: int,
    prompt_token_count: int,
    device: torch.device,
    dtype: torch.dtype,
    max_samples: int | None,
    mbr_anchor_dir: Path | None,
) -> dict[str, Any]:
    test_rows = read_test_rows(manifest)
    references = read_references(manifest, reference_field)
    if max_samples is not None:
        test_rows = test_rows[:max_samples]
        references = references[:max_samples]
    listed_paths = [str(row["audio_path"]).strip() for row in test_rows]
    reference_paths = [str(row["audio_path"]).strip() for row in references]
    if listed_paths != reference_paths:
        raise ValueError(f"{surface}: reference order differs from inference order")

    eos_config = model.generation_config.eos_token_id
    eos_values = [eos_config] if isinstance(eos_config, int) else list(eos_config or [])
    if not eos_values and processor.tokenizer.eos_token_id is not None:
        eos_values = [processor.tokenizer.eos_token_id]
    eos_ids = {int(value) for value in eos_values}
    special_ids = {int(value) for value in processor.tokenizer.all_special_ids}

    predictions: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    raw_references = [str(row[reference_field]) for row in references]

    anchor_predictions: list[dict[str, Any]] | None = None
    anchor_tokens: list[dict[str, Any]] | None = None
    if arm == "D7_MBR_B5":
        if mbr_anchor_dir is None:
            raise ValueError("D7_MBR_B5 requires a D3_BEAM5 anchor directory")
        anchor_predictions = [
            json.loads(line)
            for line in (mbr_anchor_dir / "predictions.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        anchor_tokens = [
            json.loads(line)
            for line in (mbr_anchor_dir / "generation_tokens.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        if max_samples is not None:
            anchor_predictions = anchor_predictions[:max_samples]
            anchor_tokens = anchor_tokens[:max_samples]
        anchor_paths = [str(row["audio_path"]) for row in anchor_predictions]
        token_paths = [str(row["audio_path"]) for row in anchor_tokens]
        if anchor_paths != listed_paths or token_paths != listed_paths:
            raise ValueError(
                f"{surface}: D3 anchor order/length differs from D7 input order"
            )

    surface_started = time.perf_counter()
    inference_seconds: list[float] = []
    for start in range(0, len(listed_paths), batch_size):
        batch_paths = listed_paths[start : start + batch_size]
        audio = [
            librosa.load(
                resolve_surface_audio(project_root, path),
                sr=16_000,
                mono=True,
            )[0]
            for path in batch_paths
        ]
        processed = processor.feature_extractor(
            audio,
            sampling_rate=16_000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        attention_mask = processed.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        generate_kwargs = generation_kwargs(arm, generation_max_length)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_started = time.perf_counter()
        with torch.inference_mode():
            if arm == "D7_MBR_B5":
                generated = model.generate(
                    processed.input_features.to(device=device, dtype=dtype),
                    attention_mask=attention_mask,
                    num_return_sequences=5,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **generate_kwargs,
                )
                all_ids = generated.sequences.detach().cpu().tolist()
                all_texts = processor.batch_decode(
                    all_ids, skip_special_tokens=True
                )
                if generated.sequences_scores is None:
                    raise RuntimeError("Beam MBR requires sequences_scores")
                all_scores = generated.sequences_scores.detach().float().cpu().tolist()
                selected = []
                for item_index in range(len(batch_paths)):
                    begin = item_index * 5
                    absolute_index = start + item_index
                    assert anchor_predictions is not None
                    assert anchor_tokens is not None
                    # Candidate rank 0 is the exact D3 single-output beam5 result.
                    # The remaining four candidates come from the five-return beam
                    # search.  This makes the D3/D7 comparison invariant to tiny
                    # cross-device beam-search tie differences.
                    candidate_ids = [
                        [int(value) for value in anchor_tokens[absolute_index]["token_ids"]]
                    ] + all_ids[begin : begin + 4]
                    anchor_text = str(anchor_predictions[absolute_index]["pred_text"])
                    returned_texts = all_texts[begin : begin + 5]
                    returned_scores = [
                        float(value) for value in all_scores[begin : begin + 5]
                    ]
                    candidate_texts = [anchor_text] + returned_texts[:4]
                    candidate_scores = [
                        anchor_sequence_score(
                            anchor_text, returned_texts, returned_scores
                        )
                    ] + returned_scores[:4]
                    winner, risks = select_mbr_candidate(
                        candidate_texts, candidate_scores
                    )
                    selected.append(
                        {
                            "raw_text": candidate_texts[winner],
                            "raw_ids": candidate_ids[winner],
                            "selected_index": winner,
                            "candidates": [
                                {
                                    "rank": rank,
                                    "text": candidate_texts[rank].strip(),
                                    "sequence_score": candidate_scores[rank],
                                    "mbr_risk": risks[rank],
                                }
                                for rank in range(5)
                            ],
                        }
                    )
            else:
                predicted_ids = model.generate(
                    processed.input_features.to(device=device, dtype=dtype),
                    attention_mask=attention_mask,
                    **generate_kwargs,
                )
                raw_token_rows = predicted_ids.detach().cpu().tolist()
                raw_sequence_scores = [None] * len(raw_token_rows)
                texts = processor.batch_decode(
                    raw_token_rows, skip_special_tokens=True
                )
                selected = [
                    {
                        "raw_text": raw_text,
                        "raw_ids": raw_ids,
                        "selected_index": 0,
                        "candidates": None,
                        "sequence_score": sequence_score,
                    }
                    for raw_text, raw_ids, sequence_score in zip(
                        texts, raw_token_rows, raw_sequence_scores
                    )
                ]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        batch_inference_seconds = time.perf_counter() - inference_started
        inference_seconds.extend(
            [batch_inference_seconds / len(batch_paths)] * len(batch_paths)
        )

        for offset, (path, selection) in enumerate(zip(batch_paths, selected)):
            raw_text = str(selection["raw_text"])
            raw_ids = list(selection["raw_ids"])
            text = raw_text.strip()
            tokens = actual_tokens([int(value) for value in raw_ids], eos_ids)
            has_eos = bool(tokens and tokens[-1] in eos_ids)
            token_cycle = trailing_token_cycle(tokens, special_ids)
            text_cycle = long_text_cycle(text)
            replacement_count = text.count("\ufffd")
            effective_max = (
                len(tokens) + prompt_token_count >= generation_max_length
            )
            runaway_reasons: list[str] = []
            if not has_eos:
                runaway_reasons.append("no_eos")
            if token_cycle["triggered"]:
                runaway_reasons.append("token_cycle_1_to_4")
            if text_cycle["triggered"]:
                runaway_reasons.append("long_text_repeat")
            if replacement_count >= 3:
                runaway_reasons.append("many_replacement_characters")
            if effective_max:
                runaway_reasons.append("effective_max_length")
            repeated_runaway = bool(
                token_cycle["triggered"]
                or text_cycle["triggered"]
                or replacement_count >= 3
            )

            reference = raw_references[start + offset]
            normalized_reference = normalize_reference(reference)
            normalized_prediction = normalize_prediction(text)
            distance, _ = levenshtein_ops(
                list(normalized_reference),
                list(normalized_prediction),
            )
            operation = operation_counts(reference, text)
            predictions.append({"audio_path": path, "pred_text": text})
            token_rows.append(
                {
                    "audio_path": path,
                    "pred_text": text,
                    "token_ids": tokens,
                    "raw_sequence_width": len(raw_ids),
                    "generated_token_count": len(tokens),
                    "decoder_prompt_token_count": prompt_token_count,
                    "effective_total_length": len(tokens) + prompt_token_count,
                    "eos_token_ids": sorted(eos_ids),
                    "has_eos": has_eos,
                    "selected_candidate_index": int(selection["selected_index"]),
                    "sequence_score": selection.get("sequence_score"),
                    "candidates": selection["candidates"],
                }
            )
            sample_rows.append(
                {
                    "audio_path": path,
                    "reference": reference,
                    "prediction": text,
                    "normalized_reference": normalized_reference,
                    "normalized_prediction": normalized_prediction,
                    "edit_distance": distance,
                    **operation,
                    "generated_token_count": len(tokens),
                    "effective_total_length": len(tokens) + prompt_token_count,
                    "has_eos": has_eos,
                    "effective_max_length": effective_max,
                    "replacement_character_count": replacement_count,
                    "token_cycle": token_cycle,
                    "text_cycle": text_cycle,
                    "runaway": bool(runaway_reasons),
                    "repeated_runaway": repeated_runaway,
                    "runaway_reasons": runaway_reasons,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "predictions.jsonl").write_text(
        "".join(strict_json(row) + "\n" for row in predictions),
        encoding="utf-8",
    )
    (output_dir / "generation_tokens.jsonl").write_text(
        "".join(strict_json(row) + "\n" for row in token_rows),
        encoding="utf-8",
    )
    (output_dir / "sample_metrics.jsonl").write_text(
        "".join(strict_json(row) + "\n" for row in sample_rows),
        encoding="utf-8",
    )

    prediction_texts = [str(row["pred_text"]) for row in predictions]
    official = compute_official_metrics(raw_references, prediction_texts)
    diagnostic = compute_diagnostic_metrics(raw_references, prediction_texts)
    error_analysis = build_error_analysis(
        [
            {
                "audio_path": row["audio_path"],
                "reference": row["reference"],
                "prediction": row["prediction"],
                "scene": "unknown",
            }
            for row in sample_rows
        ]
    )
    token_lengths = [int(row["generated_token_count"]) for row in sample_rows]
    distribution = Counter(token_lengths)
    total_audio_seconds = sum(
        float(librosa.get_duration(path=resolve_surface_audio(project_root, path)))
        for path in listed_paths
    )
    total_inference_seconds = sum(inference_seconds)
    summary = {
        "surface": surface,
        "manifest": str(manifest),
        "reference_field": reference_field,
        "rows": len(sample_rows),
        "runtime_seconds": time.perf_counter() - surface_started,
        "inference_benchmark": {
            "batch_size": batch_size,
            "samples": len(inference_seconds),
            "total_generation_seconds": total_inference_seconds,
            "latency_seconds": {
                "mean": mean(inference_seconds),
                "p50": percentile(inference_seconds, 0.50),
                "p90": percentile(inference_seconds, 0.90),
                "p95": percentile(inference_seconds, 0.95),
                "p99": percentile(inference_seconds, 0.99),
                "max": max(inference_seconds),
            },
            "throughput_samples_per_second": (
                len(inference_seconds) / total_inference_seconds
            ),
            "audio_seconds": total_audio_seconds,
            "real_time_factor": total_inference_seconds / total_audio_seconds,
        },
        "metrics": official,
        "operations": {
            "substitutions": diagnostic["substitutions"],
            "deletions": diagnostic["deletions"],
            "insertions": diagnostic["insertions"],
        },
        "severe_error_count": diagnostic["severe_error_count"],
        "severe_error_rate": diagnostic["severe_error_rate"],
        "top_20_error_contribution": diagnostic["top_20_error_contribution"],
        "generation": {
            "runaway_count": sum(bool(row["runaway"]) for row in sample_rows),
            "repeated_runaway_count": sum(
                bool(row["repeated_runaway"]) for row in sample_rows
            ),
            "no_eos_count": sum(not bool(row["has_eos"]) for row in sample_rows),
            "replacement_character_count": sum(
                int(row["replacement_character_count"]) for row in sample_rows
            ),
            "replacement_character_sample_count": sum(
                int(row["replacement_character_count"]) > 0 for row in sample_rows
            ),
            "effective_max_length_count": sum(
                bool(row["effective_max_length"]) for row in sample_rows
            ),
            "token_cycle_count": sum(
                bool(row["token_cycle"]["triggered"]) for row in sample_rows
            ),
            "long_text_repeat_count": sum(
                bool(row["text_cycle"]["triggered"]) for row in sample_rows
            ),
            "output_token_length": {
                "mean": mean(token_lengths) if token_lengths else 0.0,
                "min": min(token_lengths) if token_lengths else 0,
                "p50": percentile(token_lengths, 0.50),
                "p90": percentile(token_lengths, 0.90),
                "p95": percentile(token_lengths, 0.95),
                "p99": percentile(token_lengths, 0.99),
                "max": max(token_lengths) if token_lengths else 0,
                "histogram": {
                    str(length): count for length, count in sorted(distribution.items())
                },
            },
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "generation.json").write_text(
        json.dumps(
            summary["generation"],
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {**official, "diagnostics": diagnostic},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "error_examples.json").write_text(
        json.dumps(
            error_analysis["error_examples"],
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with (output_dir / "top_confusions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["type", "reference", "prediction", "count"])
        for row in error_analysis["substitutions"]:
            writer.writerow(
                ["substitution", row["reference"], row["prediction"], row["count"]]
            )
        for row in error_analysis["deletions"]:
            writer.writerow(["deletion", row["reference"], "<del>", row["count"]])
        for row in error_analysis["insertions"]:
            writer.writerow(["insertion", "<ins>", row["prediction"], row["count"]])
    known = [
        row
        for row in sample_rows
        if any(identifier in Path(str(row["audio_path"])).name for identifier in KNOWN_REPEAT_IDS)
    ]
    (output_dir / "known_repeats.json").write_text(
        json.dumps(known, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.output_dir}")
    if args.batch_size < 1 or args.generation_max_length < 1:
        raise SystemExit("Batch size and generation max length must be positive")
    if args.arm == "D7_MBR_B5" and args.mbr_anchor_root is None:
        raise SystemExit("D7_MBR_B5 requires --mbr-anchor-root")
    if args.arm != "D7_MBR_B5" and args.mbr_anchor_root is not None:
        raise SystemExit("--mbr-anchor-root is only valid for D7_MBR_B5")

    weight_path = args.model_dir / "model.safetensors"
    before_stat = weight_path.stat()
    before_hash = sha256_file(weight_path)
    if before_hash != args.expected_weight_sha256:
        raise SystemExit(
            f"Weight SHA-256 mismatch before inference: {before_hash} "
            f"!= {args.expected_weight_sha256}"
        )

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA is required for the four-arm decoding experiment")
    dtype = torch.float16
    processor = WhisperProcessor.from_pretrained(
        args.processor_dir,
        local_files_only=True,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_dir,
        local_files_only=True,
        use_safetensors=True,
        dtype=dtype,
    ).to(device)
    model.generation_config.language = "zh"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.eval()
    loaded_generation = {
        "max_length": int(model.generation_config.max_length),
        "num_beams": int(model.generation_config.num_beams),
        "no_repeat_ngram_size": int(
            model.generation_config.no_repeat_ngram_size
        ),
        "repetition_penalty": float(model.generation_config.repetition_penalty),
    }
    expected_loaded = {
        "max_length": 225,
        "num_beams": 2,
        "no_repeat_ngram_size": 4,
        "repetition_penalty": 1.05,
    }
    if loaded_generation != expected_loaded:
        raise SystemExit(
            "Frozen RAW_WINNER checkpoint generation config mismatch: "
            f"{loaded_generation} != {expected_loaded}"
        )

    prompt_ids = processor.get_decoder_prompt_ids(
        language="zh",
        task="transcribe",
        no_timestamps=True,
    )
    prompt_token_count = 1 + len(prompt_ids)
    if prompt_token_count != 4:
        raise SystemExit(
            f"Unexpected Whisper decoder prompt length: {prompt_token_count}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    config_receipt = {
        "arm": args.arm,
        "arm_config": ARM_CONFIGS[args.arm],
        "shared_generation": generation_kwargs(
            args.arm, args.generation_max_length
        ),
        "length_parameter_semantics": "max_length",
        "max_new_tokens": None,
        "loaded_checkpoint_generation_config": loaded_generation,
        "predict_py_effective_overrides": {
            "num_beams": int(ARM_CONFIGS[args.arm]["num_beams"]),
            "max_length_parameter": "max_length",
            "max_length": int(ARM_CONFIGS[args.arm].get("max_length", 225)),
        },
        "model_dir": str(args.model_dir),
        "processor_dir": str(args.processor_dir),
        "weight_path": str(weight_path),
        "expected_weight_sha256": args.expected_weight_sha256,
        "decoder_prompt_ids": prompt_ids,
        "decoder_prompt_token_count": prompt_token_count,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dtype": str(dtype).removeprefix("torch."),
        "mbr_anchor_root": (
            str(args.mbr_anchor_root) if args.mbr_anchor_root is not None else None
        ),
    }
    (args.output_dir / "config_receipt.json").write_text(
        json.dumps(config_receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"effective_generate_kwargs": config_receipt["shared_generation"]},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    summaries = {}
    effective_generation_max_length = int(
        ARM_CONFIGS[args.arm].get("max_length", args.generation_max_length)
    )
    all_surfaces = [
        ("validation", args.validation_manifest, "text"),
        ("public", args.public_manifest, "ref_text"),
    ]
    if args.ood_manifest is not None:
        all_surfaces.append(("ood", args.ood_manifest, "text"))
    requested = set(args.surfaces)
    if "ood" in requested and args.ood_manifest is None:
        raise SystemExit("--surfaces ood requires --ood-manifest")
    surfaces = [row for row in all_surfaces if row[0] in requested]
    for surface, manifest, reference_field in surfaces:
        summaries[surface] = evaluate_surface(
            surface=surface,
            manifest=manifest,
            reference_field=reference_field,
            model=model,
            processor=processor,
            arm=args.arm,
            project_root=args.project_root,
            output_dir=args.output_dir / surface,
            batch_size=args.batch_size,
            generation_max_length=effective_generation_max_length,
            prompt_token_count=prompt_token_count,
            device=device,
            dtype=dtype,
            max_samples=args.max_samples,
            mbr_anchor_dir=(
                args.mbr_anchor_root / surface
                if args.mbr_anchor_root is not None
                else None
            ),
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    after_stat = weight_path.stat()
    after_hash = sha256_file(weight_path)
    parity = {
        "weight_path": str(weight_path),
        "expected_sha256": args.expected_weight_sha256,
        "before_sha256": before_hash,
        "after_sha256": after_hash,
        "before_size_bytes": before_stat.st_size,
        "after_size_bytes": after_stat.st_size,
        "before_mtime_ns": before_stat.st_mtime_ns,
        "after_mtime_ns": after_stat.st_mtime_ns,
        "passed": (
            before_hash == after_hash == args.expected_weight_sha256
            and before_stat.st_size == after_stat.st_size
            and before_stat.st_mtime_ns == after_stat.st_mtime_ns
        ),
    }
    (args.output_dir / "weight_parity.json").write_text(
        json.dumps(parity, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not parity["passed"]:
        raise SystemExit("Weight parity failed after inference")
    (args.output_dir / "arm_summary.json").write_text(
        json.dumps(
            {
                "arm": args.arm,
                "config": config_receipt,
                "surfaces": summaries,
                "weight_parity": parity,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "arm": args.arm,
                "surfaces": summaries,
                "weight_parity": parity["passed"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
