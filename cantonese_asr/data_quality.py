"""Deterministic quality checks for external Cantonese ASR training manifests."""

from __future__ import annotations

import csv
import html
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

import librosa
import numpy as np
import soundfile as sf

from cantonese_asr.io import sha256_file, write_jsonl
from cantonese_asr.metrics import levenshtein_ops, normalize_prediction, to_simplified


QC_VERSION = "qc-v1"
MAX_WHISPER_TOKENS = 225
SILENCE_ACTIVE_RATIO = 0.02
SILENCE_RMS_DBFS = -50.0
SEVERE_CLIP_RATIO = 0.01
NEAR_DUPLICATE_DURATION_RATIO = 0.01
NEAR_DUPLICATE_COSINE = 0.999
HARD_MISMATCH_CER = 0.95
REVIEW_MISMATCH_CER = 0.50
TEXT_NEAR_DUPLICATE_DISTANCE = 0.10
_LANGUAGE_TAG = re.compile(r"<\|(?P<language>yue|zh|en|ja|ko|nospeech)\|>")
_RICH_TAG = re.compile(r"<\|[^|>]+\|>")
_PUNCT = re.compile(r"[\s\W_]+", flags=re.UNICODE)
_CANTONESE_MARKERS = (
    "嘅", "咗", "喺", "冇", "唔", "佢", "哋", "啲", "嚟", "咩", "乜", "嗰",
    "呢", "噉", "咁", "俾", "畀", "攞", "睇", "而家", "點樣", "唔該",
)


class SenseVoiceRunner(Protocol):
    """Small interface that lets tests provide a deterministic recognition stub."""

    def transcribe(self, audio_path: Path) -> tuple[str, str, str]:
        """Return raw output, parsed language, and post-processed transcription."""

    def provenance(self) -> dict[str, Any]:
        """Return model/runtime evidence for the audit report."""


class TokenCounter(Protocol):
    def count(self, text: str) -> int:
        """Return the label length consumed by the current trainer."""


class WhisperTokenCounter:
    def __init__(self, model_dir: Path) -> None:
        try:
            from transformers import WhisperProcessor
        except ImportError as exc:
            raise RuntimeError("transformers is required for Whisper token auditing") from exc
        if not model_dir.is_dir():
            raise RuntimeError(f"Whisper model directory is missing: {model_dir}")
        self.processor = WhisperProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model_dir = model_dir

    def count(self, text: str) -> int:
        ids = self.processor.tokenizer(text).input_ids
        bos = self.processor.tokenizer.bos_token_id
        return len(ids) - 1 if ids and ids[0] == bos else len(ids)


class FunASRSenseVoiceRunner:
    """SenseVoice adapter with a deliberately strict result parser."""

    def __init__(self, model_dir: Path, device: str, batch_size: int) -> None:
        if not model_dir.is_dir():
            raise RuntimeError(f"SenseVoice model directory is missing: {model_dir}")
        try:
            import funasr
            from funasr import AutoModel
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
        except ImportError as exc:
            raise RuntimeError("FunASR/SenseVoice is not installed in this environment") from exc
        self.funasr_version = getattr(funasr, "__version__", "unknown")
        self.model_dir = model_dir
        self.device = device
        self.batch_size = batch_size
        self._postprocess = rich_transcription_postprocess
        try:
            self.model = AutoModel(
                model=str(model_dir), trust_remote_code=False, disable_update=True,
                device=device,
            )
        except Exception as exc:
            raise RuntimeError(f"Could not load SenseVoice from {model_dir}: {exc}") from exc

    def transcribe(self, audio_path: Path) -> tuple[str, str, str]:
        try:
            result = self.model.generate(
                input=str(audio_path), cache={}, language="auto", use_itn=False,
                batch_size=self.batch_size,
            )
            item = result[0] if isinstance(result, list) and result else result
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                raise ValueError(f"unexpected SenseVoice result: {type(item).__name__}")
            raw = item["text"]
            language = parse_language(raw)
            if language is None:
                raise ValueError("SenseVoice output has no supported language tag")
            return raw, language, str(self._postprocess(raw))
        except Exception as exc:
            raise RuntimeError(f"SenseVoice inference failed for {audio_path}: {exc}") from exc

    def provenance(self) -> dict[str, Any]:
        return {
            "runner": "FunASRSenseVoiceRunner",
            "model_dir": str(self.model_dir.resolve()),
            "funasr_version": self.funasr_version,
            "device": self.device,
            "language": "auto",
            "use_itn": False,
            "parser": "language tag from raw <|...|> output; rich postprocessor for text",
        }


@dataclass(frozen=True)
class AudioFeatures:
    duration_s: float
    sample_rate: int
    channels: int
    rms_dbfs: float
    active_rms_dbfs: float
    active_ratio: float
    clipping_ratio: float
    mfcc_mean: tuple[float, ...]
    mfcc_std: tuple[float, ...]
    fingerprint_bands: tuple[int, ...]


def parse_language(raw_text: str) -> str | None:
    match = _LANGUAGE_TAG.search(raw_text)
    return match.group("language") if match else None


def canonical_text(value: Any) -> str:
    """NFKC + t2s + punctuation removal for cross-script comparison."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    return _PUNCT.sub("", to_simplified(text)).strip()


def count_cantonese_markers(value: str) -> dict[str, int]:
    return {marker: value.count(marker) for marker in _CANTONESE_MARKERS if marker in value}


def ratio_to_dbfs(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-12))


def cosine(left: Iterable[float], right: Iterable[float]) -> float:
    a = np.asarray(tuple(left), dtype=np.float64)
    b = np.asarray(tuple(right), dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def _fingerprint_bands(features: np.ndarray) -> tuple[int, ...]:
    rng = np.random.default_rng(20260724)
    projections = rng.standard_normal((16, features.size))
    bits = (projections @ features >= 0).astype(np.uint8)
    signature = 0
    for index, bit in enumerate(bits):
        signature |= int(bit) << index
    return tuple((signature >> shift) & 0xFF for shift in range(0, 16, 8))


def extract_audio_features(path: Path) -> AudioFeatures:
    try:
        samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:
        raise RuntimeError(f"cannot decode audio: {exc}") from exc
    if samples.size == 0 or sample_rate <= 0:
        raise RuntimeError("audio has no decoded samples")
    channels = int(samples.shape[1])
    mono = np.mean(samples, axis=1)
    duration_s = len(mono) / float(sample_rate)
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise RuntimeError(f"invalid decoded duration: {duration_s}")
    mono_16k = librosa.resample(mono, orig_sr=sample_rate, target_sr=16_000)
    rms = librosa.feature.rms(y=mono_16k, frame_length=400, hop_length=160)[0]
    active = rms >= 10 ** (SILENCE_RMS_DBFS / 20.0)
    active_rms = float(np.sqrt(np.mean(np.square(rms[active])))) if active.any() else 0.0
    mfcc = librosa.feature.mfcc(
        y=mono_16k, sr=16_000, n_mfcc=20, n_fft=min(2048, len(mono_16k))
    )
    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)
    feature_vector = np.concatenate((mean, std))
    return AudioFeatures(
        duration_s=round(duration_s, 6), sample_rate=int(sample_rate), channels=channels,
        rms_dbfs=round(ratio_to_dbfs(float(np.sqrt(np.mean(np.square(mono))))), 4),
        active_rms_dbfs=round(ratio_to_dbfs(active_rms), 4),
        active_ratio=round(float(np.mean(active)), 6),
        clipping_ratio=round(float(np.mean(np.abs(mono) >= 0.999)), 8),
        mfcc_mean=tuple(round(float(item), 6) for item in mean),
        mfcc_std=tuple(round(float(item), 6) for item in std),
        fingerprint_bands=_fingerprint_bands(feature_vector),
    )


def resolve_path(value: Any, project_root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else project_root / path


def normalized_cer(reference: str, prediction: str) -> tuple[float, int, int]:
    ref = canonical_text(reference)
    hyp = canonical_text(_RICH_TAG.sub("", prediction))
    distance = levenshtein_ops(list(ref), list(hyp))[0]
    return distance / max(1, len(ref)), distance, len(set(ref) & set(hyp))


def _base_record(
    row: dict[str, Any], audio_path: Path, audio_sha256: str, features: AudioFeatures,
    raw: str, language: str, prediction: str, token_count: int,
) -> dict[str, Any]:
    cer, distance, common_chars = normalized_cer(str(row.get("text", "")), prediction)
    text = str(row.get("text", ""))
    normalized = canonical_text(text)
    simplified = to_simplified(unicodedata.normalize("NFKC", text))
    return {
        **row,
        "qc_version": QC_VERSION,
        "qc_status": "accepted",
        "qc_hard_reasons": [],
        "qc_review_reasons": [],
        "audio_path_resolved": str(audio_path.resolve()),
        "audio_sha256_observed": audio_sha256,
        "audio_features": {
            "duration_s": features.duration_s,
            "sample_rate": features.sample_rate,
            "channels": features.channels,
            "rms_dbfs": features.rms_dbfs,
            "active_rms_dbfs": features.active_rms_dbfs,
            "active_ratio": features.active_ratio,
            "clipping_ratio": features.clipping_ratio,
        },
        "sensevoice": {
            "raw_output": raw,
            "language": language,
            "transcription": prediction,
            "normalized_transcription": canonical_text(_RICH_TAG.sub("", prediction)),
            "normalized_cer": round(cer, 6),
            "edit_distance": distance,
            "shared_character_types": common_chars,
        },
        "whisper_effective_token_count": token_count,
        "text_profile": {
            "normalized": normalized,
            "simplified": simplified,
            "char_count": len(normalized),
            "t2s_changed": text != simplified,
            "cantonese_markers": count_cantonese_markers(text),
        },
        "near_audio_duplicate_group": None,
        "exact_text_duplicate_group": None,
        "near_text_duplicate_groups": [],
    }


def _failed_audio_record(
    row: dict[str, Any], audio_path: Path, reason: str, error: str,
    token_count: int,
) -> dict[str, Any]:
    text = str(row.get("text", ""))
    simplified = to_simplified(unicodedata.normalize("NFKC", text))
    return {
        **row,
        "qc_version": QC_VERSION,
        "qc_status": "hard_quarantine",
        "qc_hard_reasons": [reason],
        "qc_review_reasons": [],
        "audio_path_resolved": str(audio_path.resolve()),
        "audio_sha256_observed": sha256_file(audio_path) if audio_path.is_file() else None,
        "audio_features": None,
        "sensevoice": None,
        "audio_error": error,
        "whisper_effective_token_count": token_count,
        "text_profile": {
            "normalized": canonical_text(text),
            "simplified": simplified,
            "char_count": len(canonical_text(text)),
            "t2s_changed": text != simplified,
            "cantonese_markers": count_cantonese_markers(text),
        },
        "near_audio_duplicate_group": None,
        "exact_text_duplicate_group": None,
        "near_text_duplicate_groups": [],
    }


def _add_hard(record: dict[str, Any], reason: str) -> None:
    if reason not in record["qc_hard_reasons"]:
        record["qc_hard_reasons"].append(reason)
        record["qc_status"] = "hard_quarantine"


def _add_review(record: dict[str, Any], reason: str) -> None:
    if reason not in record["qc_review_reasons"] and record["qc_status"] == "accepted":
        record["qc_review_reasons"].append(reason)
        record["qc_status"] = "review"


def apply_sample_rules(record: dict[str, Any]) -> None:
    audio = record["audio_features"]
    sensevoice = record["sensevoice"]
    if sensevoice["language"] != "yue":
        _add_hard(record, "sensevoice_non_yue")
    if audio["active_ratio"] < SILENCE_ACTIVE_RATIO and audio["rms_dbfs"] <= SILENCE_RMS_DBFS:
        _add_hard(record, "near_silence")
    if audio["clipping_ratio"] >= SEVERE_CLIP_RATIO:
        _add_hard(record, "severe_clipping")
    if record["whisper_effective_token_count"] > MAX_WHISPER_TOKENS:
        _add_hard(record, "whisper_token_limit")
    if (
        sensevoice["language"] == "yue"
        and len(record["text_profile"]["normalized"]) >= 8
        and len(sensevoice["normalized_transcription"]) >= 8
        and sensevoice["normalized_cer"] >= HARD_MISMATCH_CER
        and sensevoice["shared_character_types"] <= 1
    ):
        _add_hard(record, "sensevoice_extreme_transcript_mismatch")
    elif sensevoice["normalized_cer"] >= REVIEW_MISMATCH_CER:
        _add_review(record, "sensevoice_high_cer")
    if audio["active_rms_dbfs"] < -40.0 or audio["active_rms_dbfs"] > -6.0:
        _add_review(record, "volume_outlier")


def mark_exact_audio_duplicates(records: list[dict[str, Any]]) -> None:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        digest = record.get("audio_sha256_observed")
        if digest:
            groups[str(digest)].append(record)
    for digest, group in groups.items():
        if len(group) < 2:
            continue
        canonical = min(group, key=lambda item: str(item["id"]))
        for record in group:
            record["near_audio_duplicate_group"] = f"sha256:{digest[:16]}"
            if record is not canonical:
                _add_hard(record, "duplicate_audio_sha256_regression")


def mark_near_audio_duplicates(records: list[dict[str, Any]]) -> None:
    buckets: defaultdict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        features = record.get("_features")
        if isinstance(features, AudioFeatures):
            for band_index, band in enumerate(features.fingerprint_bands):
                buckets[(band_index, band)].append(record)
    pairs: set[tuple[str, str]] = set()
    by_id = {str(record["id"]): record for record in records}
    for bucket in buckets.values():
        ordered = sorted(bucket, key=lambda item: str(item["id"]))
        for index, left in enumerate(ordered):
            for right in ordered[index + 1:]:
                pairs.add((str(left["id"]), str(right["id"])))
    adjacency: defaultdict[str, set[str]] = defaultdict(set)
    for left_id, right_id in sorted(pairs):
        left, right = by_id[left_id], by_id[right_id]
        one, two = left["_features"], right["_features"]
        duration_ratio = abs(one.duration_s - two.duration_s) / max(one.duration_s, two.duration_s)
        if duration_ratio > NEAR_DUPLICATE_DURATION_RATIO:
            continue
        if cosine(one.mfcc_mean, two.mfcc_mean) < NEAR_DUPLICATE_COSINE:
            continue
        if cosine(one.mfcc_std, two.mfcc_std) < NEAR_DUPLICATE_COSINE:
            continue
        adjacency[left_id].add(right_id)
        adjacency[right_id].add(left_id)
    group_number = 0
    seen: set[str] = set()
    for sample_id in sorted(adjacency):
        if sample_id in seen:
            continue
        stack, component = [sample_id], []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            stack.extend(sorted(adjacency[current] - seen, reverse=True))
        if len(component) < 2:
            continue
        group_number += 1
        group_id = f"near_audio:{group_number:06d}"
        canonical = min(component)
        for item_id in component:
            record = by_id[item_id]
            record["near_audio_duplicate_group"] = group_id
            if item_id != canonical:
                _add_hard(record, "near_duplicate_audio")


def mark_text_duplicates(records: list[dict[str, Any]]) -> None:
    exact: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        normalized = str(record["text_profile"]["normalized"])
        if normalized:
            exact[normalized].append(record)
    number = 0
    for group in exact.values():
        if len(group) > 1:
            number += 1
            group_id = f"exact_text:{number:06d}"
            for record in group:
                record["exact_text_duplicate_group"] = group_id
                _add_review(record, "exact_duplicate_text")

    inverted: defaultdict[str, set[str]] = defaultdict(set)
    by_id = {str(record["id"]): record for record in records}
    for sample_id, record in by_id.items():
        text = str(record["text_profile"]["normalized"])
        for gram in {text[index:index + 2] for index in range(max(0, len(text) - 1))}:
            if len(gram) == 2:
                inverted[gram].add(sample_id)
    candidates: Counter[tuple[str, str]] = Counter()
    for sample_ids in inverted.values():
        # Frequent function-word bigrams can otherwise create an impractical
        # quadratic candidate set. Exact duplicate groups are still reported.
        if len(sample_ids) > 250:
            continue
        ordered = sorted(sample_ids)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1:]:
                candidates[(left, right)] += 1
    group_number = 0
    for (left_id, right_id), shared_grams in sorted(candidates.items()):
        if shared_grams < 2:
            continue
        left, right = by_id[left_id], by_id[right_id]
        left_text = str(left["text_profile"]["normalized"])
        right_text = str(right["text_profile"]["normalized"])
        if left_text == right_text or not left_text or not right_text:
            continue
        ratio = min(len(left_text), len(right_text)) / max(len(left_text), len(right_text))
        if ratio < 0.9:
            continue
        distance = levenshtein_ops(list(left_text), list(right_text))[0]
        if distance / max(len(left_text), len(right_text)) > TEXT_NEAR_DUPLICATE_DISTANCE:
            continue
        group_number += 1
        group_id = f"near_text:{group_number:06d}"
        left["near_text_duplicate_groups"].append(group_id)
        right["near_text_duplicate_groups"].append(group_id)
        _add_review(left, "near_duplicate_text")
        _add_review(right, "near_duplicate_text")


def audit_rows(
    rows: Iterable[dict[str, Any]], project_root: Path, runner: SenseVoiceRunner,
    token_counter: TokenCounter,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in sorted((dict(item) for item in rows), key=lambda item: (str(item.get("source", "")), str(item.get("id", "")))):
        sample_id = str(row.get("id", ""))
        if not sample_id or sample_id in seen_ids:
            raise RuntimeError(f"Input manifests must contain globally unique non-empty IDs; got {sample_id!r}")
        seen_ids.add(sample_id)
        audio_path = resolve_path(row.get("audio_path", ""), project_root)
        token_count = token_counter.count(str(row.get("text", "")))
        if not audio_path.is_file():
            records.append(_failed_audio_record(row, audio_path, "missing_audio", "file does not exist", token_count))
            continue
        try:
            features = extract_audio_features(audio_path)
        except RuntimeError as exc:
            records.append(_failed_audio_record(row, audio_path, "unreadable_audio", str(exc), token_count))
            continue
        raw, language, prediction = runner.transcribe(audio_path)
        if language not in {"yue", "zh", "en", "ja", "ko", "nospeech"}:
            raise RuntimeError(f"Unsupported SenseVoice language for {sample_id}: {language!r}")
        record = _base_record(
            row, audio_path, sha256_file(audio_path), features, raw, language, prediction,
            token_count,
        )
        record["_features"] = features
        apply_sample_rules(record)
        records.append(record)
    mark_exact_audio_duplicates(records)
    mark_near_audio_duplicates(records)
    mark_text_duplicates(records)
    for record in records:
        record.pop("_features", None)
        record["qc_hard_reasons"].sort()
        record["qc_review_reasons"].sort()
        record["near_text_duplicate_groups"].sort()
    return records


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p50": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(values)
    def at(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]
    return {"min": at(0), "p50": at(0.5), "p95": at(0.95), "p99": at(0.99), "max": at(1)}


def build_report(records: list[dict[str, Any]], provenance: dict[str, Any], input_hashes: dict[str, str]) -> dict[str, Any]:
    by_source: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[str(record.get("source") or "unknown")].append(record)
    sources: dict[str, Any] = {}
    for source, items in sorted(by_source.items()):
        sensevoice_items = [item["sensevoice"] for item in items if isinstance(item.get("sensevoice"), dict)]
        audio_items = [item["audio_features"] for item in items if isinstance(item.get("audio_features"), dict)]
        sources[source] = {
            "rows": len(items),
            "accepted": sum(item["qc_status"] == "accepted" for item in items),
            "review": sum(item["qc_status"] == "review" for item in items),
            "hard_quarantine": sum(item["qc_status"] == "hard_quarantine" for item in items),
            "language": dict(sorted(Counter((item.get("sensevoice") or {}).get("language") or "not_run" for item in items).items())),
            "cer": _quantiles([float(item["normalized_cer"]) for item in sensevoice_items]),
            "text_chars": _quantiles([float(item["text_profile"]["char_count"]) for item in items]),
            "whisper_tokens": _quantiles([float(item["whisper_effective_token_count"]) for item in items]),
            "active_rms_dbfs": _quantiles([float(item["active_rms_dbfs"]) for item in audio_items]),
            "scene_distribution": dict(sorted(Counter(str(item.get("scene") or "unknown") for item in items).items())),
            "traditional_to_simplified_changed": sum(bool(item["text_profile"]["t2s_changed"]) for item in items),
            "cantonese_marker_rows": sum(bool(item["text_profile"]["cantonese_markers"]) for item in items),
            "cantonese_marker_counts": dict(sorted(Counter(
                marker
                for item in items
                for marker, count in item["text_profile"]["cantonese_markers"].items()
                for _ in range(count)
            ).items())),
        }
    hard = [item for item in records if item["qc_status"] == "hard_quarantine"]
    reviews = [item for item in records if item["qc_status"] == "review"]
    return {
        "qc_version": QC_VERSION,
        "inputs": input_hashes,
        "sensevoice": provenance,
        "thresholds": {
            "language_required": "yue", "silence_active_ratio_lt": SILENCE_ACTIVE_RATIO,
            "silence_rms_dbfs_lte": SILENCE_RMS_DBFS, "clipping_ratio_gte": SEVERE_CLIP_RATIO,
            "whisper_effective_tokens_lte": MAX_WHISPER_TOKENS,
            "near_audio_duration_ratio_lte": NEAR_DUPLICATE_DURATION_RATIO,
            "near_audio_cosine_gte": NEAR_DUPLICATE_COSINE,
            "hard_mismatch_cer_gte": HARD_MISMATCH_CER,
            "near_text_distance_lte": TEXT_NEAR_DUPLICATE_DISTANCE,
        },
        "counts": {"input": len(records), "accepted": sum(item["qc_status"] == "accepted" for item in records), "review": len(reviews), "hard_quarantine": len(hard)},
        "hard_reasons": dict(sorted(Counter(reason for item in hard for reason in item["qc_hard_reasons"]).items())),
        "review_reasons": dict(sorted(Counter(reason for item in reviews for reason in item["qc_review_reasons"]).items())),
        "sources": sources,
        "notes": ["Text duplicate groups are review-only and never automatically removed.", "External scene values are reported as supplied; no scene taxonomy is inferred."],
    }


def write_csv_reports(output_dir: Path, records: list[dict[str, Any]]) -> None:
    scene_rows: Counter[tuple[str, str]] = Counter()
    length_rows: list[dict[str, Any]] = []
    for record in records:
        source = str(record.get("source") or "unknown")
        scene_rows[(source, str(record.get("scene") or "unknown"))] += 1
        length_rows.append({"id": record["id"], "source": source, "text_chars": record["text_profile"]["char_count"], "whisper_effective_tokens": record["whisper_effective_token_count"], "status": record["qc_status"]})
    with (output_dir / "source_scene_distribution.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "scene", "rows"])
        writer.writeheader()
        writer.writerows({"source": source, "scene": scene, "rows": rows} for (source, scene), rows in sorted(scene_rows.items()))
    with (output_dir / "text_token_lengths.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "source", "text_chars", "whisper_effective_tokens", "status"])
        writer.writeheader()
        writer.writerows(sorted(length_rows, key=lambda item: (item["source"], str(item["id"]))))


def write_html_report(path: Path, report: dict[str, Any]) -> None:
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (source, details["rows"], details["accepted"], details["review"], details["hard_quarantine"], json.dumps(details["language"], ensure_ascii=False))) + "</tr>"
        for source, details in report["sources"].items()
    )
    body = f"""<!doctype html><meta charset=\"utf-8\"><title>External QC v1</title>
<h1>External data QC v1</h1><p>Input: {report['counts']['input']} · accepted: {report['counts']['accepted']} · review: {report['counts']['review']} · quarantined: {report['counts']['hard_quarantine']}</p>
<h2>By source</h2><table border=\"1\"><thead><tr><th>source</th><th>rows</th><th>accepted</th><th>review</th><th>quarantine</th><th>SenseVoice language</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Hard reasons</h2><pre>{html.escape(json.dumps(report['hard_reasons'], ensure_ascii=False, indent=2))}</pre>
<h2>Review reasons</h2><pre>{html.escape(json.dumps(report['review_reasons'], ensure_ascii=False, indent=2))}</pre>
<h2>Thresholds</h2><pre>{html.escape(json.dumps(report['thresholds'], ensure_ascii=False, indent=2))}</pre>"""
    path.write_text(body + "\n", encoding="utf-8")


def write_audit_outputs(
    output_dir: Path, records: list[dict[str, Any]], provenance: dict[str, Any],
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(records, provenance, input_hashes)
    clean = [record for record in records if record["qc_status"] == "accepted"]
    common_voice = [record for record in clean if record.get("source") == "common_voice_26_zh_HK"]
    mdcc = [record for record in clean if record.get("source") == "mdcc"]
    write_jsonl(output_dir / "audit_results.jsonl", records)
    write_jsonl(output_dir / "hard_quarantine.jsonl", [record for record in records if record["qc_status"] == "hard_quarantine"])
    write_jsonl(output_dir / "review_queue.jsonl", [record for record in records if record["qc_status"] == "review"])
    write_jsonl(output_dir / "common_voice_train.jsonl", common_voice)
    write_jsonl(output_dir / "mdcc_train.jsonl", mdcc)
    write_jsonl(output_dir / "external_all.jsonl", common_voice + mdcc)
    (output_dir / "data_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "audit_config.json").write_text(json.dumps({"qc_version": QC_VERSION, "thresholds": report["thresholds"], "sensevoice": provenance, "inputs": input_hashes}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv_reports(output_dir, records)
    write_html_report(output_dir / "report.html", report)
    return report
