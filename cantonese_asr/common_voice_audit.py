"""Deterministic Common Voice locale admission-audit primitives."""

from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cantonese_asr.metrics import normalize_prediction


CANTONESE_MARKERS = tuple("唔冇喺嘅咁佢啲嚟嘢㗎喎咩噉嗰仲俾畀")
WRITTEN_RISK_MARKERS = ("的", "是", "这", "那", "他们", "我们", "没有", "什么", "怎么")
LATIN_RE = re.compile(r"[A-Za-z]")
DIGIT_RE = re.compile(r"\d")


def clean_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def canonical_text(value: object) -> str:
    return normalize_prediction(clean_text(value))


def profile_text(value: object) -> dict[str, Any]:
    clean = clean_text(value)
    canonical = canonical_text(clean)
    cantonese = {marker: clean.count(marker) for marker in CANTONESE_MARKERS}
    written = {marker: clean.count(marker) for marker in WRITTEN_RISK_MARKERS}
    return {
        "raw": str(value or ""),
        "clean": clean,
        "canonical": canonical,
        "characters": len(canonical),
        "cantonese_markers": {key: count for key, count in cantonese.items() if count},
        "cantonese_marker_count": sum(cantonese.values()),
        "written_risk_markers": {key: count for key, count in written.items() if count},
        "written_risk_count": sum(written.values()),
        "latin_count": len(LATIN_RE.findall(clean)),
        "digit_count": len(DIGIT_RE.findall(clean)),
    }


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values: Iterable[float]) -> dict[str, float | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"min": None, "p25": None, "p50": None, "p75": None, "p95": None, "max": None}

    def at(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "min": at(0.0),
        "p25": at(0.25),
        "p50": at(0.5),
        "p75": at(0.75),
        "p95": at(0.95),
        "max": at(1.0),
    }


def profile_split(rows: Sequence[Mapping[str, object]], split: str) -> dict[str, Any]:
    speakers = Counter(str(row.get("client_id") or "") for row in rows)
    speakers.pop("", None)
    durations = [float(row.get("duration_s") or 0.0) for row in rows]
    total_duration = sum(value for value in durations if value > 0)
    top_counts = sorted(speakers.values(), reverse=True)
    return {
        "split": split,
        "rows": len(rows),
        "unique_audio_paths": len({str(row.get("path") or "") for row in rows}),
        "unique_transcripts": len({clean_text(row.get("sentence")) for row in rows}),
        "unique_canonical_sentences": len({canonical_text(row.get("sentence")) for row in rows}),
        "speakers": len(speakers),
        "hours": total_duration / 3600.0,
        "duration_s": quantiles(value for value in durations if value > 0),
        "top1_speaker_share": (top_counts[0] / len(rows)) if rows and top_counts else 0.0,
        "top10_speaker_share": (sum(top_counts[:10]) / len(rows)) if rows else 0.0,
    }


def enrich_row(row: Mapping[str, object], source: str, split: str) -> dict[str, Any]:
    result = dict(row)
    profile = profile_text(row.get("sentence", row.get("text", "")))
    result.update(
        {
            "id": str(row.get("id") or f"{source}:{row.get('path', '')}"),
            "source": source,
            "publisher_split": split,
            "text": profile["clean"],
            "canonical": profile["canonical"],
            "text_profile": profile,
        }
    )
    return result


def compare_rows(
    candidate_rows: Sequence[Mapping[str, object]],
    reference_rows: Sequence[Mapping[str, object]],
    reference_name: str,
) -> dict[str, Any]:
    def values(rows: Sequence[Mapping[str, object]], field: str) -> set[str]:
        return {str(row.get(field) or "") for row in rows if str(row.get(field) or "")}

    candidate_count = len(candidate_rows)
    reference_audio = values(reference_rows, "audio_sha256")
    reference_pcm = values(reference_rows, "pcm_sha256")
    reference_text = values(reference_rows, "text")
    reference_canonical = values(reference_rows, "canonical")
    reference_speakers = values(reference_rows, "client_id")
    audio_overlap = sum(str(row.get("audio_sha256") or "") in reference_audio for row in candidate_rows)
    pcm_overlap = sum(
        bool(row.get("pcm_sha256")) and str(row.get("pcm_sha256")) in reference_pcm
        for row in candidate_rows
    )
    text_overlap = sum(str(row.get("text") or "") in reference_text for row in candidate_rows)
    canonical_overlap = sum(
        str(row.get("canonical") or "") in reference_canonical for row in candidate_rows
    )
    speaker_overlap = sum(
        bool(row.get("client_id")) and str(row.get("client_id")) in reference_speakers
        for row in candidate_rows
    )
    same_sentence_new_audio = sum(
        str(row.get("canonical") or "") in reference_canonical
        and str(row.get("audio_sha256") or "") not in reference_audio
        and (
            not row.get("pcm_sha256")
            or str(row.get("pcm_sha256")) not in reference_pcm
        )
        for row in candidate_rows
    )
    reference_pairs = {
        (kind, digest, str(reference.get("canonical") or ""))
        for reference in reference_rows
        for kind, digest in (
            ("pcm", str(reference.get("pcm_sha256") or "")),
            ("audio", str(reference.get("audio_sha256") or "")),
        )
        if digest and reference.get("canonical")
    }
    pair_overlap = sum(
        any(
            (kind, digest, str(row.get("canonical") or "")) in reference_pairs
            for kind, digest in (
                ("pcm", str(row.get("pcm_sha256") or "")),
                ("audio", str(row.get("audio_sha256") or "")),
            )
            if digest
        )
        for row in candidate_rows
    )

    def metric(count: int) -> dict[str, float | int]:
        return {"count": count, "rate": count / candidate_count if candidate_count else 0.0}

    return {
        "reference": reference_name,
        "candidate_rows": candidate_count,
        "reference_rows": len(reference_rows),
        "audio_sha256_overlap": metric(audio_overlap),
        "pcm_sha256_overlap": metric(pcm_overlap),
        "clean_transcript_overlap": metric(text_overlap),
        "canonical_sentence_overlap": metric(canonical_overlap),
        "audio_transcript_pair_overlap": metric(pair_overlap),
        "speaker_row_overlap": metric(speaker_overlap),
        "same_sentence_new_audio": metric(same_sentence_new_audio),
    }


def classify_candidate(
    row: Mapping[str, object],
    protected_text: set[str],
    protected_audio: set[str],
    max_duration_s: float = 30.0,
    max_tokens: int = 225,
) -> dict[str, Any]:
    reasons: list[str] = []
    duration = float(row.get("duration_s") or 0.0)
    if not str(row.get("text") or "") or not str(row.get("canonical") or ""):
        reasons.append("empty_text")
    if duration <= 0:
        reasons.append("invalid_duration")
    elif duration > max_duration_s:
        reasons.append("over_30_seconds")
    if not bool(row.get("audio_readable", True)):
        reasons.append("unreadable_audio")
    if str(row.get("canonical") or "") in protected_text:
        reasons.append("protected_text_overlap")
    hashes = {str(row.get("audio_sha256") or ""), str(row.get("pcm_sha256") or "")}
    if any(value and value in protected_audio for value in hashes):
        reasons.append("protected_audio_overlap")
    if int(row.get("label_tokens") or 0) > max_tokens:
        reasons.append("whisper_token_limit")
    return {"status": "hard_quarantine" if reasons else "accepted", "reasons": sorted(set(reasons))}


def select_review_sample(
    rows: Sequence[Mapping[str, object]], seed: int = 42
) -> list[dict[str, Any]]:
    ordered = sorted((dict(row) for row in rows), key=lambda row: str(row.get("id") or ""))
    rng = random.Random(seed)
    result: list[dict[str, Any]] = []
    selected: set[str] = set()

    def add(items: Iterable[Mapping[str, object]], limit: int, bucket: str) -> None:
        for row in items:
            sample_id = str(row.get("id") or "")
            if not sample_id or sample_id in selected:
                continue
            result.append({**row, "review_bucket": bucket})
            selected.add(sample_id)
            if sum(item["review_bucket"] == bucket for item in result) >= limit:
                break

    by_speaker: dict[str, dict[str, Any]] = {}
    shuffled = ordered[:]
    rng.shuffle(shuffled)
    for row in shuffled:
        speaker = str(row.get("client_id") or row.get("id") or "")
        by_speaker.setdefault(speaker, row)
    duration_sorted = sorted(by_speaker.values(), key=lambda row: float(row.get("duration_s") or 0))
    strata = [duration_sorted[index::4] for index in range(4)]
    random_rows: list[dict[str, Any]] = []
    for stratum in strata:
        rng.shuffle(stratum)
        random_rows.extend(stratum[:25])
    add(random_rows, 100, "random_duration_stratified")
    add(
        sorted(
            ordered,
            key=lambda row: (
                -int((row.get("text_profile") or {}).get("written_risk_count", 0)),
                str(row.get("id") or ""),
            ),
        ),
        50,
        "written_risk",
    )
    add(
        sorted(
            ordered,
            key=lambda row: (
                -int((row.get("text_profile") or {}).get("cantonese_marker_count", 0)),
                str(row.get("id") or ""),
            ),
        ),
        50,
        "cantonese_high_value",
    )
    add(
        (row for row in ordered if row.get("same_sentence_new_audio")),
        30,
        "same_sentence_new_audio",
    )
    return result


def decide_admission(metrics: Mapping[str, object]) -> dict[str, Any]:
    gates = [
        ("retention_rate", float(metrics.get("retention_rate") or 0), ">=", 0.80),
        ("new_utterances", int(metrics.get("new_utterances") or 0), ">=", 1000),
        ("new_speakers", int(metrics.get("new_speakers") or 0), ">=", 100),
        ("new_hours", float(metrics.get("new_hours") or 0), ">=", 3.0),
        ("sample_yue_rate", metrics.get("sample_yue_rate"), ">=", 0.85),
        ("protected_text_overlap", int(metrics.get("protected_text_overlap") or 0), "==", 0),
        ("protected_audio_overlap", int(metrics.get("protected_audio_overlap") or 0), "==", 0),
        ("top10_speaker_share", float(metrics.get("top10_speaker_share") or 0), "<", 0.50),
        ("incremental_coverage", bool(metrics.get("incremental_coverage")), "==", True),
    ]
    details: list[dict[str, Any]] = []
    pending = False
    passed = True
    for name, observed, operator, threshold in gates:
        if name == "sample_yue_rate" and observed is None:
            ok = False
            pending = True
        elif operator == ">=":
            ok = float(observed) >= float(threshold)
        elif operator == "<":
            ok = float(observed) < float(threshold)
        else:
            ok = observed == threshold
        details.append(
            {
                "name": name,
                "observed": observed,
                "operator": operator,
                "threshold": threshold,
                "passed": ok,
            }
        )
        passed = passed and ok
    if pending:
        decision = "pending_acoustic_review"
    elif passed:
        decision = "admit"
    elif (
        float(metrics.get("sample_yue_rate") or 0) < 0.85
        and bool(metrics.get("high_purity_subset_available"))
    ):
        decision = "conditional_admit"
    else:
        decision = "reject"
    return {"decision": decision, "gates": details}
