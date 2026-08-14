"""Auditable detection and recovery rules for Whisper runaway decoding."""

from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RunawayDetection:
    triggered: bool
    non_special_length: int
    reasons: tuple[str, ...]
    trailing_ngram_size: int | None = None
    trailing_repetitions: int | None = None
    trailing_coverage: int | None = None
    tail_diversity: float | None = None
    reached_max_length: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RecoveryAssessment:
    accepted: bool
    reasons: tuple[str, ...]
    detection: RunawayDetection

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "detection": self.detection.to_dict(),
        }


def non_special_tokens(
    token_ids: Sequence[int],
    special_token_ids: Iterable[int],
) -> list[int]:
    special = {int(token) for token in special_token_ids}
    return [int(token) for token in token_ids if int(token) not in special]


def trailing_period(tokens: Sequence[int], max_ngram: int = 8) -> tuple[int, int, int] | None:
    """Return the strongest trailing repeated n-gram as (n, repeats, coverage)."""
    best: tuple[int, int, int] | None = None
    for size in range(1, min(max_ngram, len(tokens) // 2) + 1):
        unit = list(tokens[-size:])
        repeats = 1
        cursor = len(tokens) - size
        while cursor >= size and list(tokens[cursor - size : cursor]) == unit:
            repeats += 1
            cursor -= size
        coverage = repeats * size
        candidate = (size, repeats, coverage)
        if repeats >= 2 and (best is None or (coverage, repeats, -size) > (best[2], best[1], -best[0])):
            best = candidate
    return best


def detect_runaway(
    token_ids: Sequence[int],
    *,
    special_token_ids: Iterable[int],
    sequence_width: int | None = None,
    max_length: int = 225,
) -> RunawayDetection:
    tokens = non_special_tokens(token_ids, special_token_ids)
    length = len(tokens)
    reached_max = (sequence_width or len(token_ids)) >= max_length
    if length < 64:
        return RunawayDetection(
            triggered=False,
            non_special_length=length,
            reasons=(),
            reached_max_length=reached_max,
        )

    reasons: list[str] = []
    periodic = trailing_period(tokens, max_ngram=8)
    size = repeats = coverage = None
    if periodic is not None:
        size, repeats, coverage = periodic
        if repeats >= 4 and coverage >= 24:
            reasons.append("trailing_ngram_cycle")

    tail = tokens[-64:]
    diversity = len(set(tail)) / len(tail)
    if diversity < 0.25:
        reasons.append("low_tail_token_diversity")

    if reached_max:
        max_period = trailing_period(tokens, max_ngram=16)
        if max_period is not None and max_period[1] >= 3 and max_period[2] >= 18:
            reasons.append("max_length_periodic_cycle")

    return RunawayDetection(
        triggered=bool(reasons),
        non_special_length=length,
        reasons=tuple(reasons),
        trailing_ngram_size=size,
        trailing_repetitions=repeats,
        trailing_coverage=coverage,
        tail_diversity=diversity,
        reached_max_length=reached_max,
    )


def abnormal_text_reasons(text: str) -> list[str]:
    reasons: list[str] = []
    if not text.strip():
        reasons.append("empty_text")
    if "\ufffd" in text:
        reasons.append("unicode_replacement_character")
    if any(
        unicodedata.category(character) in {"Cc", "Cs"}
        and character not in "\n\r\t"
        for character in text
    ):
        reasons.append("control_or_surrogate_character")
    if any(character * 8 in text for character in "，。！？,.!?;；"):
        reasons.append("punctuation_run")
    return reasons


def assess_recovery(
    *,
    original_token_ids: Sequence[int],
    candidate_token_ids: Sequence[int],
    candidate_text: str,
    special_token_ids: Iterable[int],
    eos_token_id: int,
    sequence_width: int,
    max_length: int = 225,
) -> RecoveryAssessment:
    candidate_non_special = non_special_tokens(candidate_token_ids, special_token_ids)
    reasons = abnormal_text_reasons(candidate_text)
    if int(eos_token_id) not in {int(token) for token in candidate_token_ids}:
        reasons.append("missing_eos")
    detection = detect_runaway(
        candidate_token_ids,
        special_token_ids=special_token_ids,
        sequence_width=sequence_width,
        max_length=max_length,
    )
    if detection.triggered:
        reasons.append("runaway_persists")
    if len(candidate_non_special) < 4:
        reasons.append("obvious_abnormal_truncation")
    return RecoveryAssessment(
        accepted=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        detection=detection,
    )
