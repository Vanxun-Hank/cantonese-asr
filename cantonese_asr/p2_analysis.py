"""Deterministic language-model and multi-system fusion helpers for P2."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from cantonese_asr.metrics import levenshtein_ops, normalize_prediction


def zscores(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValueError("cannot standardize an empty sequence")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    if variance == 0:
        return [0.0] * len(values)
    scale = math.sqrt(variance)
    return [(value - mean) / scale for value in values]


@dataclass
class CharacterNgramLM:
    order: int = 5
    alpha: float = 0.1

    def __post_init__(self) -> None:
        if self.order < 1 or self.alpha <= 0:
            raise ValueError("invalid n-gram configuration")
        self.counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.context_totals: Counter[str] = Counter()
        self.vocabulary: set[str] = {"</s>"}

    def fit(self, texts: Iterable[str]) -> None:
        prefix = "<s>" * (self.order - 1)
        for raw in texts:
            text = normalize_prediction(raw)
            self.vocabulary.update(text)
            sequence = prefix + text + "</s>"
            for index in range(self.order - 1, len(sequence)):
                context = sequence[index - self.order + 1 : index]
                token = sequence[index]
                self.counts[context][token] += 1
                self.context_totals[context] += 1

    def score(self, text: str) -> float:
        text = normalize_prediction(text)
        prefix = "<s>" * (self.order - 1)
        sequence = prefix + text + "</s>"
        vocabulary_size = max(1, len(self.vocabulary))
        total = 0.0
        events = 0
        for index in range(self.order - 1, len(sequence)):
            context = sequence[index - self.order + 1 : index]
            token = sequence[index]
            numerator = self.counts[context][token] + self.alpha
            denominator = self.context_totals[context] + self.alpha * vocabulary_size
            total += math.log(numerator / denominator)
            events += 1
        return total / max(events, 1)


def rerank_candidates(
    texts: Sequence[str],
    sequence_scores: Sequence[float],
    lm_scores: Sequence[float],
    weight: float,
) -> int:
    if not texts or not (len(texts) == len(sequence_scores) == len(lm_scores)):
        raise ValueError("candidate arrays must be non-empty and aligned")
    asr_z, lm_z = zscores(sequence_scores), zscores(lm_scores)
    return max(
        range(len(texts)),
        key=lambda index: (asr_z[index] + weight * lm_z[index], -index),
    )


def normalized_distance(left: str, right: str) -> float:
    left = normalize_prediction(left)
    right = normalize_prediction(right)
    distance, _ = levenshtein_ops(list(left), list(right))
    return distance / max(len(left), len(right), 1)


def mbr_medoid(hypotheses: Sequence[str], model_ranks: Sequence[int]) -> int:
    if not hypotheses or len(hypotheses) != len(model_ranks):
        raise ValueError("hypotheses and model ranks must align")
    risks = [
        sum(normalized_distance(text, other) for j, other in enumerate(hypotheses) if j != i)
        / max(len(hypotheses) - 1, 1)
        for i, text in enumerate(hypotheses)
    ]
    return min(range(len(hypotheses)), key=lambda index: (risks[index], model_ranks[index], index))


def rover_anchor_vote(hypotheses: Sequence[str], anchor_index: int) -> str:
    """Character ROVER using pairwise anchor alignments and anchor tie policy."""
    if not 0 <= anchor_index < len(hypotheses):
        raise ValueError("invalid anchor index")
    anchor = normalize_prediction(hypotheses[anchor_index])
    columns: list[list[str]] = [[character] for character in anchor]
    insertions: dict[int, list[str]] = defaultdict(list)
    for index, hypothesis in enumerate(hypotheses):
        if index == anchor_index:
            continue
        operations = levenshtein_ops(list(anchor), list(normalize_prediction(hypothesis)))[1]
        anchor_position = 0
        for source, target in operations:
            if source == "<ins>":
                insertions[anchor_position].append(target)
                continue
            columns[anchor_position].append("" if target == "<del>" else target)
            anchor_position += 1
    result: list[str] = []
    for position in range(len(columns) + 1):
        if insertions[position]:
            winner, count = Counter(insertions[position]).most_common(1)[0]
            if count > len(hypotheses) / 2:
                result.append(winner)
        if position == len(columns):
            break
        votes = Counter(columns[position])
        best_count = max(votes.values())
        winners = {char for char, count in votes.items() if count == best_count}
        anchor_char = anchor[position]
        result.append(anchor_char if anchor_char in winners else sorted(winners)[0])
    return "".join(result)
