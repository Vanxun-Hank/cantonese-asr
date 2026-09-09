"""Deterministic language-model and multi-system fusion helpers for P2."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"

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
        self.counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.context_totals: Counter[tuple[str, ...]] = Counter()
        self.vocabulary: set[str] = {EOS, UNK}

    def fit(self, texts: Iterable[str]) -> None:
        normalized = [normalize_prediction(raw) for raw in texts]
        for text in normalized:
            self.vocabulary.update(text)
        for text in normalized:
            sequence = [BOS] * (self.order - 1) + list(text) + [EOS]
            for index in range(self.order - 1, len(sequence)):
                context = tuple(sequence[index - self.order + 1 : index])
                token = sequence[index]
                self.counts[context][token] += 1
                self.context_totals[context] += 1

    def score(self, text: str) -> float:
        text = normalize_prediction(text)
        sequence = [BOS] * (self.order - 1) + [
            token if token in self.vocabulary else UNK for token in text
        ] + [EOS]
        vocabulary_size = max(1, len(self.vocabulary))
        total = 0.0
        events = 0
        for index in range(self.order - 1, len(sequence)):
            context = tuple(sequence[index - self.order + 1 : index])
            token = sequence[index]
            numerator = self.counts[context][token] + self.alpha
            denominator = self.context_totals[context] + self.alpha * vocabulary_size
            total += math.log(numerator / denominator)
            events += 1
        # Include EOS likelihood, but use the same character denominator as the
        # neural LM. This is not the ASR reference normalizer.
        return total / max(len(text), 1)


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


def rover_anchor_vote(hypotheses: Sequence[str], anchor_index: int, *, _gap_ties: bool = False) -> str:
    """Character ROVER using pairwise anchor alignments and anchor tie policy."""
    if not 0 <= anchor_index < len(hypotheses):
        raise ValueError("invalid anchor index")
    anchor = normalize_prediction(hypotheses[anchor_index])
    system_count = len(hypotheses)
    columns: list[list[str]] = [[""] * system_count for _ in anchor]
    for position, character in enumerate(anchor):
        columns[position][anchor_index] = character
    # Each system contributes one insertion *string* per anchor gap.  Treating
    # characters from one system as separate votes spuriously creates a majority.
    insertion_votes: dict[int, list[str]] = {
        position: [""] * system_count for position in range(len(anchor) + 1)
    }
    for index, hypothesis in enumerate(hypotheses):
        if index == anchor_index:
            continue
        operations = levenshtein_ops(list(anchor), list(normalize_prediction(hypothesis)))[1]
        anchor_position = 0
        pending: list[str] = []
        for source, target in operations:
            if source == "<ins>":
                pending.append(target)
                continue
            if pending:
                insertion_votes[anchor_position][index] = "".join(pending)
                pending.clear()
            columns[anchor_position][index] = "" if target == "<del>" else target
            anchor_position += 1
        if pending:
            insertion_votes[anchor_position][index] = "".join(pending)
    result: list[str] = []
    for position in range(len(columns) + 1):
        strings = insertion_votes[position]
        # Align insertion strings to a deterministic longest insertion skeleton.
        # Empty strings still vote (gap); a lone repeated insertion cannot win.
        if any(strings):
            skeleton = min(range(system_count), key=lambda i: (-len(strings[i]), i))
            result.append(rover_anchor_vote(strings, skeleton, _gap_ties=True))
        if position == len(columns):
            break
        votes = Counter(columns[position])
        best_count = max(votes.values())
        winners = {char for char, count in votes.items() if count == best_count}
        anchor_char = anchor[position]
        if _gap_ties and "" in winners:
            result.append("")
        else:
            result.append(anchor_char if anchor_char in winners else sorted(winners)[0])
    return "".join(result)
