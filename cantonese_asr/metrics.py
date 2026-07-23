"""Official competition text normalization, metrics, and error analysis."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence


PUNCT_REGEX = re.compile(
    r"[，。！？；：、“”‘’（）()〖〗《》〈〉—…·,\.!\?;:\"'`\[\]\{\}\s]"
)
PREFIX_REGEX = re.compile(r"^\[(RAW|NORM)\]\s*")
CHAR_TOLERANCE = 2
_OPENCC: Any = None


def _get_opencc() -> Any:
    global _OPENCC
    if _OPENCC is None:
        try:
            from opencc import OpenCC
        except ImportError as exc:
            raise ImportError(
                "opencc is required for official traditional-to-simplified scoring; "
                "install opencc-python-reimplemented"
            ) from exc
        _OPENCC = OpenCC("t2s")
    return _OPENCC


def to_simplified(text: str) -> str:
    return _get_opencc().convert(text) if text else text


def normalize_reference(text: str) -> str:
    """Match evaluator.py: references are not converted to simplified Chinese."""
    return PUNCT_REGEX.sub("", PREFIX_REGEX.sub("", text or "")).strip()


def normalize_prediction(text: str) -> str:
    """Match evaluator.py: predictions are converted to simplified before cleanup."""
    return normalize_reference(to_simplified(text or ""))


def levenshtein_ops(ref: Sequence[str], hyp: Sequence[str]) -> tuple[int, list[tuple[str, str]]]:
    rows, cols = len(ref), len(hyp)
    dp = [[0] * (cols + 1) for _ in range(rows + 1)]
    back: list[list[str | None]] = [[None] * (cols + 1) for _ in range(rows + 1)]
    for row in range(1, rows + 1):
        dp[row][0] = row
        back[row][0] = "D"
    for col in range(1, cols + 1):
        dp[0][col] = col
        back[0][col] = "I"

    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            substitution = dp[row - 1][col - 1] + (ref[row - 1] != hyp[col - 1])
            deletion = dp[row - 1][col] + 1
            insertion = dp[row][col - 1] + 1
            best = min(substitution, deletion, insertion)
            dp[row][col] = best
            if best == substitution:
                back[row][col] = "M" if ref[row - 1] == hyp[col - 1] else "S"
            elif best == deletion:
                back[row][col] = "D"
            else:
                back[row][col] = "I"

    row, col = rows, cols
    pairs: list[tuple[str, str]] = []
    while row > 0 or col > 0:
        operation = back[row][col]
        if operation in {"M", "S"}:
            pairs.append((ref[row - 1], hyp[col - 1]))
            row -= 1
            col -= 1
        elif operation == "D":
            pairs.append((ref[row - 1], "<del>"))
            row -= 1
        elif operation == "I":
            pairs.append(("<ins>", hyp[col - 1]))
            col -= 1
        else:
            break
    pairs.reverse()
    return dp[rows][cols], pairs


def compute_official_metrics(
    references: Sequence[str],
    predictions: Sequence[str],
    tolerance: int = CHAR_TOLERANCE,
) -> dict[str, float | int]:
    if len(references) != len(predictions):
        raise ValueError(
            f"Reference/prediction length mismatch: {len(references)} != {len(predictions)}"
        )
    normalized_refs = [normalize_reference(text) for text in references]
    normalized_preds = [normalize_prediction(text) for text in predictions]
    distances = [
        levenshtein_ops(list(ref), list(pred))[0]
        for ref, pred in zip(normalized_refs, normalized_preds)
    ]
    total_chars = sum(max(1, len(ref)) for ref in normalized_refs)
    sample_count = len(normalized_refs)
    cer = sum(distances) / total_chars if total_chars else 0.0
    return {
        "num_samples": sample_count,
        "total_edit_distance": sum(distances),
        "cer": cer,
        "char_accuracy_approx": 1.0 - cer,
        "sentence_accuracy_tol2": (
            sum(distance <= tolerance for distance in distances) / sample_count
            if sample_count
            else 0.0
        ),
        "sentence_accuracy_exact": (
            sum(distance == 0 for distance in distances) / sample_count
            if sample_count
            else 0.0
        ),
        "tolerance": tolerance,
    }


def build_error_analysis(
    rows: Iterable[dict[str, Any]],
    max_examples: int = 100,
) -> dict[str, Any]:
    substitutions: Counter[tuple[str, str]] = Counter()
    insertions: Counter[str] = Counter()
    deletions: Counter[str] = Counter()
    scene_totals: Counter[str] = Counter()
    scene_matches: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for row in rows:
        reference = normalize_reference(str(row.get("reference", "")))
        prediction = normalize_prediction(str(row.get("prediction", "")))
        distance, operations = levenshtein_ops(list(reference), list(prediction))
        scene = str(row.get("scene") or "unknown")
        scene_totals[scene] += 1
        if distance <= CHAR_TOLERANCE:
            scene_matches[scene] += 1
        for source, target in operations:
            if source == "<ins>":
                insertions[target] += 1
            elif target == "<del>":
                deletions[source] += 1
            elif source != target:
                substitutions[(source, target)] += 1
        if distance and len(examples) < max_examples:
            examples.append(
                {
                    "audio_path": row.get("audio_path"),
                    "scene": scene,
                    "reference": reference,
                    "prediction": prediction,
                    "edit_distance": distance,
                }
            )

    scene_metrics = {
        scene: {
            "num_samples": total,
            "sentence_accuracy_tol2": scene_matches[scene] / total,
        }
        for scene, total in sorted(scene_totals.items())
    }
    return {
        "substitutions": [
            {"reference": src, "prediction": dst, "count": count}
            for (src, dst), count in substitutions.most_common(100)
        ],
        "insertions": [
            {"prediction": char, "count": count}
            for char, count in insertions.most_common(100)
        ],
        "deletions": [
            {"reference": char, "count": count}
            for char, count in deletions.most_common(100)
        ],
        "scene_metrics": scene_metrics,
        "error_examples": examples,
    }

