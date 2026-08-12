"""Deterministic epoch-aware sampling policies for controlled ASR experiments."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import Sampler


SOURCE_OFFICIAL = "official"
SOURCE_CV = "cv"
SOURCE_MDCC = "mdcc"
KNOWN_GROUPS = (SOURCE_OFFICIAL, SOURCE_CV, SOURCE_MDCC)
ROUND11_GROUPS = ("official", "cv_zh_hk", "cv_yue", "mdcc")
ROUND12_GROUPS = ("wenet", "official")
EXP004_WENET_GROUPS = ("official", "cv", "mdcc", "wenet")


class FixedExposureSampler(Sampler[int]):
    """Yield a once-shuffled frozen manifest exactly once in row order."""

    def __init__(self, length: int) -> None:
        if length < 1:
            raise ValueError("fixed exposure length must be positive")
        self.length = int(length)

    def __iter__(self):
        return iter(range(self.length))

    def __len__(self) -> int:
        return self.length


def fixed_exposure_topology_receipt(
    sample_ids: Sequence[str],
    *,
    world_size: int,
    per_device_batch: int,
    gradient_accumulation_steps: int,
) -> dict[str, Any]:
    """Describe optimizer-step global IDs for one distributed topology."""
    if min(world_size, per_device_batch, gradient_accumulation_steps) < 1:
        raise ValueError("topology values must be positive")
    global_batch = world_size * per_device_batch * gradient_accumulation_steps
    if len(sample_ids) % global_batch:
        raise ValueError(
            f"exposure length {len(sample_ids)} is not divisible by global batch {global_batch}"
        )
    steps = []
    for start in range(0, len(sample_ids), global_batch):
        global_ids = list(sample_ids[start : start + global_batch])
        rank_microbatches = {str(rank): [] for rank in range(world_size)}
        cursor = 0
        for _microstep in range(gradient_accumulation_steps):
            for rank in range(world_size):
                stop = cursor + per_device_batch
                rank_microbatches[str(rank)].append(global_ids[cursor:stop])
                cursor = stop
        steps.append(
            {
                "optimizer_step": len(steps) + 1,
                "global_ids": global_ids,
                "rank_microbatches": rank_microbatches,
            }
        )
    digest = hashlib.sha256(
        json.dumps(
            [step["global_ids"] for step in steps],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "world_size": world_size,
        "per_device_batch": per_device_batch,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "global_effective_batch": global_batch,
        "optimizer_steps": len(steps),
        "global_step_ids_sha256": digest,
        "steps": steps,
    }


def source_group(row: Mapping[str, Any]) -> str:
    """Map immutable manifest source labels to the three EXP004 source groups."""
    source = str(row.get("source", "")).strip().lower()
    if source == "official" or source.startswith("official_"):
        return SOURCE_OFFICIAL
    if source == "mdcc" or source.startswith("mdcc_"):
        return SOURCE_MDCC
    if "common_voice" in source or source in {"cv", "yue", "zh-hk", "zh_hk"}:
        return SOURCE_CV
    raise ValueError(f"Unsupported training source: {row.get('source')!r}")


def round11_source_group(row: Mapping[str, Any]) -> str:
    """Return the four preregistered Round 11 source components."""
    explicit = str(row.get("round11_source", "")).strip().lower()
    if explicit:
        if explicit not in ROUND11_GROUPS:
            raise ValueError(f"Unsupported Round 11 source: {explicit!r}")
        return explicit
    source = str(row.get("source", "")).strip().lower()
    if source == "official" or source.startswith("official_"):
        return "official"
    if source == "mdcc" or source.startswith("mdcc_"):
        return "mdcc"
    if "common_voice" in source and ("yue" in source or source.endswith("_yue")):
        return "cv_yue"
    if "common_voice" in source or source in {"cv", "zh-hk", "zh_hk"}:
        return "cv_zh_hk"
    raise ValueError(f"Unsupported Round 11 training source: {row.get('source')!r}")


def round12_source_group(row: Mapping[str, Any]) -> str:
    """Return the two source groups used by the Round 12 Wenet experiments."""
    source = str(row.get("source", "")).strip().lower()
    if source == "official" or source.startswith("official_"):
        return "official"
    if "wenetspeech" in source or source.startswith("wenet"):
        return "wenet"
    raise ValueError(f"Unsupported Round 12 training source: {row.get('source')!r}")


def exp004_wenet_source_group(row: Mapping[str, Any]) -> str:
    """Return immutable EXP004 components plus the WenetSpeech-Yue group."""
    source = str(row.get("source", "")).strip().lower()
    if "wenetspeech" in source or source.startswith("wenet"):
        return "wenet"
    return source_group(row)


def largest_remainder_counts(
    total: int,
    weights: Mapping[str, int],
) -> dict[str, int]:
    """Allocate an integer total proportionally with deterministic remainders."""
    if total < 0 or not weights or any(int(weight) < 0 for weight in weights.values()):
        raise ValueError("largest_remainder_counts requires non-negative inputs")
    weight_total = sum(int(weight) for weight in weights.values())
    if weight_total < 1:
        raise ValueError("largest_remainder_counts requires positive total weight")
    exact = {
        group: Fraction(total * int(weight), weight_total)
        for group, weight in weights.items()
    }
    result = {
        group: int(value.numerator // value.denominator)
        for group, value in exact.items()
    }
    missing = total - sum(result.values())
    order = sorted(
        weights,
        key=lambda group: (
            -(exact[group] - result[group]),
            group,
        ),
    )
    for group in order[:missing]:
        result[group] += 1
    return result


def stable_seed(seed: int, *parts: object) -> int:
    payload = ":".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big"
    )


def deterministic_permutation(values: Sequence[int], seed: int, *parts: object) -> list[int]:
    generator = torch.Generator()
    generator.manual_seed(stable_seed(seed, *parts) % (2**63 - 1))
    order = torch.randperm(len(values), generator=generator).tolist()
    return [int(values[index]) for index in order]


def cyclic_unique_sample(
    values: Sequence[int],
    count: int,
    *,
    seed: int,
    epoch: int,
    group: str,
) -> list[int]:
    """Rotate a fixed permutation so omissions change between epochs."""
    if count < 0 or count > len(values):
        raise ValueError("cyclic_unique_sample requires 0 <= count <= pool size")
    if not values or count == 0:
        return []
    ordered = deterministic_permutation(values, seed, "pool", group)
    start = (epoch * count) % len(ordered)
    doubled = ordered + ordered
    return doubled[start : start + count]


def repeated_sample(
    values: Sequence[int],
    count: int,
    *,
    seed: int,
    epoch: int,
    group: str,
) -> list[int]:
    """Deterministically sample a pool, cycling permutations when repeats are needed."""
    if count < 0 or not values:
        raise ValueError("repeated_sample requires a non-empty pool and non-negative count")
    draws: list[int] = []
    cycle = 0
    while len(draws) < count:
        ordered = deterministic_permutation(values, seed, "repeat", group, epoch, cycle)
        draws.extend(ordered[: count - len(draws)])
        cycle += 1
    return draws


def avoid_batch_duplicates(indices: list[int], batch_size: int) -> list[int]:
    """Swap later draws so an utterance is not repeated inside one minibatch."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    result = list(indices)
    for start in range(0, len(result), batch_size):
        end = min(start + batch_size, len(result))
        seen: set[int] = set()
        for position in range(start, end):
            current = result[position]
            if current not in seen:
                seen.add(current)
                continue
            replacement = next(
                (
                    later
                    for later in range(end, len(result))
                    if result[later] not in seen
                    and result[later] not in result[position + 1 : end]
                ),
                None,
            )
            if replacement is None:
                raise ValueError("Unable to remove duplicate utterance from minibatch")
            result[position], result[replacement] = result[replacement], result[position]
            seen.add(result[position])
    return result


def avoid_batch_duplicates_preserving_groups(
    indices: list[int],
    batch_size: int,
    group_fn: Any,
    rows: Sequence[Mapping[str, Any]],
) -> list[int]:
    """Remove within-batch duplicates without changing source positions."""
    result = list(indices)
    for start in range(0, len(result), batch_size):
        end = min(start + batch_size, len(result))
        seen: set[int] = set()
        for position in range(start, end):
            current = result[position]
            if current not in seen:
                seen.add(current)
                continue
            current_group = group_fn(rows[current])
            replacement = next(
                (
                    later
                    for later in range(end, len(result))
                    if group_fn(rows[result[later]]) == current_group
                    and result[later] not in seen
                    and result[later] not in result[position + 1 : end]
                ),
                None,
            )
            if replacement is None:
                raise ValueError(
                    "Unable to remove duplicate utterance while preserving source positions"
                )
            result[position], result[replacement] = (
                result[replacement],
                result[position],
            )
            seen.add(result[position])
    return result


class EpochEncodedSampler(Sampler[int]):
    """Yield epoch-encoded integer indices and persist deterministic draw receipts."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        policy: str,
        seed: int,
        batch_size: int,
        receipt_dir: Path | None = None,
        draws_per_source: int = 7_768,
        round12_stream_length: int = 64_000,
        exp004_wenet_ratio: float = 0.0,
        exp004_wenet_stream_length: int = 25_600,
        exp004_wenet_window_length: int = 1_600,
        exp004_wenet_per_window: int | None = None,
        exp004_wenet_fixed_manifest_order: bool = False,
    ) -> None:
        if policy not in {
            "official_speed",
            "source_balanced",
            "round11_source_balanced",
            "wenet_official_80_20",
            "exp004_wenet_mix",
        }:
            raise ValueError(f"Unsupported sampling policy: {policy}")
        self.rows = rows
        self.policy = policy
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.receipt_dir = receipt_dir
        self.draws_per_source = int(draws_per_source)
        self.round12_stream_length = int(round12_stream_length)
        self.exp004_wenet_stream_length = int(exp004_wenet_stream_length)
        self.exp004_wenet_window_length = int(exp004_wenet_window_length)
        self.exp004_wenet_per_window = (
            int(exp004_wenet_per_window)
            if exp004_wenet_per_window is not None
            else None
        )
        self.exp004_wenet_fixed_manifest_order = bool(
            exp004_wenet_fixed_manifest_order
        )
        self.exp004_wenet_ratio = (
            Fraction(
                self.exp004_wenet_per_window,
                self.exp004_wenet_window_length,
            )
            if self.exp004_wenet_per_window is not None
            else Fraction(str(exp004_wenet_ratio))
        )
        self.epoch = 0
        self.pools: dict[str, list[int]] = defaultdict(list)
        if policy == "round11_source_balanced":
            group_fn = round11_source_group
        elif policy == "wenet_official_80_20":
            group_fn = round12_source_group
        elif policy == "exp004_wenet_mix":
            group_fn = exp004_wenet_source_group
        else:
            group_fn = source_group
        self.group_fn = group_fn
        for index, row in enumerate(rows):
            self.pools[group_fn(row)].append(index)
        if policy == "round11_source_balanced":
            expected_groups = ROUND11_GROUPS
        elif policy == "wenet_official_80_20":
            expected_groups = ROUND12_GROUPS
        elif policy == "exp004_wenet_mix":
            expected_groups = (
                EXP004_WENET_GROUPS
                if self.exp004_wenet_ratio > 0
                else KNOWN_GROUPS
            )
        else:
            expected_groups = KNOWN_GROUPS
        self.expected_groups = expected_groups
        if set(self.pools) != set(expected_groups):
            raise ValueError(
                f"Expected source groups {sorted(expected_groups)}, "
                f"got {sorted(self.pools)}"
            )
        if policy == "official_speed" and len(rows) != 23_304:
            raise ValueError("Official speed policy requires the 23,304-row EXP004 pool")
        if policy == "source_balanced" and draws_per_source * 3 != 23_304:
            raise ValueError("Source-balanced policy must produce exactly 23,304 draws")
        if policy == "round11_source_balanced":
            if 23_304 % len(ROUND11_GROUPS):
                raise ValueError("Round 11 fixed budget is not divisible by four")
            self.draws_per_source = 23_304 // len(ROUND11_GROUPS)
        if policy == "wenet_official_80_20":
            if self.round12_stream_length < 5:
                raise ValueError("Round 12 stream length must be at least five")
            if self.round12_stream_length % 6_400:
                raise ValueError(
                    "Round 12 stream length must be divisible by 6,400 so every "
                    "400 optimizer-step window at effective batch 16 is auditable"
                )
        if policy == "exp004_wenet_mix":
            if not 0 <= self.exp004_wenet_ratio < 1:
                raise ValueError("EXP004/Wenet ratio must be within [0, 1)")
            if (
                self.exp004_wenet_per_window is not None
                and not 0
                <= self.exp004_wenet_per_window
                < self.exp004_wenet_window_length
            ):
                raise ValueError(
                    "Exact Wenet count must be within [0, audit window length)"
                )
            if self.exp004_wenet_stream_length < self.exp004_wenet_window_length:
                raise ValueError("EXP004/Wenet stream must contain at least one window")
            if self.exp004_wenet_stream_length % self.exp004_wenet_window_length:
                raise ValueError(
                    "EXP004/Wenet stream length must be divisible by its audit window"
                )
            if self.exp004_wenet_window_length % self.batch_size:
                raise ValueError(
                    "EXP004/Wenet audit window must be divisible by minibatch size"
                )
            if (
                self.exp004_wenet_window_length * self.exp004_wenet_ratio
            ).denominator != 1:
                raise ValueError(
                    "EXP004/Wenet ratio must produce an integer Wenet count per window"
                )
            expected_exp004_sizes = {
                "official": 6_292,
                "cv": 8_451,
                "mdcc": 8_561,
            }
            actual_exp004_sizes = {
                group: len(self.pools[group])
                for group in expected_exp004_sizes
            }
            if actual_exp004_sizes != expected_exp004_sizes:
                raise ValueError(
                    "EXP004/Wenet policy requires the immutable 23,304-row "
                    f"EXP004 pool; got {actual_exp004_sizes}"
                )

    def __len__(self) -> int:
        if self.policy == "wenet_official_80_20":
            return self.round12_stream_length
        if self.policy == "exp004_wenet_mix":
            return self.exp004_wenet_stream_length
        if self.policy == "official_speed":
            return len(self.rows)
        return self.draws_per_source * len(self.expected_groups)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _indices(self) -> list[int]:
        if self.policy == "official_speed":
            indices = deterministic_permutation(
                list(range(len(self.rows))), self.seed, "official_speed", self.epoch
            )
        elif self.policy == "wenet_official_80_20":
            wenet_count = self.round12_stream_length * 4 // 5
            official_count = self.round12_stream_length // 5
            draws = {
                "wenet": repeated_sample(
                    self.pools["wenet"],
                    wenet_count,
                    seed=self.seed,
                    epoch=self.epoch,
                    group="wenet",
                ),
                "official": repeated_sample(
                    self.pools["official"],
                    official_count,
                    seed=self.seed,
                    epoch=self.epoch,
                    group="official",
                ),
            }
            cursors = {"wenet": 0, "official": 0}
            indices = []
            for block in range(self.round12_stream_length // 5):
                source_order = deterministic_permutation(
                    [0, 1, 2, 3, 4],
                    self.seed,
                    "round12_source_order",
                    self.epoch,
                    block,
                )
                official_position = source_order[0]
                for position in range(5):
                    group = "official" if position == official_position else "wenet"
                    indices.append(draws[group][cursors[group]])
                    cursors[group] += 1
            indices = avoid_batch_duplicates_preserving_groups(
                indices,
                self.batch_size,
                self.group_fn,
                self.rows,
            )
        elif self.policy == "exp004_wenet_mix":
            window_length = self.exp004_wenet_window_length
            window_count = self.exp004_wenet_stream_length // window_length
            wenet_per_window = (
                self.exp004_wenet_per_window
                if self.exp004_wenet_per_window is not None
                else int(window_length * self.exp004_wenet_ratio)
            )
            exp004_per_window = window_length - wenet_per_window
            exp004_weights = {
                "official": len(self.pools["official"]),
                "cv": len(self.pools["cv"]),
                "mdcc": len(self.pools["mdcc"]),
            }
            total_counts = largest_remainder_counts(
                exp004_per_window * window_count,
                exp004_weights,
            )
            if self.exp004_wenet_ratio > 0:
                total_counts["wenet"] = wenet_per_window * window_count
            draws = {
                group: repeated_sample(
                    self.pools[group],
                    count,
                    seed=self.seed,
                    epoch=self.epoch,
                    group=group,
                )
                for group, count in total_counts.items()
            }
            if self.exp004_wenet_fixed_manifest_order:
                requested_wenet = int(total_counts.get("wenet", 0))
                if requested_wenet != len(self.pools.get("wenet", [])):
                    raise ValueError(
                        "Fixed Wenet manifest order requires the Wenet pool size "
                        "to equal requested Wenet exposure; got "
                        f"pool={len(self.pools.get('wenet', []))}, "
                        f"requested={requested_wenet}"
                    )
                draws["wenet"] = list(self.pools["wenet"])
            cursors = {group: 0 for group in total_counts}
            prior_exp004 = {group: 0 for group in exp004_weights}
            indices = []
            for window in range(window_count):
                cumulative_exp004 = largest_remainder_counts(
                    exp004_per_window * (window + 1),
                    exp004_weights,
                )
                window_counts = {
                    group: cumulative_exp004[group] - prior_exp004[group]
                    for group in exp004_weights
                }
                prior_exp004 = cumulative_exp004
                if self.exp004_wenet_ratio > 0:
                    window_counts["wenet"] = wenet_per_window
                window_indices: list[int] = []
                for group, count in window_counts.items():
                    start = cursors[group]
                    end = start + count
                    window_indices.extend(draws[group][start:end])
                    cursors[group] = end
                indices.extend(
                    deterministic_permutation(
                        window_indices,
                        self.seed,
                        "exp004_wenet_window",
                        self.epoch,
                        window,
                    )
                )
            if self.exp004_wenet_fixed_manifest_order:
                frozen_wenet = iter(draws["wenet"])
                indices = [
                    (
                        next(frozen_wenet)
                        if self.group_fn(self.rows[index]) == "wenet"
                        else index
                    )
                    for index in indices
                ]
            indices = avoid_batch_duplicates_preserving_groups(
                indices,
                self.batch_size,
                self.group_fn,
                self.rows,
            )
        else:
            indices = []
            for group in self.expected_groups:
                pool = self.pools[group]
                if self.draws_per_source <= len(pool):
                    selected = cyclic_unique_sample(
                        pool,
                        self.draws_per_source,
                        seed=self.seed,
                        epoch=self.epoch,
                        group=group,
                    )
                else:
                    selected = repeated_sample(
                        pool,
                        self.draws_per_source,
                        seed=self.seed,
                        epoch=self.epoch,
                        group=group,
                    )
                indices.extend(selected)
            indices = deterministic_permutation(
                indices, self.seed, "source_balanced_merge", self.epoch
            )
            indices = avoid_batch_duplicates(indices, self.batch_size)
        return indices

    def _write_receipt(self, indices: Sequence[int]) -> None:
        if self.receipt_dir is None:
            return
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        group_counts = Counter(self.group_fn(self.rows[index]) for index in indices)
        unique_by_group = {
            group: len(
                {
                    index
                    for index in indices
                    if self.group_fn(self.rows[index]) == group
                }
            )
            for group in self.expected_groups
        }
        duplicate_draws = len(indices) - len(set(indices))
        payload = {
            "policy": self.policy,
            "seed": self.seed,
            "epoch": self.epoch,
            "total_draws": len(indices),
            "source_draws": dict(sorted(group_counts.items())),
            "unique_rows": len(set(indices)),
            "unique_rows_by_source": unique_by_group,
            "duplicate_draws": duplicate_draws,
            "duplicate_draw_rate": duplicate_draws / len(indices),
            "batch_size": self.batch_size,
            "within_minibatch_duplicates": sum(
                len(batch) - len(set(batch))
                for batch in (
                    indices[start : start + self.batch_size]
                    for start in range(0, len(indices), self.batch_size)
                )
            ),
        }
        if self.policy == "exp004_wenet_mix":
            payload.update(
                {
                    "target_wenet_ratio": float(self.exp004_wenet_ratio),
                    "stream_length": self.exp004_wenet_stream_length,
                    "audit_window_length": self.exp004_wenet_window_length,
                    "wenet_per_audit_window": (
                        self.exp004_wenet_per_window
                        if self.exp004_wenet_per_window is not None
                        else int(
                            self.exp004_wenet_window_length
                            * self.exp004_wenet_ratio
                        )
                    ),
                    "exp004_per_audit_window": (
                        self.exp004_wenet_window_length
                        - (
                            self.exp004_wenet_per_window
                            if self.exp004_wenet_per_window is not None
                            else int(
                                self.exp004_wenet_window_length
                                * self.exp004_wenet_ratio
                            )
                        )
                    ),
                    "quota_mode": (
                        "exact_counts"
                        if self.exp004_wenet_per_window is not None
                        else "ratio"
                    ),
                    "wenet_fixed_manifest_order": (
                        self.exp004_wenet_fixed_manifest_order
                    ),
                }
            )
        path = self.receipt_dir / f"epoch_{self.epoch + 1}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    def __iter__(self) -> Iterable[int]:
        indices = self._indices()
        self._write_receipt(indices)
        width = len(self.rows)
        return iter([self.epoch * width + index for index in indices])


def build_speed_assignments(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    epochs: int,
) -> list[dict[int, str]]:
    """Assign Official rows to exactly balanced speed labels for every epoch."""
    official_indices = [
        index for index, row in enumerate(rows) if source_group(row) == SOURCE_OFFICIAL
    ]
    if len(official_indices) != 6_292:
        raise ValueError(f"Expected 6,292 Official rows, found {len(official_indices)}")
    labels = ("0.9", "1.0", "1.1")
    assignments: list[dict[int, str]] = []
    for epoch in range(epochs):
        ordered = deterministic_permutation(
            official_indices, seed, "speed_assignment", epoch
        )
        assignments.append(
            {index: labels[position % len(labels)] for position, index in enumerate(ordered)}
        )
    return assignments


def speed_assignment_report(assignments: Sequence[Mapping[int, str]]) -> dict[str, Any]:
    return {
        "epochs": [
            {
                "epoch": epoch + 1,
                "counts": dict(sorted(Counter(mapping.values()).items())),
            }
            for epoch, mapping in enumerate(assignments)
        ]
    }
