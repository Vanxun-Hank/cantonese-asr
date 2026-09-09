"""Dependency-light invariants for the approved three-to-five epoch P2 round."""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Iterable


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def code_fingerprint(root: Path) -> str:
    """Fingerprint the training/evaluation execution closure, not later reports.

    Analysis and packaging keep their own provenance; editing a report must not
    invalidate a byte-identical training resume. Shared Python modules remain
    covered conservatively.
    """
    paths = [root / "train.py", root / "predict.py"]
    paths += sorted((root / "cantonese_asr").glob("*.py"))
    paths += [root / "scripts" / name for name in (
        "train_p2_full.py", "p2_full_capacity_worker.py", "evaluate_p2_full.py",
        "evaluate_raw_winner_decode.py", "verify_p2_full_smoke.py")]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def epoch_batches(rows: int, seed: int, epochs: int = 5, batch: int = 16) -> list[dict]:
    if rows <= 0 or epochs <= 0 or batch <= 0:
        raise ValueError("positive dimensions required")
    result = []
    for epoch in range(epochs):
        indices = list(range(rows))
        random.Random(f"p2-full:{seed}:epoch:{epoch}").shuffle(indices)
        for offset in range(0, rows, batch):
            result.append({"step": len(result) + 1, "epoch": epoch + 1,
                           "offset": offset, "indices": indices[offset:offset + batch]})
    return result


def rank_microbatches(indices: list[int], world_size: int, batch: int) -> list[list[list[int]]]:
    """Return [microstep][rank][row]; tail is never padded or repeated."""
    width = world_size * batch
    if world_size < 1 or batch < 1 or len(indices) % width:
        raise ValueError("tail cannot be evenly partitioned by this topology")
    return [[indices[start + rank * batch:start + (rank + 1) * batch]
             for rank in range(world_size)] for start in range(0, len(indices), width)]


def cosine_factor(step: int, warmup: int = 365, horizon: int = 7285) -> float:
    if step < 0 or not 0 < warmup < horizon or step > horizon:
        raise ValueError("invalid absolute scheduler position")
    if step < warmup:
        return step / warmup
    return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / (horizon - warmup)))


def select_validation(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    if not rows:
        raise ValueError("no validation candidates")
    for row in rows:
        if row.get("surface") != "validation" or row.get("complete") is not True:
            raise ValueError("only complete validation evidence may select checkpoints")
        for key in ("tol2", "cer", "severe", "step"):
            if key not in row or not math.isfinite(float(row[key])):
                raise ValueError(f"missing/nonfinite {key}")
    peak = max(row["tol2"] for row in rows)
    band = [row for row in rows if peak - row["tol2"] <= 0.005 + 1e-12]
    return min(band, key=lambda r: (r["cer"], -r["tol2"], r["severe"], r["step"]))


def extension_decision(by_seed: dict[int, list[dict]]) -> dict:
    if set(by_seed) != {42, 43}:
        raise ValueError("both seeds 42 and 43 are mandatory")
    pairs = {}
    failures = []
    improvements = []
    for seed, rows in sorted(by_seed.items()):
        for step in (2186, 2914, 3643, 4371):
            if sum(r.get("step") == step for r in rows) != 1:
                raise ValueError(f"seed {seed}: expected exactly one checkpoint {step}")
        early = select_validation(r for r in rows if r["step"] in (2186, 2914))
        late = select_validation(r for r in rows if r["step"] in (3643, 4371))
        improvements.append(early["cer"] - late["cer"])
        if late["cer"] > early["cer"] + 1e-12:
            failures.append(f"seed{seed}:cer_regression")
        if early["tol2"] - late["tol2"] > 0.005 + 1e-12:
            failures.append(f"seed{seed}:tol2_regression")
        for key in ("severe", "repeated_runaway", "replacement", "max_length"):
            if key not in early or key not in late:
                raise ValueError(f"missing stability evidence: {key}")
            if late[key] > early[key]:
                failures.append(f"seed{seed}:{key}_regression")
        if any(r.get("integrity_pass") is not True for r in rows):
            failures.append(f"seed{seed}:integrity_not_passed")
        pairs[str(seed)] = {"early": early, "late": late}
    mean_improvement = sum(improvements) / 2
    if mean_improvement < 0.001 - 1e-12:
        failures.append("mean_cer_improvement_below_0.001")
    return {"extend_both_seeds": not failures, "mean_cer_improvement": mean_improvement,
            "failures": failures, "pairs": pairs, "selection_surface": "validation"}


def split_lm_texts(texts: list[str], seed: int = 42, dev_fraction: float = 0.05) -> dict:
    """Input texts must already use the frozen project normalizer."""
    groups = sorted(set(texts))
    if len(groups) < 2:
        raise ValueError("at least two distinct training texts required")
    random.Random(seed).shuffle(groups)
    count = min(len(groups) - 1, max(1, round(len(groups) * dev_fraction)))
    dev_groups = set(groups[:count])
    train = [text for text in texts if text not in dev_groups]
    dev = [text for text in texts if text in dev_groups]
    assert not set(train) & set(dev)
    digest = hashlib.sha256(json.dumps({"train": train, "dev": dev}, ensure_ascii=False).encode()).hexdigest()
    return {"train": train, "dev": dev, "seed": seed, "split_sha256": digest,
            "train_rows": len(train), "dev_rows": len(dev), "distinct_groups": len(groups)}


def exposure_counts(batches: list[dict], sources: list[str]) -> dict:
    return dict(Counter(sources[index] for batch in batches for index in batch["indices"]))
