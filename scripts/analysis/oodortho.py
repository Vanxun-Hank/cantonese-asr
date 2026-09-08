#!/usr/bin/env python3
"""Quantify how much of a surface's recorded edit distance is traditional/simplified
orthography rather than recognition error.

The official scorer converts predictions to simplified but deliberately leaves
references alone (see cantonese_asr/metrics.py). That is correct when references are
already simplified. Panels whose references are traditional -- the OOD panel is built
from Common Voice zh-HK -- are therefore scored on orthography as much as on speech.
This rescores with both sides converted, and reports the difference.

Usage:  P2_RUN_ROOT=<dir> python scripts/analysis/oodortho.py [run_root]
"""
import json
import os, sys
from pathlib import Path

R = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("P2_RUN_ROOT", "."))
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from cantonese_asr.metrics import to_simplified

ARMS = ("LARGE_V2_FULL", "LARGE_V2_LORA", "MEDIUM_FULL", "SMALL_FULL")


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


print("%-16s %-6s %-5s %-10s %-10s %-10s %-10s %s" % (
    "arm", "surface", "seed", "tol2", "tol2 norm", "CER", "CER norm", "share of edits"))
print("-" * 96)
for surface in ("ood", "public"):
    for arm in ARMS:
        for seed in (42, 43):
            f = R / "outputs/surfaces" / f"{arm}_S{seed}" / surface / "sample_metrics.jsonl"
            if not f.exists():
                continue
            rows = [json.loads(line) for line in f.open(encoding="utf-8")]
            chars = sum(len(r["normalized_reference"]) for r in rows)
            raw = sum(r["edit_distance"] for r in rows)
            tol2 = sum(1 for r in rows if r["edit_distance"] <= 2)
            norm = norm_tol2 = 0
            for r in rows:
                d = levenshtein(to_simplified(r["normalized_reference"]),
                                to_simplified(r["normalized_prediction"]))
                norm += d
                norm_tol2 += d <= 2
            print("%-16s %-6s %-5d %-10.6f %-10.6f %-10.6f %-10.6f %.1f%%" % (
                arm, surface, seed, tol2 / len(rows), norm_tol2 / len(rows),
                raw / chars, norm / chars, (raw - norm) / raw * 100))
