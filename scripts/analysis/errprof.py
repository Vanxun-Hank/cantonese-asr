#!/usr/bin/env python3
"""Residual-error profile for LARGE_V2_FULL: composition, concentration,
suspected audio/text misalignment, and overlap with SMALL_FULL."""
import json
import os, sys
from pathlib import Path

# Run root: the directory holding outputs/<run>/<arm>_S<seed>/train/... .
# Override with P2_RUN_ROOT, or pass it as the first argument.
R = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("P2_RUN_ROOT", "."))
REPO = Path(__file__).resolve().parents[2]

def load(run, arm, seed, step):
    f = R/f"outputs/{run}/{arm}_S{seed}/train/diagnostics/checkpoint-{step}/validation/sample_metrics.jsonl"
    return {r["audio_path"]: r for r in map(json.loads, f.open(encoding="utf-8"))}

big = load("p2-full-h100-tf457", "LARGE_V2_FULL", 42, 4371)
sml = load("p2-full-h100-tf457", "SMALL_FULL",    42, 4371)

tot   = sum(r["edit_distance"] for r in big.values())
chars = sum(len(r["normalized_reference"]) for r in big.values())
rows  = sorted(big.values(), key=lambda r: -r["edit_distance"])
print(f"LARGE_V2_FULL seed42 @4371   edits {tot}  chars {chars}  CER {tot/chars:.6f}")
print(f"exactly correct {sum(1 for r in big.values() if r['edit_distance']==0)}/{len(big)} utterances")
S = sum(r["substitutions"] for r in big.values())
D = sum(r["deletions"] for r in big.values())
I = sum(r["insertions"] for r in big.values())
print(f"composition: sub {S} ({S/tot*100:.1f}%)  del {D} ({D/tot*100:.1f}%)  ins {I} ({I/tot*100:.1f}%)")
for n in (5, 10, 20, 50):
    s = sum(r["edit_distance"] for r in rows[:n])
    print(f"  worst {n:3d} hold {s/tot*100:5.1f}% of all edits   ({n/len(big)*100:4.1f}% of utterances)")

print("\n=== suspected audio/text misalignment (edit distance >= 80% of reference) ===")
mis = [r for r in rows if r["edit_distance"] >= 0.8*max(len(r["normalized_reference"]),1)]
print(f"  {len(mis)} utterances, {sum(r['edit_distance'] for r in mis)/tot*100:.1f}% of all edits")
for r in mis[:5]:
    print(f"   ed={r['edit_distance']:<4d} runaway={r['runaway']}")
    print(f"     ref={r['normalized_reference'][:30]}")
    print(f"     hyp={r['normalized_prediction'][:30]}")

both   = [u for u in big if big[u]["edit_distance"]>0 and sml[u]["edit_distance"]>0]
only_s = [u for u in big if big[u]["edit_distance"]==0 and sml[u]["edit_distance"]>0]
only_b = [u for u in big if big[u]["edit_distance"]>0 and sml[u]["edit_distance"]==0]
print(f"\n=== against SMALL_FULL ===")
print(f"  wrong in both {len(both)}   only small wrong (fixed by capacity) {len(only_s)}   only large wrong (introduced) {len(only_b)}")
print(f"  jointly-wrong utterances hold {sum(big[u]['edit_distance'] for u in both)/tot*100:.1f}% of large's edit mass  <- capacity cannot reach these")
