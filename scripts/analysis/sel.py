#!/usr/bin/env python3
"""Compare arms at the checkpoint the round's own select_validation picks,
rather than blindly at step 4371."""
import json
import os, sys
from pathlib import Path

# Run root: the directory holding outputs/<run>/<arm>_S<seed>/train/... .
# Override with P2_RUN_ROOT, or pass it as the first argument.
R = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("P2_RUN_ROOT", "."))
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from cantonese_asr.p2_full import select_validation

K = ("step","surface","tol2","cer","severe","repeated_runaway","replacement","max_length","complete")

def rows(root, arm, seed, steps=(729,1457,2186,2914,3643,4371)):
    out = []
    for st in steps:
        f = R/root/f"{arm}_S{seed}"/"train/diagnostics"/f"checkpoint-{st}"/"validation/evaluation_receipt.json"
        if f.exists():
            d = json.loads(f.read_text())
            out.append({k: d[k] for k in K if k in d})
    return out

runs = [
    ("baseline b8", "outputs/p2-full-h100-tf457",    "SMALL_FULL"),
    ("baseline b4", "outputs/p2-full-h100-b4ctl",    "SMALL_FULL"),
    ("SC       b4", "outputs/p2-full-h100-tf457-sc", "SMALL_FULL"),
]
sel = {}
print("%-14s %-6s %-7s %-10s %-10s %-6s %s" % ("run","seed","step","tol2","CER","severe","unstable(repl/runaway/maxlen)"))
print("-"*92)
for name, root, arm in runs:
    for seed in (42, 43):
        rs = rows(root, arm, seed)
        if len(rs) < 4:
            print("%-14s %-6d incomplete (%d points)" % (name, seed, len(rs))); continue
        s = select_validation(rs)
        sel[(name, seed)] = s
        print("%-14s %-6d %-7d %-10.6f %-10.6f %-6d %d/%d/%d" % (
            name, seed, s["step"], s["tol2"], s["cer"], s["severe"],
            s.get("replacement",0), s.get("repeated_runaway",0), s.get("max_length",0)))
print()
print("=== SC vs batch-matched baseline (after selection) ===")
for seed in (42, 43):
    b = sel.get(("baseline b4", seed)); c = sel.get(("SC       b4", seed))
    if b and c:
        print("  seed %d  Δtol2 %+.6f   Δcer %+.6f" % (seed, c["tol2"]-b["tol2"], c["cer"]-b["cer"]))
print("=== batch8 vs batch4 baseline (isolating the batch effect) ===")
for seed in (42, 43):
    a = sel.get(("baseline b8", seed)); b = sel.get(("baseline b4", seed))
    if a and b:
        print("  seed %d  Δtol2 %+.6f   Δcer %+.6f" % (seed, b["tol2"]-a["tol2"], b["cer"]-a["cer"]))
