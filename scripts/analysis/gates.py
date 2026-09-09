#!/usr/bin/env python3
"""Run the pre-registered extension gate over every completed arm, using the
round's own extension_decision rule."""
import json
import os, sys
from pathlib import Path

# Run root: the directory holding outputs/<run>/<arm>_S<seed>/train/... .
# Override with P2_RUN_ROOT, or pass it as the first argument.
R = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("P2_RUN_ROOT", "."))
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from cantonese_asr.p2_full import extension_decision

KEYS = ("step","surface","tol2","cer","severe","repeated_runaway",
        "replacement","max_length","integrity_pass","complete")

def collect(root, arm):
    by = {}
    for seed in (42, 43):
        rows = []
        for st in (2186, 2914, 3643, 4371):
            f = root/f"{arm}_S{seed}"/"train/diagnostics"/f"checkpoint-{st}"/"validation/evaluation_receipt.json"
            if not f.exists():
                return None
            d = json.loads(f.read_text())
            rows.append({k: d[k] for k in KEYS if k in d})
        by[seed] = rows
    return by

roots = {
    "": R/"outputs/p2-full-h100-tf457",
    "SC ": R/"outputs/p2-full-h100-tf457-sc",
    "b4ctl ": R/"outputs/p2-full-h100-b4ctl",
}
print("%-22s %-8s %-12s %s" % ("arm", "extend?", "mean CER gain", "failures"))
print("-" * 78)
for tag, root in roots.items():
    if not root.is_dir():
        continue
    for arm in sorted({d.rsplit("_S", 1)[0] for d in os.listdir(root) if "_S" in d}):
        by = collect(root, arm)
        if by is None:
            continue
        try:
            r = extension_decision(by)
        except Exception as e:
            print("%-22s %-8s %s" % (tag+arm, "ERR", e)); continue
        print("%-22s %-8s %+.6f    %s" % (
            tag+arm, "yes" if r["extend_both_seeds"] else "no",
            r["mean_cer_improvement"], ", ".join(r["failures"]) or "—"))
