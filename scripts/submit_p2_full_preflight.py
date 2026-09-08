#!/usr/bin/env python3
"""Submit only the six gated preflight paths on one authorized node.

No formal training, retries, cancellation, or modification of unrelated jobs.
Each successful submission is persisted before attempting the next one.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cantonese_asr.p2_full import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset-root", type=Path, required=True)
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--prepare-job", type=int, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--node", choices=("gpu001",), default="gpu001")
    p.add_argument("--arms", nargs="+", help="Explicit bounded retry subset; existing evidence remains untouched")
    a = p.parse_args()
    if a.receipt.exists():
        raise FileExistsError("submission receipt exists; inspect it instead of duplicating jobs")
    config = json.loads((ROOT/"configs/rounds/raw_winner_p2_full.json").read_text())
    subprocess.run(["squeue", "-w", a.node], check=True)
    subprocess.run(["scontrol", "show", "node", a.node], check=True)
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"status": "SUBMITTING_PREFLIGHT_ONLY", "node": a.node,
               "prepare_job": a.prepare_job, "jobs": []}
    atomic_json(a.receipt, receipt)
    arms = a.arms or ["MEDIUM_FULL", "LARGE_V2_LORA", "MEDIUM_LORA", "LARGE_V2_FULL", "SMALL_FULL", "SMALL_LORA"]
    if len(set(arms)) != len(arms) or any(arm not in config["arms"] for arm in arms):
        raise ValueError("invalid or duplicate retry arms")
    for arm in arms:
        dependency = a.prepare_job
        for phase in ("smoke", "benchmark"):
            environment = {"P2_CODE_ROOT": str(ROOT), "P2_ASSET_ROOT": str(a.asset_root),
                "P2_PREPARATION": str(a.preparation), "P2_RUN_ROOT": str(a.run_root),
                "P2_ARM": arm, "P2_SEED": "42", "P2_PHASE": phase}
            command = ["sbatch", "--parsable", f"--nodelist={a.node}",
                f"--gres=gpu:rtx:{config['arms'][arm]['world_size']}",
                "--time=04:00:00", f"--dependency=afterok:{dependency}",
                "--kill-on-invalid-dep=yes", "--no-requeue",
                f"--job-name=p2-{phase}-{arm}",
                f"--output={a.receipt.parent}/{phase}-{arm}-%j.log",
                "--export=ALL," + ",".join(f"{key}={value}" for key,value in environment.items()),
                str(ROOT/"slurm/p2_full_capacity.slurm")]
            job = int(subprocess.check_output(command, text=True).strip().split(";")[0])
            receipt["jobs"].append({"arm": arm, "phase": phase, "job": job, "dependency": dependency,
                                    "command": command})
            atomic_json(a.receipt, receipt)
            print(json.dumps(receipt["jobs"][-1]), flush=True)
            dependency = job
    receipt["status"] = "PREFLIGHT_SUBMITTED_NO_FORMAL_TRAINING"
    atomic_json(a.receipt, receipt)


if __name__ == "__main__":
    main()
