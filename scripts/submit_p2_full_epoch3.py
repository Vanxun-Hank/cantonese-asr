#!/usr/bin/env python3
"""Bounded formal submission: three base evaluations and twelve epoch3 arms."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--code-root", type=Path, required=True)
p.add_argument("--asset-root", type=Path, required=True)
p.add_argument("--receipt", type=Path, required=True)
a = p.parse_args()
sys.path.insert(0, str(a.code_root))
from cantonese_asr.p2_full import atomic_json, code_fingerprint
from cantonese_asr.io import sha256_file
b = a.asset_root
config_path = a.code_root / "configs/rounds/raw_winner_p2_full.json"
cfg = json.loads(config_path.read_text())
gate_path = b / "artifacts/raw_winner_p2_full_v1/launch/deterministic1/gpu_acceptance.json"
gate = json.loads(gate_path.read_text())
assert gate["status"] == "PASS" and gate["all_six_arms_verified"]
assert set(gate["accepted_arms"]) == set(cfg["arms"])
assert gate["code_sha256"] == code_fingerprint(a.code_root)
assert gate["config_sha256"] == sha256_file(config_path)
if a.receipt.exists():
    raise FileExistsError("inspect existing submission instead of duplicating")
subprocess.run(["squeue", "-w", "gpu001"], check=True)
subprocess.run(["scontrol", "show", "node", "gpu001"], check=True)
py = "/home/bolin/envs/cantonese-asr-whisper/bin/python"
out = b / "outputs/raw-winner-p2-full-v1-deterministic1"
a.receipt.parent.mkdir(parents=True, exist_ok=True)
r = {"status": "SUBMITTING", "acceptance_sha256": sha256_file(gate_path),
     "code_sha256": gate["code_sha256"], "maximum_formal_step": 4371, "jobs": []}
atomic_json(a.receipt, r)
def submit(args, record):
    job = int(subprocess.check_output(["sbatch", "--parsable", "--partition=gpu", "--nodelist=gpu001",
        "--nodes=1", "--ntasks=1", "--no-requeue", "--kill-on-invalid-dep=yes", *args], text=True).strip().split(";")[0])
    r["jobs"].append({**record, "job": job})
    atomic_json(a.receipt, r)
    print(json.dumps(r["jobs"][-1]), flush=True)
    return job
base_jobs = {}
for name, model in cfg["models"].items():
    commands = []
    for surface in cfg["surfaces"]:
        commands.append(shlex.join([py, str(a.code_root/"scripts/evaluate_p2_full.py"),
            "--config", str(config_path), "--asset-root", str(b), "--model-dir", str(b/model),
            "--processor-dir", str(b/model), "--surface", surface, "--step", "0",
            "--output-dir", str(out/"step0"/name/surface)]))
    base_jobs[name] = submit(["--gres=gpu:rtx:1", "--cpus-per-task=4", "--mem=32G", "--time=12:00:00",
        "--job-name=p2-step0-"+name, "--output="+str(a.receipt.parent/f"step0-{name}-%j.log"),
        "--export=ALL,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,OMP_NUM_THREADS=1",
        "--wrap", "set -e; " + " && ".join(commands)], {"phase": "step0_all_surfaces", "model": name})
for arm in ("MEDIUM_FULL", "LARGE_V2_LORA", "SMALL_FULL", "SMALL_LORA", "MEDIUM_LORA", "LARGE_V2_FULL"):
    for seed in (42, 43):
        recipe = cfg["arms"][arm]
        env = {"P2_CODE_ROOT":str(a.code_root), "P2_ASSET_ROOT":str(b),
            "P2_PREPARATION":str(b/"artifacts/raw_winner_p2_full_v1/preparation"),
            "P2_RUN_ROOT":str(out), "P2_ARM":arm, "P2_SEED":str(seed), "P2_PHASE":"train",
            "P2_ACCEPTANCE":str(gate_path)}
        submit([f"--gres=gpu:rtx:{recipe['world_size']}",
            f"--dependency=afterok:{base_jobs[recipe['model']]}",
            "--job-name="+f"p2-e3-{arm}-s{seed}",
            "--output="+str(a.receipt.parent/f"train-{arm}-s{seed}-%j.log"),
            "--export=ALL,"+",".join(f"{k}={v}" for k,v in env.items()),
            str(a.code_root/"slurm/p2_full_capacity.slurm")],
            {"phase":"epoch3", "arm":arm, "seed":seed, "step0_dependency":base_jobs[recipe['model']]})
r["status"] = "EPOCH3_SUBMITTED_NO_EXTENSION"
atomic_json(a.receipt, r)
