#!/usr/bin/env python3
"""Run one allocated arm sequentially; release trainer memory before evaluation."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cantonese_asr.p2_full import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--asset-root", type=Path, required=True)
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--seed", type=int, choices=(42,43), required=True)
    p.add_argument("--phase", choices=("smoke", "benchmark", "train", "extend"), required=True)
    p.add_argument("--acceptance", type=Path)
    p.add_argument("--extension-receipt", type=Path)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    arm = cfg["arms"][a.arm]
    common = ["--config", str(a.config.resolve()), "--asset-root", str(a.asset_root.resolve())]
    output = a.run_root / f"{a.arm}_S{a.seed}" / a.phase
    if output.exists():
        raise FileExistsError("worker attempt already exists; do not silently retry")
    output.mkdir(parents=True)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false"}
    model = a.asset_root / cfg["models"][arm["model"]]
    def run(command):
        with (output / "commands.jsonl").open("a") as handle:
            handle.write(json.dumps(command) + "\n")
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    def train(until, directory, resume=None):
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   f"--nproc_per_node={arm['world_size']}", str(ROOT/"scripts/train_p2_full.py"),
                   *common, "--preparation", str(a.preparation.resolve()), "--output-dir", str(directory),
                   "--arm", a.arm, "--seed", str(a.seed), "--until-step", str(until),
                   "--phase", "train" if a.phase == "extend" else a.phase]
        if resume:
            command += ["--resume", str(resume)]
        if a.acceptance:
            command += ["--acceptance", str(a.acceptance.resolve())]
        if a.extension_receipt:
            command += ["--extension-receipt", str(a.extension_receipt.resolve())]
        run(command)
        return directory/f"checkpoint-{until}"
    def evaluate(checkpoint, step, destination, maximum=None):
        command = [sys.executable, str(ROOT/"scripts/evaluate_p2_full.py"), *common,
                   "--model-dir", str(checkpoint), "--processor-dir", str(model),
                   "--output-dir", str(destination), "--surface", "validation", "--step", str(step)]
        if maximum:
            command += ["--max-samples", str(maximum)]
        run(command)
    if a.phase == "smoke":
        first = train(5, output/"interrupted")
        restored = train(6, output/"resumed", first)
        uninterrupted = train(6, output/"uninterrupted")
        evaluate(restored/"model", 6, output/"offline32", 32)
        # This receipt is evidence, not automatic acceptance. Actual checkpoint
        # equality and all-rank GPU gates must pass before an acceptance is issued.
        atomic_json(output/"SMOKE_EVIDENCE.json", {"status": "AWAITING_PARITY_VERIFICATION",
                    "resumed": str(restored), "uninterrupted": str(uninterrupted)})
    elif a.phase == "benchmark":
        train(50, output/"training")
    else:
        resume = None
        milestones = cfg["training"]["save_steps"]
        if a.phase == "extend":
            resume = a.run_root/f"{a.arm}_S{a.seed}"/"train"/"training"/"checkpoint-4371"
            milestones = [step for step in milestones if step > 4371]
        else:
            milestones = [step for step in milestones if step <= 4371]
        for step in milestones:
            resume = train(step, output/"training", resume)
            evaluate(resume/"model", step, output/"diagnostics"/f"checkpoint-{step}"/"validation")
    atomic_json(output/"WORKER_COMPLETE.json", {"arm": a.arm, "seed": a.seed, "phase": a.phase})


if __name__ == "__main__":
    main()
