#!/usr/bin/env python3
"""Collect P2 evidence without promoting partial work to completed results."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cantonese_asr.io import sha256_file
from cantonese_asr.p2_full import atomic_json, select_validation, extension_decision

REQUIRED = ("metrics.json", "top_confusions.csv", "error_examples.json", "generation.json",
            "predictions.jsonl", "generation_tokens.jsonl", "evaluation_receipt.json")


def verified_surface(directory, spec, surface, step):
    """Fail closed: malformed or corrupt surfaces never enter the result matrix."""
    for name in REQUIRED:
        if not (directory/name).is_file():
            raise ValueError(f"missing artifact: {directory/name}")
    row = json.loads((directory/"evaluation_receipt.json").read_text())
    if (not row.get("complete") or not row.get("integrity_pass")
        or row.get("manifest_sha256") != spec["sha256"]
        or row.get("decoder") != "D0_CURRENT" or row.get("surface") != surface
        or row.get("step") != step):
        raise ValueError(f"invalid receipt: {directory}")
    hashes = {f["path"]: f["sha256"] for f in row.get("files", [])}
    for name in REQUIRED[:-1]:
        if name not in hashes or sha256_file(directory/name) != hashes[name]:
            raise ValueError(f"unverified artifact: {directory/name}")
    for name in ("predictions.jsonl", "generation_tokens.jsonl"):
        with (directory/name).open() as handle:
            if sum(bool(line.strip()) for line in handle) != spec["rows"]:
                raise ValueError(f"incorrect row count: {directory/name}")
    return row


def collect(config, run_root):
    rows, missing, decisions = [], [], {}
    for arm in config["arms"]:
        seeds = {}
        for seed in config["seeds"]:
            root = run_root/f"{arm}_S{seed}"/"train"
            checkpoints = []
            for step in config["training"]["save_steps"][:6]:
                path = root/"diagnostics"/f"checkpoint-{step}"/"validation"/"evaluation_receipt.json"
                if not path.is_file():
                    missing.append(str(path))
                    continue
                try:
                    value = verified_surface(path.parent, config["surfaces"]["validation"], "validation", step)
                except (ValueError, OSError) as exc:
                    missing.append(str(exc))
                    continue
                checkpoints.append(value)
            if len(checkpoints) != 6:
                continue
            selected = select_validation(checkpoints)
            seeds[seed] = checkpoints
            for role, checkpoint in (("epoch3", next(r for r in checkpoints if r["step"] == 4371)), ("selected3", selected)):
                step = checkpoint["step"]
                surfaces = {}
                for surface, spec in config["surfaces"].items():
                    directory = root/"diagnostics"/f"checkpoint-{step}"/surface
                    for name in REQUIRED:
                        if not (directory/name).is_file():
                            missing.append(str(directory/name))
                    if all((directory/name).is_file() for name in REQUIRED):
                        try:
                            surfaces[surface] = verified_surface(directory, spec, surface, step)
                        except (ValueError, OSError) as exc:
                            missing.append(str(exc))
                rows.append({"arm": arm, "seed": seed, "role": role, "step": step, "surfaces": surfaces})
        if len(seeds) == 2:
            decisions[arm] = extension_decision(seeds)
    return {"capacity": rows, "extension_decisions": decisions, "missing": missing}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    result = collect(cfg, a.run_root)
    # This collector covers the fixed-budget capacity table only. Final round
    # completion also requires extension, tokenizer, LM/fusion and offline gates.
    result["status"] = "PARTIAL"
    result["pending_final_gates"] = ["step0_surfaces", "conditional_extensions", "tokenizer_matrix",
        "both_neural_seeds", "native_nbest_analysis", "new_capacity_fusion", "offline_research_packages"]
    result["config_sha256"] = sha256_file(a.config)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(a.output_dir/"capacity_matrix.json", result)
    with (a.output_dir/"capacity.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm","seed","role","step","surface","tol2","cer","severe"])
        for row in result["capacity"]:
            for surface, value in row["surfaces"].items():
                writer.writerow([row["arm"],row["seed"],row["role"],row["step"],surface,value["tol2"],value["cer"],value["severe"]])
    lines = ["# Full P2 evidence status", "", "Status: PARTIAL — not a completed experiment report.",
        "", "Shared 7285-step cosine horizon; all arms mandatory 3 epochs, paired conditional extension to 5.",
        "", "| Arm | Seed | Role | Step | Validation tol2 | CER |", "|---|---:|---|---:|---:|---:|"]
    for row in result["capacity"]:
        v = row["surfaces"].get("validation", {})
        lines.append(f"| {row['arm']} | {row['seed']} | {row['role']} | {row['step']} | {v.get('tol2','missing')} | {v.get('cer','missing')} |")
    lines += ["", f"Missing capacity evidence items: {len(result['missing'])}.",
              "No convergence or architecture-winner claim is made from incomplete evidence."]
    (a.output_dir/"report.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
