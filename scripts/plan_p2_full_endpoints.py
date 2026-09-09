#!/usr/bin/env python3
"""Validation-only epoch3 endpoint/selected evaluation plan; never submits jobs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cantonese_asr.io import sha256_file
from cantonese_asr.p2_full import atomic_json, select_validation, extension_decision


def build_plan(config, config_sha, run_root):
    tasks, selections, decisions, missing = [], {}, {}, []
    for arm in config["arms"]:
        paired = {}
        for seed in config["seeds"]:
            root = run_root / f"{arm}_S{seed}" / "train"
            evidence = []
            for step in config["training"]["save_steps"][:6]:
                path = root / "diagnostics" / f"checkpoint-{step}" / "validation" / "evaluation_receipt.json"
                if not path.exists():
                    missing.append(str(path))
                    continue
                row = json.loads(path.read_text())
                if (row.get("step") != step or row.get("surface") != "validation"
                    or not row.get("complete") or not row.get("integrity_pass")
                    or row.get("decoder") != "D0_CURRENT" or row.get("config_sha256") != config_sha
                    or row.get("manifest_sha256") != config["surfaces"]["validation"]["sha256"]):
                    raise ValueError(f"invalid selection evidence: {path}")
                for file in row["files"]:
                    if sha256_file(path.parent/file["path"]) != file["sha256"]:
                        raise ValueError(f"corrupt surface: {path}")
                for name in ("predictions.jsonl", "generation_tokens.jsonl"):
                    if sum(bool(line.strip()) for line in (path.parent/name).open()) != 702:
                        raise ValueError(f"incorrect row count: {path}")
                evidence.append(row)
            if len(evidence) != 6:
                continue
            paired[seed] = evidence
            selected = select_validation(evidence)
            selections[f"{arm}_S{seed}"] = selected
            # Deduplicate selected==endpoint, but never omit a non-winning arm.
            for step in sorted({4371, selected["step"]}):
                checkpoint = root/"training"/f"checkpoint-{step}"
                receipt = json.loads((checkpoint/"COMPLETE.json").read_text())
                if receipt["step"] != step or receipt["seed"] != seed or receipt["arm"] != arm or receipt["phase"] != "train":
                    raise ValueError(f"wrong checkpoint identity: {checkpoint}")
                for surface in ("public", "ood"):
                    tasks.append({"arm": arm, "seed": seed, "step": step, "surface": surface,
                        "roles": [role for role, value in (("epoch3",4371),("selected3",selected["step"])) if value == step],
                        "model_dir": str(checkpoint/"model"),
                        "output_dir": str(root/"diagnostics"/f"checkpoint-{step}"/surface)})
        if len(paired) == 2:
            decisions[arm] = {"arm": arm, **extension_decision(paired)}
    return {"status": "READY" if not missing else "PARTIAL", "selection_surface": "validation",
        "config_sha256": config_sha, "selected3": selections, "extension_decisions": decisions,
        "evaluation_tasks": tasks, "missing": missing,
        "extension_submission_allowed": False,
        "extension_note": "Complete all twelve epoch3 runs first; this plan does not submit extensions."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    result = build_plan(json.loads(a.config.read_text()), sha256_file(a.config), a.run_root)
    atomic_json(a.output, result)
    print(json.dumps({"status": result["status"], "tasks": len(result["evaluation_tasks"]), "missing": len(result["missing"])}))


if __name__ == "__main__":
    main()
