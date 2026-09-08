#!/usr/bin/env python3
"""Issue a GPU gate only from six real smoke/benchmark runs and verified receipts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import numpy as np
from safetensors.torch import load_file
from cantonese_asr.io import read_jsonl, sha256_file
from cantonese_asr.p2_full import atomic_json, code_fingerprint, rank_microbatches


def compare_tree(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            compare_tree(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            compare_tree(a, b)
    else:
        assert left == right


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arms", nargs="+", help="Verify an explicit subset without claiming full-round acceptance")
    p.add_argument("--evidence-code-root", type=Path, help="Immutable code version that produced these runs")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    config = json.loads(a.config.read_text())
    stream = json.loads((a.preparation/"smoke_s42.json").read_text())
    assert len(stream) == 6 and len(stream[-1]["indices"]) == 16
    stream[-1]["indices"] = stream[-1]["indices"][:8]
    accepted, benchmarks = [], {}
    evidence_root = (a.evidence_code_root or ROOT).resolve()
    fingerprint = code_fingerprint(evidence_root)
    config_sha = sha256_file(a.config)
    arms = a.arms or list(config["arms"])
    if len(set(arms)) != len(arms) or any(arm not in config["arms"] for arm in arms):
        raise ValueError("invalid evidence arm subset")
    for arm in arms:
        recipe = config["arms"][arm]
        root = a.run_root/f"{arm}_S42"
        smoke = root/"smoke"
        resumed = smoke/"resumed"/"checkpoint-6"
        uninterrupted = smoke/"uninterrupted"/"checkpoint-6"
        for checkpoint in (resumed, uninterrupted):
            receipt = json.loads((checkpoint/"COMPLETE.json").read_text())
            assert receipt["phase"] == "smoke" and receipt["step"] == 6
            assert receipt["code_sha256"] == fingerprint and receipt["config_sha256"] == config_sha
            for file in receipt["files"]:
                assert sha256_file(checkpoint/file["path"]) == file["sha256"]
            for rank in range(recipe["world_size"]):
                resource = json.loads((checkpoint/f"resources_rank{rank}.json").read_text())
                assert resource["peak_allocated_bytes"] > 0
                assert resource.get("deterministic_algorithms") is True
                assert resource.get("cudnn_deterministic") is True
                assert resource.get("cudnn_benchmark") is False
                assert resource.get("cublas_workspace_config") == ":4096:8"
                observations = resource["specaugment_forward_observations"]
                assert observations["calls"] > 0 and observations["calls"] == observations["unchanged"]
        weights = sorted((uninterrupted/"model").glob("*.safetensors"))
        assert weights
        for file in weights:
            compare_tree(load_file(file), load_file(resumed/"model"/file.name))
        compare_tree(torch.load(uninterrupted/"optimizer.pt", map_location="cpu", weights_only=False),
                     torch.load(resumed/"optimizer.pt", map_location="cpu", weights_only=False))
        for rank in range(recipe["world_size"]):
            records = read_jsonl(smoke/"uninterrupted"/f"observed_rank{rank}_segment_0_6.jsonl")
            assert len(records) == len(stream) == 6
            assert len(stream[-1]["indices"]) == 8
            for record, expected in zip(records, stream):
                assert record["microbatch_indices"] == [m[rank] for m in rank_microbatches(expected["indices"], recipe["world_size"], recipe["batch"])]
            l = torch.load(uninterrupted/f"rng_rank{rank}.pt", weights_only=False, map_location="cpu")
            r = torch.load(resumed/f"rng_rank{rank}.pt", weights_only=False, map_location="cpu")
            assert torch.equal(l["torch"], r["torch"]) and torch.equal(l["cuda"], r["cuda"])
            compare_tree(l["python"], r["python"])
            compare_tree(l["numpy"], r["numpy"])
        evaluation = json.loads((smoke/"offline32"/"evaluation_receipt.json").read_text())
        assert evaluation["step"] == 6 and evaluation["decoder"] == "D0_CURRENT"
        assert len(read_jsonl(smoke/"offline32"/"predictions.jsonl")) == 32
        benchmark = json.loads((root/"benchmark"/"training"/"checkpoint-50"/"COMPLETE.json").read_text())
        assert benchmark["phase"] == "benchmark" and benchmark["code_sha256"] == fingerprint
        assert benchmark["step"] == 50 and benchmark["config_sha256"] == config_sha
        for file in benchmark["files"]:
            assert sha256_file(root/"benchmark"/"training"/"checkpoint-50"/file["path"]) == file["sha256"]
        benchmarks[arm] = benchmark
        accepted.append(arm)
    atomic_json(a.output, {"status": "PASS", "accepted_arms": accepted,
        "all_six_arms_verified": len(accepted) == len(config["arms"]),
        "evidence_code_root": str(evidence_root), "verifier_sha256": sha256_file(Path(__file__)),
        "code_sha256": fingerprint, "config_sha256": config_sha,
        "resume_comparison": "tensor rtol=1e-5 atol=1e-6; Python/NumPy/torch/CUDA RNG exact; tail8 included",
        "benchmarks": benchmarks,
        "estimated_training_gpu_hours_3epochs": sum(b["training_gpu_hours"]*4371/50*2 for b in benchmarks.values()),
        "estimated_training_gpu_hours_5epochs": sum(b["training_gpu_hours"]*7285/50*2 for b in benchmarks.values())})


if __name__ == "__main__":
    main()
