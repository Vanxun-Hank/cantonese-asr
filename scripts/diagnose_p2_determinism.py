#!/usr/bin/env python3
"""Bounded diagnostic only: deterministic CUDA around an unchanged trainer.

This cannot issue acceptance or launch formal training. The underlying script's
hash remains the old one; this wrapper and its execution settings are separately
recorded to avoid mislabelling the diagnostic as the original recipe execution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--training-script", type=Path, required=True)
a, rest = p.parse_known_args()
if "--phase" not in rest or rest[rest.index("--phase") + 1] != "smoke":
    raise ValueError("only diagnostic smoke is allowed")
if "--until-step" not in rest or int(rest[rest.index("--until-step") + 1]) > 3 or "--resume" in rest:
    raise ValueError("diagnostic limited to independent first three steps")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(json.dumps({"diagnostic": "deterministic_cuda_first3",
    "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "trainer_sha256": hashlib.sha256(a.training_script.read_bytes()).hexdigest(),
    "cublas_workspace": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    "deterministic_algorithms": True, "cudnn_benchmark": False,
    "bf16_tf32": "unchanged by wrapper", "argv": rest}), flush=True)
sys.argv = [str(a.training_script), *rest]
runpy.run_path(str(a.training_script), run_name="__main__")
