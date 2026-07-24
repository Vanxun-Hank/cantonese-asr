#!/usr/bin/env bash
#SBATCH --job-name=canto-preflight
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

REPORT_DIR=artifacts/reports/deployment
mkdir -p "$REPORT_DIR"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv \
  > "$REPORT_DIR/gpu-${SLURM_JOB_ID}.csv"

"$PYTHON" - <<'PY' > "$REPORT_DIR/gpu-preflight-${SLURM_JOB_ID}.json"
import json
import os
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the allocated GPU job")
x = torch.arange(1024, device="cuda", dtype=torch.float32)
y = (x.square().sum()).item()
print(json.dumps({
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "cuda_available": True,
    "gpu_count_visible": torch.cuda.device_count(),
    "gpu_name": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "tensor_check": y,
}, indent=2))
PY

