#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/bolin/cantonese-asr}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}

cd "$PROJECT_DIR"
mkdir -p logs
GRID_JOB=$(sbatch --parsable \
  --export="ALL,BATCH_SIZE=$BATCH_SIZE,GRAD_ACCUM=$GRAD_ACCUM" \
  slurm/grid_search.slurm)
REPORT_JOB=$(sbatch --parsable --dependency="afterany:$GRID_JOB" slurm/report_all.slurm)

echo "grid_job=$GRID_JOB"
echo "report_job=$REPORT_JOB"

