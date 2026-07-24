# Cantonese ASR agent guide

This repository implements the 点心杯 Cantonese ASR pipeline with
`openai/whisper-small` Full SFT. The stack is Python, PyTorch, Transformers, and
Slurm. Start with [README.md](README.md); use [docs/RUNBOOK.md](docs/RUNBOOK.md)
for operational procedures.

## Repository and runtime boundaries

- The Mac project root `/Users/zhangxun/Desktop/service/cantonese-asr` holds
  source, tests, configurations, lightweight reports, and documentation.
- The server root `/home/bolin/cantonese-asr` holds full datasets, base models,
  checkpoints, outputs, logs, and other heavy artifacts. Do not copy those
  assets to the Mac.
- Sync code with `scripts/sync_server.sh`, whose exclusions protect server
  artifacts. From an isolated worktree, set `LOCAL_DIR` to that worktree
  explicitly before syncing.
- The canonical server environment is
  `/home/bolin/envs/cantonese-asr-whisper`.
- GPU jobs source `slurm/common.sh` and run with Hugging Face, Transformers, and
  Datasets offline. Do not add runtime downloads to GPU jobs.

## Data, training, and selection invariants

- The SFT target is Cantonese transcription text. Preserve source, license,
  publisher split, hashes, and split-isolation evidence for every dataset.
- Keep the Whisper-small architecture and tokenizer fixed. Training,
  validation, and submission generation use a maximum length of 225.
- Fixed official validation is the only checkpoint and model selection
  surface. `train_probe`, OOD data, public samples, and platform results are
  diagnostics only.
- Fixed validation, public exclusions, and OOD manifests must never enter
  training. Update overlap and deterministic-sampling tests when data behavior
  changes.
- Use fixed manifests, seeds, and resolved configurations as experiment
  evidence. Write each run to a new run-specific output directory; never
  overwrite manifests, reports, checkpoints, or submission artifacts.
- An offline submission is flat at the ZIP root, contains exactly one
  `model.safetensors`, and must not depend on server paths or network access.
- Round 3 design decisions are documented in
  [the MDCC coverage design](docs/superpowers/specs/2026-07-23-round3-mdcc-coverage-design.md).

## Verification

Run focused tests while iterating, then the complete suite in the canonical
server environment:

```bash
python -m pytest -q tests/test_build_external_training_mixes.py
python -m pytest -q tests/test_round3_slurm.py
/home/bolin/envs/cantonese-asr-whisper/bin/python -m pytest -q
python3 -m compileall -q cantonese_asr scripts train.py predict.py
bash -n scripts/*.sh slurm/*.sh slurm/*.slurm
```

Add or update tests whenever observable behavior changes. Before training,
verify generated manifests and their reports; before delivery, run the offline
submission checks described in [docs/RUNBOOK.md](docs/RUNBOOK.md).

## Workflow map

- Data preparation and QC: `scripts/prepare_manifest.py` and
  `scripts/build_external_training_mixes.py`.
- Round 3 preparation and four independent trials:
  `slurm/prepare_external_round3.slurm` and `slurm/external_round3.slurm`.
- Reusable reports: `scripts/plot_experiments.py` and
  `slurm/report_external_round3.slurm`.
- Validation-only global selection, full OOD diagnostics, packaging, and
  offline checks: `slurm/final_candidate_full_ood.slurm`.
- Follow [docs/RUNBOOK.md](docs/RUNBOOK.md) for command order and acceptance
  gates rather than duplicating procedures here.

## Slurm and workspace safety

- Inspect `squeue` and `sacct` before any scheduler mutation. Never cancel,
  hold, release, reprioritize, or modify unrelated jobs without explicit user
  authorization.
- If the user authorizes holding competing jobs, record their job IDs, names,
  and owners; prefer reversible `scontrol hold`, and report what must later be
  released.
- Do not assume the login node exposes `nvidia-smi`; verify GPUs inside Slurm
  jobs.
- Preserve unrelated dirty files and changes. Do not remove or rewrite user
  work to obtain a clean tree.
- Keep failures and partial evidence in their run directories. Fix or rerun
  only the affected trial unless the shared input or code is proven invalid.
