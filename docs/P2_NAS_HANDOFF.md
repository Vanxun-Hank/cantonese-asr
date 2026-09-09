# P2 algorithm handoff for NAS

## Purpose

This directory is the source-code handoff for the full P2 Cantonese Whisper
experiments. It contains the reproducible training, evaluation, text modelling,
fusion, scheduling, verification, and finalization code. It intentionally does
not contain model weights, datasets, experiment outputs, credentials, caches, or
Python virtual environments.

## Provenance and current status

- Git repository: `Vanxun-Hank/cantonese-asr`
- Base commit: `beb2396808dffdf2ffe275fd14a7777b9f99f550`
- Working branch: `codex/raw-winner-p2-full`
- P2 round: `raw_winner_p2_full_v1`
- Recipe: `p2_full_four_gpu_v2`
- Capacity budget: every family/seed first runs three epochs; a family may extend
  both seeds to five epochs only under the preregistered validation-only rule.
- The handoff includes uncommitted P2 work newer than the GitHub base commit.
- Focused local P2 test suite at packaging time: 28 passed.

The server-side accepted training deployment was frozen at:

```text
/home/bolin/cantonese-asr/worktrees/raw-winner-p2-full-v1-deterministic1
```

The server became unreachable before this handoff was made. Therefore this
package is the latest Mac-side implementation, but its files have not yet been
compared byte-for-byte with the final frozen server deployment. When the server
returns, compare hashes before treating this package as the exact archival copy
of a running job. This caveat concerns archival provenance, not the availability
of the source implementation in this package.

## Included code surfaces

- `train.py`, `predict.py`, and `cantonese_asr/`: model, data, metrics, sampling,
  P2 training mechanics, corrected n-gram/ROVER logic, and neural LM code.
- `configs/rounds/raw_winner_p2_full.json`: frozen P2 matrix and selection rules.
- `scripts/`: asset preparation, deterministic acceptance, capacity training,
  endpoint evaluation, tokenizer/LM/fusion analysis, selection, and finalization.
- `slurm/`: four-GPU preparation and capacity scheduling entrypoints.
- `tests/`: unit and regression tests for tails, topology, checkpoint recovery,
  native 5-best, LM boundaries, ROVER voting, endpoint planning, and finalizer
  integrity.
- `docs/P2_FULL_V2_EXECUTION.md`: detailed execution record and job provenance.
- requirements files, README files, paper/report source, and license.

## Required external assets

These are deliberately referenced by path and SHA rather than copied into the
algorithm package:

```text
Training manifest:
artifacts/manifests/external/round2/external73.jsonl
SHA256 2ba7edef6811b0d3e96b3347aa70f4237ec27735652d3d313bce62525f1cdea6
Rows 23304 = Official 6292 + CV zh-HK 8451 + MDCC base 8561

Validation manifest:
artifacts/manifests/validation.jsonl
SHA256 8c2310529ae8d257cb9612dc85797aff5952a5b2b6078cd146db486daf9eee4a
Rows 702

Public manifest:
artifacts/datasets/official/template_pre.jsonl
SHA256 95bb1f05b648c6a2ec2e3b23b6b725a02143da0259804af631f5afa6a4367c50
Rows 1900

OOD manifest:
artifacts/manifests/ood/v1/ood_panel.jsonl
SHA256 1ab5f1547bc7c93713e99bfb749e1359ed97e98dd8258bf448656da53abb6450
Rows 2000

Model roots:
artifacts/models/whisper-small
artifacts/models/whisper-medium
artifacts/models/whisper-large-v2
```

Audio files referenced by the manifests must be restored separately. Do not
substitute manifests, reorder samples, or download models inside GPU jobs.

## Primary entrypoints

```text
scripts/run_p2_full.py
scripts/submit_p2_full_preflight.py
scripts/submit_p2_full_epoch3.py
scripts/train_p2_full.py
scripts/p2_full_capacity_worker.py
scripts/evaluate_p2_full.py
scripts/plan_p2_full_endpoints.py
scripts/run_p2_endpoint_task.py
scripts/train_p2_character_lm.py
scripts/analyze_p2_full_text.py
scripts/finalize_p2_full.py
```

Read `docs/P2_FULL_V2_EXECUTION.md` before running. Do not infer that a submitted
job completed, do not duplicate existing Slurm jobs, and do not bypass the GPU
acceptance receipt. The formal training deployment is immutable; analysis and
finalizer fixes must be deployed under a new versioned path.

## Verification

`SHA256SUMS` in the delivery root covers every included file except itself. From
the delivery directory, verify with:

```bash
shasum -a 256 -c SHA256SUMS
```

### Install dependencies first

The test suite is not dependency-free. On a bare interpreter, 12 of the 29 test
files abort at **collection** time with `ModuleNotFoundError`; this is a missing
environment, not a defect in this package. Install before running anything:

```bash
python -m pip install -r requirements.txt          # does NOT include torch
python -m pip install -r requirements-server-torch.txt   # torch==2.6.0 (cu124 index)
```

`torch` is deliberately absent from `requirements.txt` because it is pinned to a
CUDA-specific index in `requirements-server-torch.txt`. Installing only the first
file leaves every torch test uncollectable.

### What runs without which dependency

| Requires | Test files | Notes |
| --- | ---: | --- |
| nothing beyond `pytest` | 17 files, 65 tests | metrics, io, manifests, P2 analysis, endpoint plan, finalizer integrity, w500 policy/selection |
| `torch` | 8 files | `test_p2_checkpoint_creation`, `test_p2_full_torch`, `test_p2_native_candidates`, `test_raw_winner_p2`, `test_training_sampling`, `test_w500_adaptive_interpolation`, `test_w500_adaptive_training`, `test_w500_candidate_losses` |
| `soundfile` | 3 files | `test_ood_manifest`, `test_prepare_manifest`, `test_w500_adaptive_preparation` |
| `librosa` | 2 files | `test_inference_and_selection`, `test_raw_winner_p1_decode` |

CPU-only smoke check on a machine without torch or audio libraries:

```bash
python -m pip install pytest
python -m pytest -q \
  tests/test_metrics.py tests/test_io.py tests/test_p2_analysis.py \
  tests/test_p2_full.py tests/test_p2_endpoint_plan.py \
  tests/test_p2_finalizer_integrity.py
```

Suggested focused P2 test command (**requires torch installed**):

```bash
python -m pytest -q \
  tests/test_p2_analysis.py \
  tests/test_p2_checkpoint_creation.py \
  tests/test_p2_endpoint_plan.py \
  tests/test_p2_finalizer_integrity.py \
  tests/test_p2_full.py \
  tests/test_p2_full_torch.py \
  tests/test_p2_native_candidates.py \
  tests/test_raw_winner_p2.py
```

### Known pre-existing failure

`tests/test_raw_winner_p1_decode.py` imports `scripts.select_raw_winner_p1_decode`,
which is **not present in this package**. `docs/P2_FULL_V2_EXECUTION.md` records this
as a pre-existing collection failure inherited from `main`, and that remains correct
on a fully provisioned machine. Note the ordering, though: on a machine without audio
libraries the same file fails earlier, at `scripts/evaluate_raw_winner_decode.py:20`
(`import librosa`), so a `librosa` error there is not evidence of a different problem.

## Safety boundaries

- Never include access tokens, SSH keys, Hugging Face credentials, or `.env`.
- Do not overwrite the RAW_WINNER 69.49 package or historical P0/P1/P2 evidence.
- Do not automatically upload to the competition, GitHub, or Hugging Face.
- Do not use `gpu002` or modify unrelated Slurm jobs without explicit authority.
- Public and OOD are diagnostic only; validation controls selection and extension.
