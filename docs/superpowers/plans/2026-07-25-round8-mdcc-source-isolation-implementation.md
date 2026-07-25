# Round 8 MDCC Source-Isolation Implementation Plan

**Goal:** Run one reproducible Whisper-small Full-SFT experiment that replaces
Round 2 Common Voice membership with unique MDCC publisher-train rows while
holding total training rows and all optimization settings fixed.

**Architecture:** Build an immutable 23,304-row Manifest from all 6,292
official rows and a deterministic, no-replacement sample of 17,012 rows from
the 64,779-row MDCC train pool. Launch one fresh-base Slurm trial with the
Round 2 winning recipe, reuse the current checkpoint evaluator and
validation-only selector, then finalize a Round 8 report with source-separated
OOD diagnostics.

**Tech stack:** Python 3.11, JSONL, pytest, Bash/Slurm, PyTorch 2.6,
Transformers, RTX 4090.

## Files

- Create `scripts/build_mdcc_source_isolation_manifest.py`.
- Create `tests/test_mdcc_source_isolation_manifest.py`.
- Create `slurm/prepare_round8_mdcc_source_isolation.slurm`.
- Create `slurm/round8_mdcc_source_isolation.slurm`.
- Create `tests/test_round8_mdcc_source_isolation.py`.
- Create `configs/rounds/round8.json`.
- Update `AGENTS.md` after the round is operational.

## Task 1: Specify the data contract

- Write tests proving exact source counts, unique IDs, deterministic reruns,
  no-replacement MDCC sampling, Round 2 MDCC intersection reporting and
  protected-split rejection.
- Run the focused test and confirm it fails because the builder does not yet
  exist.

## Task 2: Implement the deterministic builder

- Validate input row counts, unique IDs, non-empty text, train split and
  positive duration.
- Verify optional expected input SHA-256 values.
- Reject source-ID, audio-hash and protected-split overlaps.
- Sort MDCC by ID, shuffle with `Random(seed)`, take the first 17,012 rows,
  annotate sampling seed/rank, merge with all official rows and shuffle
  deterministically.
- Write `train.jsonl`, `smoke32.jsonl` and `data_report.json`.
- Record source counts, hours, hashes, selected-pool coverage and intersection
  with the Round 2 MDCC subset.

## Task 3: Add fixed Slurm contracts

- Add a CPU preparation job using the server's immutable Manifest paths and
  expected hashes.
- Add a one-GPU formal job using fresh Whisper-small, Full SFT, `2e-5`,
  cosine, 3 epochs, batch 8, accumulation 2, no SpecAugment and max length
  225.
- Fail before training unless the data report passes and its output hash
  matches the Manifest.
- Reuse `evaluate_checkpoints.py` and `select_best_checkpoint.py` with the
  fixed validation gates.
- Add static tests proving all fixed controls.

## Task 4: Verify and synchronize

- Run focused tests, compile checks and shell syntax checks locally.
- Commit only Round 8 implementation files, preserving unrelated dirty work.
- Sync code through `scripts/sync_server.sh`.
- Run the complete relevant test set in the canonical server environment.

## Task 5: Build data and run

- Inspect `squeue` without mutating unrelated jobs.
- Submit the preparation job and wait for a passing `data_report.json`.
- Submit a 32-row one-GPU smoke run or equivalent short training check.
- Submit the formal one-GPU Round 8 job only after smoke verification.
- Record job ID, Manifest SHA-256 and immutable output directory.

## Task 6: Evaluate and report

- After each complete epoch, report validation accuracy, CER, loss, recent
  train loss and learning rate.
- At job completion, select only by fixed official validation.
- Generate the canonical Round 8 report and source-separated CV/MDCC OOD
  comparison against Round 2.
- Do not package or upload until the user reviews the fixed validation and OOD
  evidence.
