# W500 RAW Winner Public Release Design

Date: 2026-08-07

## Objective

Publish the exact model that scored 69.49 on the competition platform and the
minimal reproducible training, evaluation, inference, and packaging workflow
that produced it. The release must not expose private datasets, credentials,
server-only paths, optimizer state, unrelated experiments, or unverified model
weights.

## Selected approach

Use a curated public-release branch based on `origin/main`, then bring across
only the canonical W500 adaptive curriculum implementation and its focused
tests. This avoids publishing the large dirty experiment worktree or its local
history while preserving the algorithm that matters.

Two destinations have separate responsibilities:

- GitHub contains source code, reproducibility documentation, configuration
  templates, inference code, and packaging/parity checks. It does not contain
  model weights or datasets.
- Hugging Face contains the exact platform-scored model files, tokenizer and
  generation configuration, `predict.py`, requirements, a model card, and a
  machine-readable provenance manifest. It does not contain training data,
  optimizer state, or private filesystem paths.

## Canonical artifact

The release source is the locally verified flat submission archive named
`W500_Adaptive_RAW_WINNER_submission.zip`. Its machine-specific source path is
kept out of the public repository.

Recorded provenance:

- Platform name: `W500_Adaptive_RAW_WINNER`
- Platform final score: `69.49`
- Platform CER: `0.2473053892215569`
- Platform sentence accuracy at tolerance 2: `0.34`
- ZIP size: `895743248` bytes
- ZIP SHA-256:
  `b4fda8ac549d37d8cac9797950636d50ca28f74d8e5e76e2479c15de9b956bc5`
- `model.safetensors` SHA-256:
  `a0f29a5a011213d5e4de34c40a02d42247255645f06e649e092d2dc495094370`
- Training checkpoint label: `checkpoint-wenet-50h`
- Method family: adaptive W500 curriculum with Wenet Encoder-only updates and
  Official full-model correction updates.

The local submission manifest and parity receipt are evidence inputs. Their
private absolute paths must be removed from public metadata.

## GitHub release scope

The curated branch will contain:

1. The W500 policy helpers and adaptive curriculum trainer.
2. A preparation/validation script for frozen manifests and experiment config.
3. Checkpoint selection, finalization, and optional decoder-interpolation tools
   where they are required by the winning workflow.
4. The exact standalone inference path used by the submission.
5. Submission packaging and parity verification.
6. Focused unit tests for source scheduling, encoder freezing, checkpoint
   selection, package layout, and inference parity.
7. Public documentation describing the alternating optimization rule:
   Wenet steps update only the Encoder; Official steps update the full model.

Historical one-off recovery scripts, private reports, raw manifests, Slurm job
IDs, local paths, credentials, and unrelated rounds are excluded.

## Hugging Face repository

Publish to the model repository:

`Vanxun-Hank/whisper-small-cantonese-w500-adaptive`

The repository will include the extracted flat model package plus:

- `README.md` model card with base model, intended use, limitations, method,
  platform metrics, and reproducibility hashes;
- `provenance.json` with artifact hashes and public-safe score metadata;
- the exact `predict.py` and generation files from the scored ZIP.

The model card will state that platform results are based on a hidden 200-item
evaluation sample and should not be interpreted as a universal Cantonese ASR
benchmark. The training datasets are referenced by name but are not
redistributed.

## Verification gates

Before upload:

1. Recompute the ZIP and embedded weight SHA-256 values.
2. Compare the embedded files with the saved submission manifest.
3. Verify the ZIP is flat and contains exactly one `model.safetensors`.
4. Re-run focused package and inference tests without network access where
   practical.
5. Scan the GitHub diff and Hugging Face staging directory for tokens, private
   paths, data files, checkpoints, caches, and oversized unrelated artifacts.

After upload:

1. Query Hugging Face repository metadata and download manifests.
2. Confirm the remote model weight size and SHA-256 match the canonical model.
3. Run GitHub tests, commit intentional files only, push the release branch,
   and open a draft pull request.
4. Report the Hugging Face URL, GitHub PR URL, commit ID, and all final hashes.

## Failure handling

- A hash mismatch stops publication; no file is substituted or re-saved.
- An interrupted Hugging Face upload resumes from the same staging directory.
- A GitHub test failure blocks the pull request from being marked ready.
- Existing local experiment work is never reset, cleaned, staged, or deleted.
