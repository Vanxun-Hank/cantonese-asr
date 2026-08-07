# W500 RAW Winner Publication Implementation Plan

## Goal

Publish the exact 69.49 platform artifact to Hugging Face and a curated,
reproducible implementation of its training and inference workflow to GitHub.

## Phase 1: Import and sanitize the winning workflow

1. Cherry-pick the W500 adaptive implementation and recovery commits from the
   experiment branch into the clean public-release branch.
2. Retain the policy, trainer, preparation, selection, finalization,
   interpolation, inference, packaging, and focused tests needed to reproduce
   the method.
3. Remove machine-specific defaults, historical job identifiers, private
   report paths, and unrelated recovery entry points from the public surface.
4. Add public documentation for the source-alternating optimization rule:
   Wenet steps update the Encoder only; Official steps update the full model.

## Phase 2: Validate the GitHub release

1. Run the focused W500 test suite.
2. Run inference, metrics, and packaging tests affected by the release.
3. Compile the Python entry points and syntax-check retained shell/Slurm files.
4. Scan tracked files for credentials, private absolute paths, raw data,
   checkpoints, model weights, caches, and oversized files.
5. Commit only the intentional public release files.

## Phase 3: Stage and verify the Hugging Face model

1. Extract `W500_Adaptive_RAW_WINNER_submission.zip` without modifying files.
2. Recompute the source ZIP and embedded `model.safetensors` hashes.
3. Verify flat layout, one weight file, required tokenizer/config files, exact
   `predict.py`, and the saved raw-to-ZIP parity receipt.
4. Add a public-safe model card and `provenance.json` without changing model or
   inference files.
5. Scan the staging directory for private paths and credentials.

## Phase 4: Publish and verify

1. Create or reuse
   `Vanxun-Hank/whisper-small-cantonese-w500-adaptive` on Hugging Face.
2. Upload the verified staging directory with resumable large-file upload.
3. Query the remote repository and verify the remote weight size and SHA-256.
4. Push `codex/publish-w500-raw-winner` to GitHub.
5. Open a draft pull request summarizing code scope, tests, model URL, platform
   score, and artifact hashes.

## Completion evidence

- Hugging Face model URL and remote commit.
- GitHub draft pull request URL and branch commit.
- ZIP and model SHA-256 values matching the canonical artifact.
- Focused tests and publication scans passing.

