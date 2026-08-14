# RAW_WINNER P2 Structural Probe Design

Date: 2026-08-12

## Objective and claim boundary

This round adds controlled P2 evidence to the Cantonese Whisper-small
technical report. It does not seek fully converged medium/large training and
must be described as a 150-step matched-exposure adaptation probe.

The immutable reference is `W500_Adaptive_RAW_WINNER`, whose platform score is
69.49. Its model SHA-256 is
`a0f29a5a011213d5e4de34c40a02d42247255645f06e649e092d2dc495094370`.
Nothing in this round may overwrite that checkpoint, its P0-P1 report, or its
submission archive.

The probe must answer five questions:

1. Does model capacity improve adaptation efficiency under a fixed 2,400-item
   exposure and 150 optimizer-step budget?
2. How do Full SFT and LoRA compare in accuracy, memory, throughput, and
   GPU-hours?
3. Is the standard Whisper tokenizer a material Cantonese bottleneck?
4. Can a leakage-free character 5-gram LM exploit the five-best oracle space?
5. Do RAW_WINNER, NOISE_S43, and FULL_LR1E6_S43 contain exploitable prediction
   diversity?

## Immutable inputs

- Training manifest: `artifacts/manifests/external/round2/external73.jsonl`
- Training manifest SHA-256:
  `2ba7edef6811b0d3e96b3347aa70f4237ec27735652d3d313bce62525f1cdea6`
- Exposure stream: exactly 2,400 unique rows selected once with seed 42.
- Evaluation surfaces: fixed validation (702), public (1,900), and OOD (2,000).
- Decode: beam 2, no-repeat n-gram 4, repetition penalty 1.05,
  `max_length=225`, Chinese transcription, no sampling.
- Training augmentation: disabled on every capacity arm.
- Qwen job 1524 stays held; the probe must not cancel or release it.

The frozen stream stores original manifest row indices and stable sample IDs.
For optimizer step `s`, the global examples are rows
`[16*s, 16*(s+1))`. A topology-aware sampler partitions that same ordered
global batch across 1, 2, or 4 ranks and gradient-accumulation microsteps. A
machine-readable receipt proves that reconstructed global step IDs are exactly
equal for all three topologies.

## Capacity and PEFT matrix

All arms initialize from the corresponding original OpenAI checkpoint, train
for 150 steps, use global effective batch 16, AdamW, weight decay 0.01, cosine
scheduling, eight warmup steps, max gradient norm 1.0, BF16, TF32, gradient
checkpointing, and seed/data seed 42.

| Arm | GPUs | Per-device batch / accumulation | LR | Method |
| --- | ---: | ---: | ---: | --- |
| Small Full | 1 | 8 / 2 | 1e-5 | Full SFT |
| Small LoRA | 1 | 8 / 2 | 1e-4 | LoRA |
| Medium Full | 2 | 2 / 4 | 1e-5 | FSDP Full SFT |
| Medium LoRA | 1 | 4 / 4 | 1e-4 | LoRA |
| Large-v2 Full | 4 | 1 / 4 | 1e-5 | FSDP Full SFT |
| Large-v2 LoRA | 1 | 1 / 16 | 1e-4 | LoRA |

LoRA is fixed to rank 8, alpha 16, dropout 0.05, targeting query/value
projections in all attention modules. FSDP uses full sharding, Whisper
Encoder/Decoder layer auto-wrap, original parameters, and a rank-zero full
safetensors checkpoint. Medium Full may fall back only to batch 1 / accumulation
8 after a recorded OOM. Large-v2 Full may not use CPU offload; an OOM under the
registered four-GPU setup is a valid feasibility result.

Every run evaluates step 0 and step 150 on fixed validation and records
accuracy/error metrics, loss, parameters, peak memory, training throughput,
GPU-hours, hashes, and a fixed 32-item batch-one latency/RTF benchmark. Public
and OOD are run only when step 150 improves both its own step-0 tolerance-2
accuracy and CER.

Capacity is considered beneficial relative to Small only when either:

- tolerance-2 accuracy improves by at least 0.005 while CER worsens by no more
  than 0.003; or
- CER improves by at least 0.003 while tolerance-2 accuracy falls by no more
  than 0.005.

## Tokenizer and Unicode audit

The small, medium, and large-v2 tokenizer snapshots are audited on Official
train, validation, public, and OOD text. The audit records characters, tokens,
characters/token, single-character multi-token rate, unknown/replacement
characters, abnormal Unicode, NFC/NFKC changes, Traditional/Simplified and
Cantonese variant collisions, and encodings for
`唔、冇、喺、咗、嘅、啲、佢、嚟、咁`.

The audit is observational: it cannot alter the tokenizer, vocabulary, model
structure, or formal normalizer. Identical tokenizer hashes and encodings are a
positive result and must be reported rather than hidden.

## Character 5-gram LM

The LM is trained only on normalized `external73` training text. Evaluation
references are never read by the training or scoring implementation. The model
uses character 5-grams and additive smoothing with alpha 0.1.

For each RAW_WINNER D7 five-best list, ASR sequence scores and per-character LM
log probabilities are standardized within the sample. Candidates are ranked by

`z_asr + lambda * z_lm`, where lambda is one of 0, 0.1, 0.2, 0.4, or 0.8.

Lambda is selected on validation only by tolerance-2 accuracy, then CER, then
severe errors, then smaller lambda. That single value is frozen for public and
OOD. The report compares ASR top-1, existing MBR, LM reranking, and the five-best
oracle; there is no dynamic fallback.

## Multi-model fusion

Existing D0 predictions for RAW_WINNER, NOISE_S43, and FULL_LR1E6_S43 are
reused. Before fusion, the implementation must prove equal row counts, audio
order, and normalized-reference parity for every surface.

Three deterministic outputs are computed:

- character-edit-distance MBR medoid over the three hypotheses;
- character-level ROVER aligned to the MBR medoid, with ties retaining its
  anchor character;
- best-of-three oracle as an upper bound.

MBR ties use each model's fixed-validation rank. Only validation chooses
between MBR and ROVER; public and OOD remain diagnostic.

## Implementation and scheduling

The round is configuration-driven through
`configs/rounds/raw_winner_p2_structural_probe.json`. Existing default training
behavior must remain unchanged. New interfaces cover:

- topology-invariant fixed-exposure sampling and receipts;
- FSDP full-shard configuration and rank-zero full-state saving;
- deterministic tokenizer, LM, fusion, benchmark, and finalization entrypoints;
- per-arm effective generation parameters and artifact hashes.

Medium and large snapshots are downloaded and hashed on the login node, then
all GPU jobs run with offline model loading. Wave 1 allocates all four gpu001
GPUs to Large Full and gpu002 GPUs to Medium Full (two), Medium LoRA (one), and
Large LoRA (one). Wave 2 runs Small Full/LoRA and parallel endpoint evaluation.
Any arm failure is preserved and retried independently. GPU allocation and
release are recorded at each wave boundary.

## Verification gates

Before formal training:

1. FSDP trainable-parameter scope, full-state save, and offline reload pass.
2. The 1/2/4-GPU reconstructed per-step exposure IDs are byte-identical.
3. LoRA exposes only registered adapter parameters.
4. Tokenizer and Unicode audits are deterministic.
5. LM training cannot access evaluation references and lambda selection reads
   validation only.
6. MBR/ROVER decisions are deterministic and reject prediction-order drift.
7. Metric recomputation and 702/1,900/2,000 row guards pass.
8. Every new training path passes a 32-sample, five-step smoke test.

## Deliverables

- `reports/raw_winner_p2_structural_probe.md`
- machine-readable capacity, tokenizer, LM, and fusion matrices
- model/data/config/artifact SHA-256 receipts
- resource and latency tables with explicit feasibility limitations
- a P2 experiments-and-limitations section in `paper/manuscript.md`
- a README entry pointing to the P2 report

No artifact is uploaded to the competition platform, GitHub, or Hugging Face
without a separate user instruction.
