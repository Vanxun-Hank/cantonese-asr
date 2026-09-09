<div align="center">

# Cantonese ASR with Whisper

**Source-alternating Whisper-small training, converged capacity/PEFT follow-up, and auditable evaluation.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Repository License](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE)

**English** · [简体中文](README.zh-CN.md)

</div>

This repository contains training, evaluation, packaging, and verification code for
Cantonese automatic speech recognition with Whisper. The method system is a
source-alternating Whisper-small checkpoint: WenetSpeech-Yue steps update the encoder,
and alternating task-provided steps update the full model. A separate P2 follow-up
compares Small, Medium, and Large-v2 Full SFT and LoRA recipes and publishes a
Large-v2 LoRA adapter. The adapter was trained independently and is not an output of
the source-alternating curriculum.

## Public artifacts

- [Source-alternating Whisper-small](https://huggingface.co/cantonese-asr-lab/whisper-small-cantonese-w500-adaptive)
- [Independent Large-v2 LoRA adapters](https://huggingface.co/cantonese-asr-lab/whisper-large-v2-cantonese-p2-lora)
- [Converged P2 report](reports/raw_winner_p2_full_converged.md)

All accuracy and CER values below are percentages.

### Source-alternating Whisper-small

| Evaluation surface | tol2 accuracy (%) | CER (%) | Selection role |
|---|---:|---:|---|
| Fixed validation | 85.47 | 8.59 | Selects the method checkpoint |
| Task-provided local evaluation | 89.42 | 8.13 | Supplementary evaluation |
| OOD, frozen asymmetric scorer | 33.15 | 34.04 | Historical task-protocol score |
| OOD, symmetric script audit | 87.45 | 8.84 | Orthography-controlled diagnostic |

### Large-v2 LoRA follow-up

| Adapter | Local-eval tol2 (%) | Local-eval CER (%) | Symmetric OOD tol2 (%) | Symmetric OOD CER (%) |
|---|---:|---:|---:|---:|
| Seed 42 | 95.16 | 5.07 | 91.40 | 6.38 |
| Seed 43 (repository root) | 95.53 | 5.10 | 90.80 | 6.66 |

Both seeds are published. Seed 43 is the repository-root adapter because it has the
higher local-evaluation tol2; seed 42 has the slightly lower CER. The local-evaluation
surface is therefore a disclosed selection surface for the default adapter, not an
untouched test set. OOD is never used for model, language-model, or fusion selection.

The Large-v2 release is a 15.8 MB LoRA adapter that must be loaded with
`openai/whisper-large-v2`; it is not a standalone 1.55B-parameter model file. It
updates 0.254% of the combined parameters and recorded 6.58 GiB peak allocated
memory in the reported run.

## OOD scoring note

The frozen scorer converts predictions from Traditional to Simplified Chinese but
leaves references unchanged. This has little effect on the mainly Simplified
Validation and local-evaluation references, but strongly affects the predominantly
Traditional OOD references. Applying the same conversion to both sides removes
72.6%–80.6% of raw OOD edit mass across the eight audited converged endpoints.
Raw scores remain available for protocol reproducibility; symmetric scores should be
used when interpreting recognition after controlling the known script asymmetry.

## Data provenance

The task-provided Cantonese material was distributed for the preliminary ASR task of
the [AI Dimsum Cup](https://www.aicompetition-pz.com/topic_detail/19), whose task page
points to the [Cantonese Life Scenarios
Corpus](https://huggingface.co/datasets/leeduckgo/cantonese-life-scenarios-corpus).
The current dataset page establishes provenance and is not claimed to be the exact
training-time snapshot. WenetSpeech-Yue, Common Voice zh-HK, and MDCC are used only in
the experiments that identify them. Training data is not redistributed here.

## Repository layout

```text
cantonese-asr/
├── cantonese_asr/          # Manifest I/O and character metrics
├── configs/rounds/         # Versioned experiment configurations
├── scripts/                # Preparation, training, evaluation, and audits
├── slurm/                  # Cluster launch definitions
├── tests/                  # Unit and integration tests
├── reports/                # Compact experiment reports and receipts
├── train.py                # Main training entry point
├── predict.py              # Standard inference
└── predict_guarded.py      # Inference with generation checks
```

## Installation

```bash
pip install -r requirements.txt
```

For server training or offline packaging, install the corresponding pinned set:

```bash
pip install -r requirements-server-torch.txt
pip install -r requirements-submission.txt
```

## Data format

Training manifests are JSONL with one utterance per line:

```json
{
  "id": "00001",
  "audio_path": "artifacts/data/train_raw/00001.wav",
  "text": "你好！",
  "duration_s": 1.24,
  "source": "task_provided",
  "split": "train"
}
```

`text` is the supervision transcript. Translations, Jyutping, and scene tags may be
retained as metadata but must not replace the label. Audio is expected as 16 kHz mono
WAV unless the selected configuration states otherwise.

## Training and evaluation

```bash
python train.py \
  --manifest artifacts/manifests/train.jsonl \
  --val-manifest artifacts/manifests/val.jsonl \
  --output-dir outputs/run_01 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 1e-5 \
  --epochs 5

python scripts/evaluate_checkpoints.py --checkpoint-dir outputs/run_01
python scripts/evaluate_predictions.py \
  --predictions predictions.jsonl \
  --reference references.jsonl
```

Core metrics are character error rate (`cer`) and the proportion of utterances with
character edit distance at most two (`sentence_accuracy_tol2`). Selection,
normalization, decoder arguments, example order, and model identity are recorded
together so that a metric is not detached from its protocol.

## Packaging and verification

```bash
python scripts/package_submission.py \
  --checkpoint outputs/best/checkpoint-5000 \
  --output model-package.zip

python scripts/verify_submission.py --submission model-package.zip
```

The verifier checks offline loading, prediction count, order, audio paths, and package
shape. PEFT checkpoints must be combined with the base model according to their model
card; the Hugging Face adapter files themselves are intentionally not replaced by
merged weights.

## Evidence boundaries

- The source-alternating curriculum is supported as an integrated recipe; its update
  scope, learning rates, initialization, exposure, and alternation frequency were not
  crossed in one factorial design.
- Validation and local evaluation rank the six converged model families differently
  (two-seed mean tol2 Spearman correlation 0.60). This is a selection risk, not proof
  that either split is defective.
- Full-SFT/LoRA ordering is evaluation-surface dependent. Neither family is claimed to
  be universally better.
- Tokenizer, character-LM, and fusion results are recipe-specific diagnostics, not
  general negative results for those method classes.
- The downloaded evidence archive provides per-utterance products for 134 receipts;
  28 later aggregate receipts have no per-utterance counterpart in that archive.

## Licenses and data access

Repository code is MIT-licensed; see [`LICENSE`](LICENSE). The Whisper-small weights
and Whisper Large-v2 base use the Apache License 2.0. The published Large-v2 LoRA
adapter is MIT-licensed. Upstream datasets remain subject to their own terms.
Information identifying the exact training-time task-data snapshot is available from
the corresponding author by email; the address will be added with the final author
record.
