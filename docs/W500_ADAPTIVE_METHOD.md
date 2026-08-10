# W500 adaptive Cantonese ASR curriculum

This document describes the public training method used by
`W500_Adaptive_RAW_WINNER`, a Whisper-small checkpoint that scored **69.49**
on the competition platform.

## Why alternate two update types?

The external WenetSpeech-Yue recordings provide additional Cantonese acoustic
coverage, but they are generally longer than the competition utterances and
may contain pseudo-label or transcription-style differences. Updating the
entire model on those batches can teach useful acoustics while also changing
Cantonese word choice, insertion behavior, and end-of-sequence behavior.

The curriculum separates those responsibilities:

```text
Wenet batch    -> Encoder trainable; Decoder and proj_out frozen
Official batch -> Encoder, Decoder, and proj_out all trainable
                 -> repeat
```

The source is fixed for the whole optimizer step, including both gradient
accumulation micro-batches. Sources are never mixed inside one optimizer step.

## Winning branch

The published checkpoint continued from a verified 35-hour Encoder-only W500
checkpoint and evaluated fixed cumulative Wenet milestones. Its selected raw
checkpoint is labeled `checkpoint-wenet-50h`.

The branch configuration is recorded in
[`configs/rounds/w500_adaptive_continuation.json`](../configs/rounds/w500_adaptive_continuation.json).
The core settings were:

- Whisper-small full model, unchanged tokenizer and architecture;
- global effective batch size 16 (`batch_size=8`, accumulation 2);
- one reinitialized AdamW optimizer so Encoder moments remain continuous;
- constant learning-rate schedule with no warmup;
- Wenet step: Encoder LR from the selected arm, Decoder/output gradients absent;
- Official step: Full-SFT LR from the selected arm;
- SpecAugment disabled;
- seed and data seed 42;
- checkpoint milestones driven by accumulated audio duration rather than epoch.

`scripts/train_w500_adaptive_continuation.py` asserts the required trainable
topology before each source step. `scripts/prepare_w500_adaptive_continuation.py`
freezes the data stream and receipts, while the selector and finalizer keep
fixed validation as the ranking surface and Public/OOD as guardrails.

## Selection and stability gates

Candidate checkpoints are ranked on fixed validation using a CER-heavy proxy
with sentence accuracy at edit-distance tolerance 2. Public and OOD results do
not rerank checkpoints; they can only reject an unstable candidate. The gates
cover insertion count, severe errors, repeated generation, replacement
characters, effective maximum-length hits, and OOD regression.

The selected package passed:

- a flat archive check with exactly one `model.safetensors`;
- offline inference on a fixed 32-item subset;
- raw-checkpoint versus extracted-ZIP text and token parity;
- SHA-256 checks for weights, model configuration, generation configuration,
  tokenizer assets, and `predict.py`.

## Exact published inference behavior

The Hugging Face repository preserves the platform-scored files without
re-saving the model. The bundled `predict.py` supplies:

```text
language=zh
task=transcribe
num_beams=1              # CLI default used by the standalone entry point
max_length=225
```

The bundled `generation_config.json` additionally supplies
`no_repeat_ngram_size=4`, `repetition_penalty=1.05`, and the Whisper suppression
lists. Change decoding parameters only as a separate experiment; doing so no
longer reproduces the submitted artifact.

## Public result

The hidden platform evaluation used 200 utterances:

| Metric | Value |
| --- | ---: |
| Final score | 69.49 |
| CER | 0.2473053892 |
| Sentence accuracy (tolerance 2) | 0.3400 |

This is competition-specific evidence, not a general Cantonese ASR benchmark.
