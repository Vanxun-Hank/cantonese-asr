<div align="center">

# Cantonese ASR with Whisper-small

**A fully supervised-fine-tuned `whisper-small` for Cantonese, with an honest, reproducible experiment trail.**
No architecture or tokenizer changes — every gain comes from data curation and training curriculum, and every claim is checked against a held-out set before it's called a result.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-whisper--small--cantonese--w500--adaptive-yellow.svg)](https://huggingface.co/Vanxun-Hank/whisper-small-cantonese-w500-adaptive)

**English** · [简体中文](README.zh-CN.md)

</div>

---

A Cantonese Automatic Speech Recognition system fine-tuned from `openai/whisper-small`. The model architecture and tokenizer are kept unchanged; adaptation is done via full supervised fine-tuning on Cantonese audio-text pairs.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Data Format](#data-format)
- [Training](#training)
- [Evaluation & Reports](#evaluation--reports)
- [Offline Submission](#offline-submission)
- [Config System](#config-system)
- [Scripts Reference](#scripts-reference)
- [Slurm Jobs](#slurm-jobs)
- [License](#license)
- [Current best public method](#current-best-public-method)
- [Update 2026-09-08: the P2 capacity matrix is converged](#update-2026-09-08-the-p2-capacity-matrix-is-converged)
- [Current score, and how competitive it actually is](#current-score-and-how-competitive-it-actually-is)
- [Roadmap: from 69.49 toward stronger leaderboard performance](#roadmap-from-6949-toward-stronger-leaderboard-performance)

---

## Project Structure

```text
cantonese-asr/
├── cantonese_asr/          # Core library (io, metrics, etc.)
│   ├── __init__.py
│   ├── io.py               # Manifest & artifact I/O
│   └── metrics.py          # CER, sentence accuracy, normalization
├── configs/
│   └── rounds/             # Per-round experiment configs (JSON)
│       ├── w500_extension_40_to_100.json
│       └── short_domain_encoder_adapt.json
├── scripts/                # Data prep, training, evaluation, analysis
│   ├── prepare_*.py        # Manifest preparation
│   ├── download_*.py       # Asset downloaders
│   ├── evaluate_*.py       # Evaluation & scoring
│   ├── predict_*.py        # Inference scripts
│   ├── train_*.py          # Specialized training variants
│   ├── analyze_*.py        # Analysis & diagnostics
│   ├── audit_*.py          # Data quality audits
│   ├── build_*.py          # Training mix builders
│   └── ...                 # See [Scripts Reference](#scripts-reference)
├── slurm/                  # Slurm job definitions
├── tests/                  # Unit & integration tests
├── reports/                # Experiment reports (by round)
├── train.py                # Main training entry point
├── predict.py              # Main prediction entry point
├── predict_guarded.py      # Guarded prediction with safety checks
├── requirements*.txt       # Dependency specs
└── README.md
```

---

## Features

- **Data pipeline** — curates official and external Cantonese audio-text data into reproducible JSONL manifests.
- **Full SFT training** — AdamW with a learning-rate schedule, runnable on a Slurm cluster.
- **Comprehensive metrics** — logs loss, sentence accuracy, CER, learning rate, GPU memory and throughput.
- **Rich reporting** — generates PNG, CSV, JSON and HTML experiment reports automatically.
- **Checkpoint selection** — ranks checkpoints on a fixed validation set, with a separate OOD set used to diagnose generalisation. <sub>Caveat (2026-09-08): the OOD panel's references are traditional Chinese while the scorer simplifies hypotheses, so its raw numbers measure orthography more than recognition — see the converged report.</sub>
- **Offline packaging** — builds a flat submission archive containing exactly one `model.safetensors`.
- **Advanced decoding** — beam search, n-best rescoring and character-level LM integration.
- **Distillation** — supports training against a teacher model such as SenseVoice or FunASR.
- **Domain adaptation** — domain-specific encoder training and directional fine-tuning.

---

## Requirements

### Base

```bash
pip install -r requirements.txt
```

### Server / Training

```bash
pip install -r requirements-server-torch.txt
```

### Submission Packaging

```bash
pip install -r requirements-submission.txt
```

Core dependencies: `torch`, `transformers`, `datasets`, `librosa`, `jiwer`, `editdistance`.

---

## Quick Start

```bash
# 1. Install the environment and dependencies
bash scripts/server_bootstrap.sh

# 2. Download, extract and organise the official data
python scripts/download_assets.py
python scripts/extract_assets.py
python scripts/prepare_manifest.py \
  --index-csv artifacts/datasets/official/index.csv \
  --audio-root artifacts/data/train_raw \
  --output-dir artifacts/manifests \
  --exclude-test-list artifacts/datasets/official/template_pre.jsonl

# 3. Pre-flight checks and baselines
sbatch slurm/gpu_preflight.slurm
sbatch slurm/zero_shot.slurm
sbatch slurm/smoke.slurm
sbatch slurm/memory_probe.slurm

# 4. Launch the hyper-parameter grid
BATCH_SIZE=4 GRAD_ACCUM=4 bash scripts/submit_grid.sh
```

Server paths, the Python environment and the remote host are configured through environment variables:

| Variable | Purpose |
|---|---|
| `PROJECT_DIR` | Project root |
| `ENV_DIR` | Python virtual environment |
| `PYTHON` | Python interpreter |
| `REMOTE_HOST` | Remote training host |
| `REMOTE_DIR` | Remote project directory |

---

## Data Format

Training manifests are JSONL, one sample per line:

```json
{
  "id": "00001",
  "audio_path": "artifacts/data/train_raw/00001.wav",
  "text": "你好！",
  "duration_s": 1.24,
  "source": "official",
  "split": "train"
}
```

- `text` — the Cantonese transcript, used as the supervision label. Mandarin translations, Jyutping and scene tags may be kept as metadata, but never substituted for it.
- `audio_path` — 16 kHz mono WAV.
- `source` — data origin (`official` / `external`).
- `split` — dataset split (`train` / `val` / `test` / `ood`).

---

## Training

### Basic training (main entry point)

```bash
python train.py \
  --manifest artifacts/manifests/train.jsonl \
  --val-manifest artifacts/manifests/val.jsonl \
  --output-dir outputs/run_01 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 1e-5 \
  --epochs 5
```

### Directional and extension training

```bash
# Domain-directional training
python scripts/train_directional.py --config configs/rounds/w500_extension_40_to_100.json

# W500 extension training
python scripts/train_w500_extension.py --config configs/rounds/w500_extension_40_to_100.json

# Distillation
python scripts/train_clean_char_lm.py --teacher funasr_nano --output-dir outputs/distill
```

### Guarded Prediction

`predict_guarded.py` runs inference with safety checks, intended for production use:

```bash
python predict_guarded.py \
  --checkpoint outputs/run_01/checkpoint-5000 \
  --manifest artifacts/manifests/test.jsonl \
  --output predictions.jsonl
```

---

## Evaluation & Reports

### Core metrics

| Metric | Description |
|---|---|
| `sentence_accuracy_tol2` | Share of sentences within a character edit distance of 2 (primary metric) |
| `cer` | Character error rate |
| `loss` | Cross-entropy loss |

### Generating reports

```bash
# Single-round experiment report
python scripts/plot_experiments.py --results-dir outputs/run_01 --output-dir reports/roundN

# Checkpoint comparison
python scripts/evaluate_checkpoints.py --checkpoint-dir outputs/run_01

# OOD generalisation diagnostics
python scripts/evaluate_predictions.py --predictions pred.jsonl --reference ref.jsonl

# N-best oracle analysis
python scripts/analyze_nbest_oracle.py --nbest nbest.jsonl
```

Reports contain checkpoint comparison curves, error samples, a character confusion matrix, per-scene metric breakdowns and OOD diagnostics.

Confirmed results for each round go in `reports/roundN/`; failed or unfinished experiments are not published.

---

## Offline Submission

```bash
# Package a submission
python scripts/package_submission.py \
  --checkpoint outputs/best/checkpoint-5000 \
  --output submission.zip

# Verify it offline
python scripts/verify_submission.py --submission submission.zip
```

`package_submission.py` builds a flat ZIP and asserts it contains exactly one
`model.safetensors`. `verify_submission.py` checks, with networking disabled, that the
model loads and that prediction count, ordering and `audio_path` values all line up.

---

## Config System

Experiments are configured by JSON files under `configs/rounds/`. Each file defines one round:

```json
{
  "round": "w500_extension_40_to_100",
  "base_checkpoint": "outputs/w500_40/checkpoint-20398",
  "training": {
    "epochs": 60,
    "batch_size": 4,
    "grad_accum": 4,
    "lr": 2.5e-5
  },
  "data": {
    "train_manifest": "artifacts/manifests/w500_extension_train.jsonl",
    "val_manifest": "artifacts/manifests/val.jsonl"
  }
}
```

---

## Scripts Reference

### Data Preparation

| Script | Purpose |
|---|---|
| `scripts/prepare_manifest.py` | Build the standard train/validation manifests |
| `scripts/prepare_ood_manifest.py` | Build the OOD test manifest |
| `scripts/prepare_external_manifest.py` | Consolidate external data sources |
| `scripts/download_assets.py` | Download the official dataset |
| `scripts/download_mdc_dataset.py` | Download the MDC dataset |
| `scripts/download_teacher_model.py` | Download teacher model weights |
| `scripts/extract_assets.py` | Extract and organise the raw data |
| `scripts/build_external_training_mixes.py` | Build mixed training sets from external data |

### Training

| Script | Purpose |
|---|---|
| `train.py` | Main training entry point (full SFT) |
| `scripts/train_directional.py` | Directional training (domain adaptation) |
| `scripts/train_w500_extension.py` | W500 extension training |
| `scripts/train_clean_char_lm.py` | Character-level LM training |
| `scripts/interpolate_checkpoints.py` | Checkpoint interpolation and merging |
| `scripts/merge_checkpoint_regions.py` | Region-wise checkpoint merging |
| `scripts/compose_decoder_layers.py` | Decoder layer composition |

### Prediction & Inference

| Script | Purpose |
|---|---|
| `predict.py` | Standard prediction |
| `predict_guarded.py` | Prediction with safety checks |
| `scripts/predict_local_guarded.py` | Local guarded prediction |
| `scripts/predict_beam2_norepeat4.py` | Beam search (beam=2, no repeat trigrams) |
| `scripts/predict_true_nbest_clean_lm.py` | N-best + clean LM rescoring |
| `scripts/predict_sensevoice_manifest.py` | SenseVoice teacher prediction |
| `scripts/predict_funasr_nano_manifest.py` | FunASR nano teacher prediction |

### Evaluation & Analysis

| Script | Purpose |
|---|---|
| `scripts/evaluate_predictions.py` | Score predictions |
| `scripts/evaluate_checkpoints.py` | Compare checkpoints |
| `scripts/evaluate_manifest_loss.py` | Manifest-level loss evaluation |
| `scripts/evaluate_nbest_char_lm.py` | N-best plus character-LM evaluation |
| `scripts/evaluate_closed_set_text_retrieval.py` | Closed-set text retrieval evaluation |
| `scripts/plot_experiments.py` | Visualise experiment reports |
| `scripts/select_best_checkpoint.py` | Select the best checkpoint |
| `scripts/select_global_candidate.py` | Select a global candidate model |
| `scripts/compare_predictions_by_scene.py` | Compare predictions by scene |
| `scripts/analyze_nbest_oracle.py` | N-best oracle ceiling analysis |
| `scripts/analyze_w500_domain_coverage.py` | W500 domain coverage analysis |
| `scripts/analyze_closed_set_hard_subsets.py` | Closed-set hard-subset analysis |
| `scripts/analyze_teacher_pairwise.py` | Pairwise teacher-model comparison |

### Auditing & Quality

| Script | Purpose |
|---|---|
| `scripts/audit_lm_validation_overlap.py` | Check LM train/validation overlap |
| `scripts/audit_closed_set_candidates.py` | Audit closed-set candidate quality |
| `scripts/audit_official_audio_duplicates.py` | Detect duplicates in official audio |
| `scripts/verify_cv_audio_origin.py` | Verify CommonVoice audio provenance |
| `scripts/score_cv_origin_labels.py` | Score CommonVoice provenance labels |
| `scripts/check_metric_parity.py` | Check metric parity |
| `scripts/validate_prediction_order.py` | Validate prediction ordering |

### Submission & Packaging

| Script | Purpose |
|---|---|
| `scripts/package_submission.py` | Build the offline submission package |
| `scripts/verify_submission.py` | Verify the offline submission package |
| `scripts/check_submission.py` | Pre-submission checks |

### Misc Builders

| Script | Purpose |
|---|---|
| `scripts/build_loss_filtered_manifests.py` | Loss-based sample filtering |
| `scripts/build_distillation_manifest.py` | Build the distillation manifest |
| `scripts/build_wenet_teacher_challenge.py` | Build the Wenet teacher challenge set |
| `scripts/build_short_domain_encoder_stream.py` | Short-domain encoder data stream |
| `scripts/retrieve_closed_set_formal.py` | Closed-set formal text retrieval |
| `scripts/attribute_formal_text_domain.py` | Formal-text domain attribution |

---

## Slurm Jobs

All Slurm job scripts live under `slurm/`, covering:

- **Training** — `train_*.slurm`, `user_*.slurm`
- **Evaluation** — `evaluate_*.slurm`
- **Verification** — `verify_*.slurm`
- **Analysis** — `analyze_*.slurm`
- **Preparation** — `prepare_*.slurm`
- **Generation** — `generate_*.slurm`

Usage:

```bash
sbatch slurm/train_w500_extension.slurm
```

---

## License

MIT — see [`LICENSE`](LICENSE).

~~`scripts/package_submission.py` builds a flat ZIP and asserts it contains exactly one `model.safetensors`. `scripts/verify_submission.py` checks, with networking disabled, that the model loads and that prediction count, ordering and `audio_path` values all line up.~~
<sub>Struck: this paragraph is duplicated verbatim from [Offline Submission](#offline-submission) and was never license text.</sub>

## Current best public method

The best public method is the **W500 adaptive curriculum**: on WenetSpeech-Yue batches
only the Whisper encoder is updated, while on Official batches the whole model is
unfrozen. The two kinds of optimizer step are interleaved, so external data mainly
broadens Cantonese acoustic coverage while short Official utterances keep correcting
Cantonese character choice, insertions, repetitions and EOS behaviour.

The exact model scoring 69.49 on the platform is released on
[Hugging Face](https://huggingface.co/Vanxun-Hank/whisper-small-cantonese-w500-adaptive).
Training logic, selection guardrails, inference configuration and the scope of what is
reproducible are documented in
[`docs/W500_ADAPTIVE_METHOD.md`](docs/W500_ADAPTIVE_METHOD.md).

## Update 2026-09-08: the P2 capacity matrix is converged

Several caveats struck through below are settled by
[`reports/raw_winner_p2_full_converged.md`](reports/raw_winner_p2_full_converged.md).
All 12 baseline arms now run to 3 epochs on two seeds, with Public and OOD for the top four.

- **Capacity has been measured, not just hypothesised.** Large-v2 Full reaches validation
  tol2 `0.8946` / CER `0.0685` against RAW_WINNER's `0.8547` / `0.0859`. But Small→Medium is
  +0.063 tol2 while Medium→Large-v2 is only **+0.005** for double the parameters.
- **LoRA is not the weaker option it looks like on validation.** It trails Full by 0.020 tol2
  there, but *leads* on Public (`0.9516` vs `0.9468`, 11% lower CER, both seeds) while training
  0.254% of the parameters.
- **The OOD panel was never measuring recognition.** Its references are traditional while the
  scorer forcibly simplifies hypotheses, so 73-81% of recorded OOD error is script conversion.
  Normalised, Large-v2 Full's OOD CER is `0.063` — within 0.006 of its Public CER.
- **Capacity is not where the remaining error is.** 89.7% of Large-v2 Full's residual edits are
  also wrong in Small Full, 70.8% are substitutions, and only 3.1% look like misaligned data.

None of this changed the released 69.49 model, which is still the W500 adaptive checkpoint.

## Current score, and how competitive it actually is

The published W500_Adaptive_RAW_WINNER scores as follows on the platform's hidden test set:

| Metric | Result |
| --- | ---: |
| Platform score | 69.49 |
| CER | 0.2473053892215569 |
| Sentence accuracy (edit distance <= 2) | 0.3400 |
| Hidden evaluation samples | 200 |

This repository holds no leaderboard snapshot and no other entrants' scores, so it cannot
honestly state a rank. 69.49 should also not be read as a general Cantonese ASR benchmark
result — it comes from one competition-specific hidden test. The following explains why
this score is not yet a stable, strong leaderboard result.

### Why

1. **Absolute error is still high.** CER is 24.73%, and sentence accuracy at a 2-character
   tolerance is only 34%. Roughly 66% of hidden-set sentences therefore fail the "at most
   two wrong characters" bar. That error level caps the achievable position on its own,
   however careful the training pipeline is.

2. **Model capacity and tokenizer were not upgraded.** The released model is still
   Whisper-small at about 241.7M parameters, with architecture and tokenizer unchanged.
   ~~The work went into data and optimisation strategy rather than a larger model, a
   Cantonese-specific tokenizer, or stronger language modelling — which imposes a visible
   ceiling.~~ Capacity has since been measured: Medium and Large-v2 both clear RAW_WINNER,
   but the Medium→Large-v2 step is only +0.005 tol2, so the ceiling is not mainly a capacity
   ceiling. Tokenizer and language modelling remain untested.

3. **External data is distributionally different from the competition corpus.**
   WenetSpeech-Yue audio is generally longer and may carry pseudo-labels or different
   transcription conventions; competition speech is closer to short utterances in a
   particular annotation style. The W500 curriculum mitigates this but cannot remove
   differences in speaker, noise, duration, vocabulary and character usage.

4. **Wenet data only updates the encoder.** This is a deliberate stability choice: Wenet
   batches broaden acoustic coverage while the decoder and `proj_out` stay frozen, and
   only Official batches perform full SFT. That limits how much external data can skew
   Cantonese character choice, insertions, repetitions and EOS behaviour — but it also
   means a large amount of external speech never trains the decoder's vocabulary or
   language-modelling ability, capping the text-side gains a leaderboard rewards.

5. **Inference is still conservative single-beam decoding.** The official reproduction
   fixes `num_beams=1`, `max_length=225`, and a fixed repetition-penalty and suppression
   configuration. No beam-search comparison, LM rescoring, candidate fusion, contextual
   biasing or test-time augmentation is included; those may help, but they are unverified
   and must not be blended into the 69.49 result.

6. **The selection objective is not the hidden leaderboard.** Checkpoints are ranked on a
   fixed validation set only, with Public/OOD used as a stability veto rather than for
   ranking. That is the more trustworthy design and avoids fitting the leaderboard, but it
   also means the validation optimum need not be the hidden-set optimum.

7. **Augmentation remains restrained.** The current configuration disables SpecAugment
   explicitly. There is likely headroom on noise, channel variation, speaking rate and
   truncation in real Cantonese recordings — but that has to be established by its own
   experiment, not inferred from this result.

### How to read this

The current work is best understood as an **auditable, reproducible competition baseline**
rather than a finished, leaderboard-optimised system. To become competitive, the next round
should record, separately:

- ~~per-sentence error types: insertions, repetitions, substitutions, truncations and EOS failures;~~ done — substitutions dominate at 70.8%, and the tail is flat;
- the gain from each decoding configuration, reported apart from the `num_beams=1` baseline;
- the independent effect of decoder-focused fine-tuning, ~~model capacity,~~ and a Cantonese tokenizer;
- how the competition corpus differs from Wenet/Official data in duration, speaker, scene and character usage;
- ~~the correlation between validation, public, OOD and the final hidden leaderboard.~~ partly answered — validation and Public disagree on Full vs LoRA, and OOD needs rebuilding before it can be correlated with anything.

Any new approach must keep passing fixed-validation ranking, the Public/OOD guardrails,
offline inference parity and SHA-256 artifact verification. Without those, a higher one-off
score does not demonstrate a better model.


## Roadmap: from 69.49 toward stronger leaderboard performance

~~The next round should not treat "train longer" as the default answer. It should separate
the main variables and verify them in order of cost and risk. What follows is a
reproducible, revertible path, in which~~ "passing" means an improvement on the fixed
validation set that also trips none of the Public/OOD or offline-parity guardrails.

The full technical configuration of the RAW_WINNER 69.49 training chain and the executed
P0-P2 experiments — data and weight provenance, the eight-GPU topology, metric semantics,
the decoding matrix, Official-only correction, single-axis augmentation, Small/Medium/Large-v2
full SFT and LoRA, tokenizer/Unicode handling, the character LM, fusion, resource cost,
failure recovery and SHA-256 sums — is consolidated in:

**[`docs/RAW_WINNER_P0_P1_P2_TECHNICAL_REPORT.md`](docs/RAW_WINNER_P0_P1_P2_TECHNICAL_REPORT.md)**

That is the recommended single entry point. An auto-generated short form of P2 is in
[`reports/raw_winner_p2_structural_probe.md`](reports/raw_winner_p2_structural_probe.md),
and the paper-style write-up with its limitations is in
[`paper/manuscript.md`](paper/manuscript.md). ~~The P2 results support only a 2,400-sample,
150-step matched-budget conclusion about adaptation efficiency; they are not an architecture
ranking at convergence~~ — superseded on 2026-09-08 by the converged matrix in
[`reports/raw_winner_p2_full_converged.md`](reports/raw_winner_p2_full_converged.md), which
also reverses the probe's ranking. None of it fed the 69.49 platform submission.

### P0: establish a comparable error baseline first

1. **Freeze the evaluation protocol.** Lock the current checkpoint, manifest, text
   normalisation, audio ordering and submission verification, and keep an immutable copy
   of the 69.49 baseline.
2. **Bucket the errors per sentence.** Count insertions, repetitions, substitutions,
   truncations and EOS failures, sliced by duration, speaker, noise/channel, data source
   and vocabulary type. The goal is to answer *where* the score is lost, not to stare at a
   single aggregate CER.
3. **Check metric correlation.** Record validation, Public, OOD and final hidden results
   together. If the validation optimum disagrees with the hidden set, later model selection
   must treat that disagreement as a risk rather than explain it away afterwards.

### P1: low-risk changes that verify quickly

1. **Decoding ablation.** Vary only `num_beams`, `max_length`, repetition penalty and
   candidate rescoring on the same underlying model, starting with a small beam 1/2/4/5
   matrix. Report CER, tol2, truncation rate and inference cost per configuration; never
   record a decoding gain as a training gain.
2. **Data and domain alignment.** Compare competition audio against Wenet/Official on
   duration, speaker, noise, sample rate and character usage. Prioritise short-utterance
   construction, speaker-disjoint validation and targeted domain sampling that does not
   leak the test set.
3. **A short decoder-side experiment.** Keeping the encoder scheme fixed, try Official-only
   decoder-focused fine-tuning and watch character choice, insertions, repetitions and EOS.
   Do not change the Wenet update rule and the decoder learning rate in the same round.
4. **Single-axis augmentation.** Test SpecAugment, speed perturbation, noise/reverberation
   and channel perturbation one at a time, so any effect remains attributable.

### P2: costlier structural changes

1. ~~**Model capacity.** Once the baseline and the data-alignment conclusions are stable,
   compare Whisper-medium/large, parameter-efficient fine-tuning and full SFT on gain,
   memory, throughput and latency.~~ **Done (2026-09-08).** Small/Medium/Large-v2 x
   full-SFT/LoRA, two seeds, 3 epochs. Returns collapse above Medium; LoRA costs 6.58 GB
   against Full's higher peak but is *not* faster in wall time (4:07 vs 3:53 at Large-v2).
2. **Cantonese text modelling.** Evaluate a Cantonese-specific tokenizer, vocabulary
   extension, or external LM rescoring — checking Unicode, traditional/simplified and
   variant-character normalisation, and offline submission compatibility alongside.
3. **Multi-model or multi-candidate fusion.** Only after single-model decoding and data
   strategy gains are confirmed should checkpoint ensembling, TTA or candidate fusion be
   assessed, so complexity does not paper over single-model problems.

### Minimum experiment matrix

| ID | Variable changed | Held fixed | Primary observation | Passing condition |
| --- | --- | --- | --- | --- |
| A | Decoding configuration | Original RAW_WINNER, same audio order | CER, tol2, truncation rate, latency | Validation improves and Public/OOD does not regress |
| B | Target-domain sampling / short-utterance strategy | Model, decoding, total update budget | Per-duration and per-source slices, overall CER | Stable gain on both target slices and overall |
| C | Decoder-focused SFT | Encoder update rule, data version | Character choice, insertions, repetitions, EOS | Text-side errors drop with no clear acoustic regression |
| D | A single augmentation axis | Data sampling, model, decoding | Noise/channel/speed slices | OOD improves without losing validation |
| E | Larger model or tokenizer | Data and evaluation protocol | Score, parameters, memory, latency | ~~Gain justifies the added cost~~ **Model half done 2026-09-08: it does not, above Medium. Tokenizer untested.** |

### Selection and release rules

- Each experiment changes one main variable; configuration, manifest, checkpoint and output
  package all record a SHA-256.
- Rank on fixed validation first, then apply Public/OOD as a veto. The hidden leaderboard is
  final external validation only and never feeds back into tuning.
- Any new result is named and reported separately from the 69.49 RAW_WINNER. It does not
  overwrite the current baseline, and an unverified hypothesis is not written up as a conclusion.
- Complete A→B→C→D before deciding whether to spend on E. If the P0 error buckets show the
  problem is mostly annotation convention or decoding, pause scaling the model up.
