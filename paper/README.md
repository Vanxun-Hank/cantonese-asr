# Technical Report

This directory contains the public Markdown source for:

> **A Source-Alternating Curriculum for Cantonese Speech Recognition with
> Whisper**

The report describes a Whisper-small adaptation recipe in which source-pure
WenetSpeech-Yue steps update the encoder and alternating task-provided steps
update the full model. It also consolidates the completed P0-P2 studies on
decoding, task-matched recentering, augmentation, early adaptation,
parameter-efficient tuning, tokenizer behavior, language-model rescoring,
model fusion, generation health, and resource use.

## Public files

- [`manuscript.md`](manuscript.md): English Technical Report v0.4.
- [`references.bib`](references.bib): bibliography for the manuscript citation
  keys.
- [`figures/output/figure1_source_alternating_curriculum.png`](figures/output/figure1_source_alternating_curriculum.png):
  source-alternating curriculum overview.

## Released-system results

All reported accuracy and CER values below are percentages.

| Evaluation surface | tol2 accuracy (%) | CER (%) |
|---|---:|---:|
| Fixed validation set | 85.47 | 8.59 |
| Task-provided local test set | 89.42 | 8.13 |
| OOD set | 33.15 | 34.04 |

The fixed validation set controls checkpoint selection. The local-test and OOD
sets provide additional task-aligned and distribution-shift evaluation. They
do not select checkpoints, language-model weights, or fusion methods.

## Data provenance

The task-provided Cantonese materials were distributed for the preliminary ASR
task of the [AI Dimsum
Cup](https://www.aicompetition-pz.com/topic_detail/19), whose task page points to
the [Cantonese Life Scenarios
Corpus](https://huggingface.co/datasets/leeduckgo/cantonese-life-scenarios-corpus).
The current dataset page documents provenance; it is not presented as the exact
training-time revision. WenetSpeech-Yue is used as the broader acoustic source.

## Release links

- [Model weights and model card](https://huggingface.co/cantonese-asr-lab/whisper-small-cantonese-w500-adaptive)
- [Training and evaluation repository](https://github.com/Vanxun-Hank/cantonese-asr)

The released model weights are available under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Information
identifying the exact training-time task-data snapshot is available from the
corresponding author by email; the contact address will be added with the final
author record. Author names and the AI-use disclosure remain to be confirmed.
