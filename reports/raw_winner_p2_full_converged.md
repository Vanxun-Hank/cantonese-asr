# P2 Converged Follow-up: Capacity, PEFT, and Evaluation Surfaces

## Scope

This report records the final three-epoch P2 follow-up under
`transformers==4.57.6`. It is additive to the earlier 150-step structural probe.
The follow-up initializes each arm from its corresponding OpenAI Whisper snapshot and
uses the independent `external73` recipe; it does not continue the released
source-alternating Whisper-small checkpoint.

The evidence used here is fixed to:

- Git commit `8167de86ab9356cd366964906a75b82201a0e174` for configurations,
  code, sanitized receipts, and the preserved technical record;
- Hugging Face revision `5627a78f20de8901345188af1e945231a90551d1` for the two
  public Large-v2 LoRA adapters;
- a downloaded NAS handoff with SHA-256
  `01337279fd3b421d75b721cf0171f8725396d4b28a043978b931cede1fbe96d9`
  for per-utterance products and the chronological execution report.

All accuracy and CER values in reader-facing tables are percentages.

## Protocol

The baseline matrix crosses three Whisper sizes with two update recipes:

- Small, Medium, and Large-v2;
- Full supervised fine-tuning and rank-8 LoRA on all attention q/v projections;
- seeds 42 and 43;
- 23,304 unique training utterances: 6,292 task-provided, 8,451 Common Voice
  zh-HK, and 8,561 MDCC;
- three epochs, global batch 16, one 80 GB H100 per arm;
- common review endpoint at step 4,371;
- fixed D0 decoding under `transformers==4.57.6`.

This design matches rows, epochs, optimizer-step count, decoder, and hardware class.
Full SFT and LoRA retain their registered method-specific learning rates, so the rows
compare complete recipes rather than update scope in isolation.

## Baseline matrix

Each cell reports seed 42 / seed 43.

| Arm | Validation tol2 (%) | Validation CER (%) | Local-eval tol2 (%) | Local-eval CER (%) |
|---|---:|---:|---:|---:|
| Small Full | 82.62 / 82.91 | 9.35 / 9.47 | 87.21 / 87.53 | 9.13 / 9.53 |
| Small LoRA | 70.09 / 72.93 | 14.02 / 13.63 | 78.95 / 78.26 | 12.60 / 12.90 |
| Medium Full | 89.17 / 88.75 | 7.36 / 6.97 | 94.05 / 92.63 | 6.26 / 7.39 |
| Medium LoRA | 82.19 / 81.91 | 9.43 / 9.69 | 93.84 / 93.53 | 6.34 / 6.52 |
| Large-v2 Full | 89.46 / 89.46 | 6.85 / 6.77 | 94.68 / 94.53 | 5.74 / 5.58 |
| Large-v2 LoRA | 87.75 / 87.18 | 7.99 / 7.96 | 95.16 / 95.53 | 5.07 / 5.10 |

The capacity trend changes between the early probe and the three-epoch endpoint.
Full SFT gains substantially from Small to Medium and more modestly from Medium to
Large-v2. This is evidence of positive but diminishing returns in the registered
recipe, not evidence that adding capacity is ineffective.

Full-SFT/LoRA ordering depends on evaluation surface. Large-v2 Full leads on
Validation, while Large-v2 LoRA leads on local evaluation in both seeds. The LoRA CER
advantage on local evaluation is 0.67 and 0.48 percentage points for seeds 42 and 43.
Paired bootstrap intervals support that CER direction; the local-evaluation tol2
intervals include or touch zero.

## Public Large-v2 LoRA artifact

The [model repository](https://huggingface.co/cantonese-asr-lab/whisper-large-v2-cantonese-p2-lora)
contains both seeds. Seed 43 is the repository-root adapter because it has the higher
local-evaluation tol2 (95.53% versus 95.16%); seed 42 has the slightly lower CER
(5.07% versus 5.10%). The task-provided local evaluation set is denoted `Public` in
experiment artifacts. Because it chooses the repository-root seed, it is a disclosed
selection surface for that adapter.

The published object is a LoRA adapter, not a standalone full-model file:

| Item | Value |
|---|---|
| Required base | `openai/whisper-large-v2` |
| Base parameters | 1,543,304,960 |
| Whisper-small parameters | 241,734,912 |
| Base-size ratio | 6.38x |
| Trainable fraction | 0.254% of combined parameters |
| Peak allocated memory | 6.58 GiB |
| Seed-43 adapter SHA-256 | `75c115813e328c9fc9c102d1a97240775244442d799a5869a7ec90a352c3c399` |
| Seed-42 adapter SHA-256 | `2b5b20b83c7938d21a7afe2eda54611312fde32007db310ecad44a2411bf472a` |

Relative to the source-alternating Whisper-small checkpoint on the same local surface,
seed 43 changes CER from 8.13% to 5.10%, a 3.03-point absolute and 37.31% relative
decrease. This is a cross-model, cross-recipe comparison and is not attributed to the
source-alternating curriculum.

LoRA reduces trainable parameters and peak memory but not wall time in this run.
Large-v2 LoRA takes about 4:07 per seed, compared with approximately 3:53 and 3:41
for Large-v2 Full. Adapter operators, topology, checkpointing, and evaluation all
contribute to those wall-clock records.

## Evaluation-surface ranking

Across the six baseline families, the Spearman correlation between two-seed mean
Validation and local-evaluation tol2 ranks is 0.60.

| Arm | Validation mean rank | Local-eval mean rank |
|---|---:|---:|
| Large-v2 Full | 1 | 2 |
| Medium Full | 2 | 4 |
| Large-v2 LoRA | 3 | 1 |
| Small Full | 4 | 5 |
| Medium LoRA | 5 | 3 |
| Small LoRA | 6 | 6 |

The observed reversals create model-selection risk, but do not establish that either
split is defective or that the cause is overfitting. Speaker, scene, device, duration,
and transcription-style provenance would be needed for that diagnosis. Validation
selects the source-alternating checkpoint and registered P1 choices; local-evaluation
tol2 selects only the default seed in the separate LoRA repository. OOD selects
nothing.

## OOD script-normalization audit

The frozen scorer converts predictions from Traditional to Simplified Chinese but
does not apply the same conversion to references. The OOD references are predominantly
Traditional Chinese, so raw character scores conflate recognition and script form.
The symmetric audit applies the same conversion to both sides without changing any
decoded text.

| System | Seed | Raw tol2 (%) | Raw CER (%) | Symmetric tol2 (%) | Symmetric CER (%) | Raw-edit share removed (%) |
|---|---:|---:|---:|---:|---:|---:|
| Small Full | 42 | 32.15 | 34.69 | 86.50 | 9.47 | 72.7 |
| Small Full | 43 | 32.35 | 34.59 | 86.10 | 9.47 | 72.6 |
| Medium Full | 42 | 35.35 | 33.06 | 90.90 | 7.12 | 78.4 |
| Medium Full | 43 | 34.70 | 33.03 | 91.15 | 7.13 | 78.4 |
| Large-v2 Full | 42 | 35.85 | 32.47 | 92.50 | 6.31 | 80.6 |
| Large-v2 Full | 43 | 35.90 | 32.62 | 92.10 | 6.50 | 80.1 |
| Large-v2 LoRA | 42 | 36.05 | 32.55 | 91.40 | 6.38 | 80.4 |
| Large-v2 LoRA | 43 | 35.55 | 32.75 | 90.80 | 6.66 | 79.7 |

For the source-alternating Whisper-small checkpoint, the same audit changes OOD from
33.15% tol2 / 34.04% CER to 87.45% / 8.84%, reducing 8,204 raw edits to 2,130.
Raw scores remain the reproducible output of the frozen scorer; symmetric scores are
the appropriate companion when interpreting recognition after controlling the known
script asymmetry. One OOD panel does not establish a general Cantonese domain ranking.

## Source-conditional follow-up

A separate source-conditional Large-v2 Full variant improves local-evaluation tol2 by
0.32 and 0.53 percentage points over the same-seed baseline and changes CER from
5.74%/5.58% to 5.20%/5.25%. Validation tol2 moves in the opposite direction by 0.57
and 0.14 points. Its data and initialization differ from the released method system,
so this result is classified as surface-dependent source conditioning rather than a
direct replication or a source for the public LoRA gain.

## Generation and residual-error diagnostics

Large-v2 Full seed 42 at step 4,371 has 641 Validation edits over 9,356 reference
characters: 454 substitutions, 97 deletions, and 90 insertions. Its 20 highest-edit
utterances account for 20.9% of the total, and Small/Large share errors on 268
utterances accounting for 89.7% of the Large-v2 edit mass. This supports structured
error analysis rather than attributing most residual error to a handful of labels.

One Small Full batch-four control at seed 43 develops two repeated runaways and two
maximum-length hits at its terminal step. The health-aware rule selects the earlier
step 3,643 for that control. This control is not the published adapter; it records why
generation checks are retained alongside CER.

## Compute and receipt coverage

The NAS report attributes 49.81 H100 GPU-hours to the formal training matrix and
57.55 H100 GPU-hours to the complete workflow, including extensions, evaluation,
checks, and failed or cancelled jobs. These totals are report-derived because the
downloaded bundle does not contain raw scheduler-accounting export. They are not the
training time of one adapter.

Receipt reconciliation gives the following coverage:

| Receipt class | Count | Status |
|---|---:|---|
| NAS receipts with per-utterance products | 134 | Hash, row-count, and aggregate recomputation pass |
| NAS/Git overlapping receipts | 132 | Exact after removing the private model path |
| Git-only aggregate receipts | 28 | No per-utterance counterpart in the downloaded archive |
| NAS-only early source-conditional receipts | 2 | Preserved in the local inventory |

The 28 Git-only items consist of eight Medium-LoRA extension and 20 later surface
receipts. Claims from those items remain bounded to aggregate cross-source agreement.
Per-utterance archives and private cluster paths are intentionally not committed.

## Route-level interpretation

The investigated directions do not share one scientific outcome:

| Direction | Evidence class | Bounded conclusion |
|---|---|---|
| `external73` at Small capacity | Recipe-specific negative | Below the source-alternating system on the recorded task-aligned surfaces |
| Source conditioning | Surface-dependent | Improves Large-v2 local evaluation and lowers Validation |
| Capacity | Diminishing returns | Large Small-to-Medium gain and smaller Medium-to-Large gain |
| Approved Medium-LoRA extension | Limited negative | Additional tol2 change with little CER movement |
| Character 5-gram | Tested negative | Validation selects lambda 0.0 for this recipe |
| Static MBR/ROVER | Tested negative | Does not exceed the best Validation single model |
| Current five-best reranking | Tested negative | Small oracle headroom in the registered candidate set |
| Data cleaning | Diagnostic | Current heuristics do not establish large-scale label mismatch |
| Worst-utterance repair | Diagnostic | The observed edit mass is not concentrated in a few examples |

No tested intervention gives an unambiguous improvement on every recorded surface.
That statement is protocol-specific and is not a universal claim that the nine method
classes are ineffective.

## Reproduction boundary

- Environment: Python 3.10.18, PyTorch 2.6.0+cu124, Transformers 4.57.6.
- Training: 23,304 unique rows, three epochs, step 4,371, global batch 16,
  seeds 42/43, one H100 per arm.
- The two LoRA adapter hashes above identify the unchanged Hugging Face weights.
- Public receipts omit absolute model paths; per-utterance archives remain outside Git.
- Training data is not redistributed and remains subject to upstream terms.
- The local-evaluation surface selects the root adapter seed; symmetric OOD scoring is
  a post-hoc audit rather than a replacement for the frozen metric.

These boundaries preserve a useful distinction between the source-alternating method
system, the independent converged follow-up, and the evidence used to interpret both.
