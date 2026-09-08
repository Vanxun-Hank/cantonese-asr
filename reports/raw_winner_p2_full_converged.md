# RAW_WINNER P2 Full — Converged Capacity Matrix

> Claim boundary: results at 3 epochs (step 4371), two seeds per arm, single H100,
> `transformers==4.57.6`. Validation for all 12 baseline arms; Public and OOD for the four
> Full/LoRA arms above Small LoRA. Text/LM/fusion and the full finalizer have not been run.

Round `raw_winner_p2_full_v1`, recipe `p2_full_four_gpu_v2`, base commit `beb2396`.
The original round ran on 4x RTX 4090 and stopped on 2026-09-04 with only Small Full and
Medium Full past epoch 3. This rebuild re-ran the whole capacity matrix on one 80 GB card per
arm and completed all 12 baseline arms.

Per-checkpoint evaluation receipts for every run below are committed under
[`reports/p2_full_converged/receipts/`](p2_full_converged/receipts/), one JSON per
`<run>/<arm>_S<seed>/checkpoint-<step>`. Each carries the config SHA, manifest SHA, weight
hashes, the effective generate kwargs and per-file SHAs of the diagnostics it summarises.

## Capacity matrix, validation @ step 4371

| arm | tol2 s42 | tol2 s43 | CER s42 | CER s43 | severe s42/s43 |
|---|---|---|---|---|---|
| LARGE_V2_FULL | 0.894587 | 0.894587 | 0.068512 | 0.067657 | 11/13 |
| MEDIUM_FULL | 0.891738 | 0.887464 | 0.073643 | 0.069688 | 15/11 |
| LARGE_V2_LORA | 0.877493 | 0.871795 | 0.079949 | 0.079628 | 16/16 |
| SMALL_FULL | 0.826211 | 0.829060 | 0.093523 | 0.094699 | 17/17 |
| MEDIUM_LORA | 0.821937 | 0.819088 | 0.094271 | 0.096943 | 18/19 |
| SMALL_LORA | 0.700855 | 0.729345 | 0.140231 | 0.136276 | 42/39 |

RAW_WINNER reference on the same surface: tol2 `0.854701`, CER `0.085934`.

### The structural probe's ranking does not survive convergence

The probe report states *"Large-v2 Full ... is not a replacement for RAW_WINNER"*. That holds
at its own 150-step matched budget and it explicitly scoped itself to adaptation efficiency
rather than convergence. At 3 epochs the ordering changes: **LARGE_V2_FULL, MEDIUM_FULL and
LARGE_V2_LORA all clear RAW_WINNER.** LARGE_V2_FULL exceeds it by +0.040 tol2 with 20% lower
CER, and also beats the best single component from the fusion study (NOISE_S43, `0.873219`).

### Capacity returns collapse above Medium

Small -> Medium is +0.063 tol2; Medium -> Large-v2 is **+0.005** for double the parameters.
Both LARGE_V2_FULL seeds land on the identical tol2 to six decimals.

### LoRA does not close the gap, but the gap narrows with capacity

Full minus LoRA at the same size: Small `-0.112`, Medium `-0.069`, Large `-0.020`. LoRA is
therefore not merely slower to converge at this budget. SMALL_LORA is also the only unstable
arm (severe 42/39 against 11-19 everywhere else).

**This holds on validation only** — on Public the Large-v2 ranking reverses, and on OOD the
margin shrinks to 0.012. See the surfaces section below before drawing a conclusion about
LoRA.

## Public and OOD surfaces, step 4371

D0_CURRENT decoding, public = 1900 rows of `template_pre`, OOD = the 2000-row v1 panel.
Receipts under [`reports/p2_full_converged/receipts/surfaces/`](p2_full_converged/receipts/surfaces/).

| arm | public tol2 s42/s43 | public CER s42/s43 | OOD tol2 s42/s43 | OOD CER s42/s43 |
|---|---|---|---|---|
| LARGE_V2_LORA | **0.951579 / 0.955263** | **0.050719 / 0.050985** | 0.360500 / 0.355500 | 0.325491 / 0.327482 |
| LARGE_V2_FULL | 0.946842 / 0.945263 | 0.057436 / 0.055790 | 0.358500 / 0.359000 | 0.324744 / 0.326238 |
| MEDIUM_FULL | 0.940526 / 0.926316 | 0.062642 / 0.073942 | 0.353500 / 0.347000 | 0.330595 / 0.330263 |
| SMALL_FULL | 0.872105 / 0.875263 | 0.091293 / 0.095342 | 0.321500 / 0.323500 | 0.346902 / 0.345948 |

**The OOD column above is not measuring recognition** — see below. Read it as an
orthography score.

### LoRA's validation deficit does not transfer to the other surfaces

On validation, LARGE_V2_LORA trails LARGE_V2_FULL by 0.020 tol2 (`0.877493/0.871795` against
`0.894587/0.894587`). **On Public the ranking reverses**: LoRA leads by +0.005 to +0.010 tol2
with 11% lower CER, in both seeds. On OOD as scored the two are indistinguishable
(0.3555-0.3605 against 0.3585-0.3590); once the orthography artefact below is removed, Full
leads by 0.012 (0.921-0.925 against 0.908-0.914) — still smaller than its 0.020 validation
margin.

Both arms were decoded from the same step, the same decoder and the same manifest hash, so this
is a surface effect and not a checkpoint mix-up. The natural reading is that full SFT is buying
its validation margin by fitting the validation split's distribution, and that margin does not
exist off that split. The claim "LoRA does not close the gap" is therefore true *only* on
validation, and the section above should be read with that restriction.

This also matters for cost: LoRA trains 0.254% of the parameters at 6.58 GB peak.

### The OOD panel is scoring orthography, not recognition

Raw OOD numbers look catastrophic next to Public — tol2 0.32-0.36 against 0.87-0.96 — and the
obvious reading is a domain gap. It is not. `cantonese_asr/metrics.py` follows the official
evaluator exactly: `normalize_prediction` converts predictions to simplified, and
`normalize_reference` deliberately does not convert references (*"references are not converted
to simplified Chinese"*). That asymmetry is correct when references are already simplified,
which is true of the official and validation surfaces. **The OOD panel's references come from
Common Voice zh-HK and are traditional**, and `prepare_ood_manifest.py` applies no orthography
handling, so every traditional character in a reference is scored as a substitution against a
forcibly-simplified hypothesis.

The top OOD confusions are exactly that and nothing else: 係→系 (336), 個→个 (244), 學→学
(132), 會→会 (94), 時→时 (91), 話→话 (87), 為→为 (85), 灣→湾 (84), 過→过 (74), 國→国 (73).

Converting **both** sides to simplified before scoring:

| arm | OOD tol2 as scored | OOD tol2 normalised | OOD CER as scored | OOD CER normalised |
|---|---|---|---|---|
| LARGE_V2_FULL | 0.358500 / 0.359000 | **0.925000 / 0.921000** | 0.324744 / 0.326238 | **0.063115 / 0.064982** |
| LARGE_V2_LORA | 0.360500 / 0.355500 | 0.914000 / 0.908000 | 0.325491 / 0.327482 | 0.063779 / 0.066559 |
| MEDIUM_FULL | 0.353500 / 0.347000 | 0.909000 / 0.911500 | 0.330595 / 0.330263 | 0.071248 / 0.071289 |
| SMALL_FULL | 0.321500 / 0.323500 | 0.865000 / 0.861000 | 0.346902 / 0.345948 | 0.094693 / 0.094651 |

**72.7% to 80.6% of the recorded OOD edit distance is script conversion.** Public moves by
0.2-0.5% under the same treatment, which confirms the effect is specific to this panel and not
an artefact of the normaliser.

Three of the first five OOD utterances are perfect recognitions scored as errors:

```
ref 我住喺堅尼地城站附近        hyp 我住喺坚尼地城站附近        ed 1 -> 0
ref 大嫂要去九龍城沐泰街嗰度買啲嘢  hyp 大嫂要去九龙城沐泰街𠮶度买啲嘢  ed 3 -> 0
ref 依法享有就業教育醫療旅遊金融等多項服務和便利                      ed 6 -> 0
```

So the honest statement is the opposite of the one the raw table suggests: **normalised OOD CER
(0.063) is within 0.006 of Public CER (0.057) for Large-v2 Full.** The models generalise to this
panel about as well as they do to the public set. The 457-529 severe flags per 2000 utterances
are the same artefact.

This is a surface-construction defect, not a metric bug, and it predates this round — the
structural probe's OOD column carries it too, so any "OOD is hard" reading drawn from earlier
rounds needs rechecking. The fix belongs in `prepare_ood_manifest.py` (normalise references at
build time) or in an explicitly symmetric OOD scorer; it is deliberately not patched here,
because changing a frozen manifest mid-round would break its recorded SHA.

## Public is not ranking the arms the way validation does

Every completed variant, Public surface, step 4371, sorted by tol2. RAW_WINNER scores
`0.894211 / 0.081328` on this surface.

| variant | arm | tol2 | CER | severe |
|---|---|---|---:|---:|
| baseline | **LARGE_V2_LORA_S43** | **0.955263** | **0.050985** | 3 |
| baseline | LARGE_V2_LORA_S42 | 0.951579 | 0.050719 | 2 |
| SC | LARGE_V2_FULL_S43 | 0.950526 | 0.052454 | 2 |
| SC | LARGE_V2_FULL_S42 | 0.950000 | 0.051964 | 4 |
| baseline | LARGE_V2_FULL_S42 | 0.946842 | 0.057436 | 2 |
| baseline | LARGE_V2_FULL_S43 | 0.945263 | 0.055790 | 3 |
| baseline | MEDIUM_FULL_S42 | 0.940526 | 0.062642 | 3 |
| baseline | MEDIUM_LORA_S42 | 0.938421 | 0.063398 | 4 |
| baseline | MEDIUM_LORA_S43 | 0.935263 | 0.065222 | 6 |
| baseline | MEDIUM_FULL_S43 | 0.926316 | 0.073942 | 5 |
| batch-4 control | SMALL_FULL_S42 | 0.878421 | 0.090804 | 7 |
| baseline | SMALL_FULL_S43 | 0.875263 | 0.095342 | 8 |
| SC | SMALL_FULL_S43 | 0.874211 | 0.093740 | 5 |
| SC | SMALL_FULL_S42 | 0.873158 | 0.093295 | 8 |
| baseline | SMALL_FULL_S42 | 0.872105 | 0.091293 | 8 |
| batch-4 control | SMALL_FULL_S43 | 0.845263 | 0.157049 | 32 |
| baseline | SMALL_LORA_S42 | 0.789474 | 0.126040 | 19 |
| baseline | SMALL_LORA_S43 | 0.782632 | 0.129021 | 21 |

### The two surfaces rank the six baseline arms differently, and the disagreement has a direction

| arm | validation mean | rank | Public mean | rank | move |
|---|---:|---:|---:|---:|---:|
| LARGE_V2_FULL | 0.894587 | 1 | 0.946052 | 2 | -1 |
| MEDIUM_FULL | 0.889601 | 2 | 0.933421 | 4 | -2 |
| LARGE_V2_LORA | 0.874644 | 3 | 0.953421 | 1 | **+2** |
| SMALL_FULL | 0.827635 | 4 | 0.873684 | 5 | -1 |
| MEDIUM_LORA | 0.820512 | 5 | 0.936842 | 3 | **+2** |
| SMALL_LORA | 0.715100 | 6 | 0.786053 | 6 | 0 |

Spearman between the two rankings is **0.60**. More telling than the magnitude is the sign:
**both arms that move up are LoRA arms, every arm that moves down is full SFT, and the SC
variant moves the same way.** LoRA at Medium also overtakes full SFT at Medium on Public
(`0.936842` against `0.933421`) — so the Large-v2 reversal reported above is not a single
capacity point.

The round's protocol ranks candidates on fixed validation and uses Public and OOD only as a
stability veto. That is the right design for avoiding leaderboard overfitting, and this report
does not propose changing a registered rule mid-round. It does record the consequence: **on
this round's evidence, validation systematically underrates the less aggressively fitted
regimes — LoRA, and the source-conditional curriculum — relative to the surface built from
competition data.** Anyone selecting a checkpoint to submit, rather than to compare, should
know that the registered rule would hand them LARGE_V2_FULL while Public prefers
LARGE_V2_LORA_S43 by 0.0087 tol2 and 11% CER.

## Pre-registered extension gates: 2 of 8 pass

Run with the round's own `extension_decision` ([`scripts/analysis/gates.py`](../scripts/analysis/gates.py)):

| decision | arm | mean CER improvement | failure |
|---|---|---|---|
| extend | MEDIUM_LORA | +0.011223 | — |
| extend | SC SMALL_FULL | +0.005398 | — |
| stop | LARGE_V2_FULL | +0.005932 | seed43 `replacement_regression` |
| stop | MEDIUM_FULL | +0.006573 | seed42 `severe_regression` |
| stop | LARGE_V2_LORA | +0.002458 | seed42 `severe_regression` |
| stop | SMALL_FULL | +0.005184 | seed42 `severe_regression` |
| stop | SMALL_LORA | +0.007589 | seed42 `severe_regression` |
| stop | SMALL_FULL (batch-4 control) | +0.006680 | seed42 `severe_regression` |

**No arm fails on CER.** Every rejection is a stability counter. The 2026-09-04 observation on
two arms reproduces across the full matrix. The consequence is worth stating plainly: the
strongest arm is barred from 5 epochs, while the arm with the largest CER improvement that
does qualify (MEDIUM_LORA) is one of the weakest in absolute terms. The gate is selecting
against the tail behaviour of strong arms, not against poor accuracy.

## Source-conditional curriculum: harmful on validation, helpful on Public

SC streams have two 4-row tail batches per epoch (official `6292 = 393*16 + 4`, external
`17012 = 1063*16 + 4`) while the baseline has one 8-row tail. `rank_microbatches` requires
entry sizes divisible by `batch * world`, so **SC arms must run at batch 4**; batch 8 raises
`ValueError: tail cannot be evenly partitioned by this topology` when resuming at the second
milestone. That made every earlier SC-vs-baseline comparison confounded by batch size.

A batch-4 baseline control (`p2-full-h100-b4ctl`, SMALL_FULL, both seeds) resolves it. Using
the round's own `select_validation` to pick the representative checkpoint
([`scripts/analysis/sel.py`](../scripts/analysis/sel.py)):

| comparison | seed 42 | seed 43 |
|---|---|---|
| SC minus batch-matched baseline, tol2 | **-0.017094** | **-0.022792** |
| SC minus batch-matched baseline, CER | +0.009085 | +0.004275 |
| batch-4 minus batch-8 baseline, tol2 | +0.001425 | -0.007123 |

The batch effect is small and changes sign across seeds; the SC effect is consistent in both
seeds and on both metrics. **On validation, the source-conditional curriculum degrades
SMALL_FULL.**

**On Public it does the opposite at Large-v2.** SC LARGE_V2_FULL scores `0.950000 / 0.950526`
against the baseline's `0.946842 / 0.945263` — ahead in both seeds — with CER `0.051964 /
0.052454` against `0.057436 / 0.055790`, 6-10% lower. Its own validation numbers were *behind*
the baseline by 0.0057 / 0.0014. At Small the Public comparison is inconclusive (`-0.0053 /
+0.0289`, and the batch-4 control's seed 43 is the runaway checkpoint described below). On OOD
as scored, SC is behind everywhere by 0.004-0.016.

So the earlier one-line reading — "SC is harmful" — holds only on validation and does not
survive the surface it was meant to generalise to. See the section below: this is the third
independent case in this round of validation ranking against Public.

Changing batch also changes `config_sha256` and trips the `preparation/config mismatch` check.
None of the preparation outputs (streams, audio receipts, model hashes, LM splits) depend on
batch or accumulation, so cloning a preparation directory and rebinding its `config_sha256` is
sufficient — there is no need to re-run the audit.

### Caveat: the batch-4 control's seed 43 blows up at the last checkpoint

| step | tol2 | CER | severe | replacement | repeated_runaway |
|---|---|---|---|---|---|
| 2186 | 0.770655 | 0.121740 | 33 | 4 | 0 |
| 2914 | 0.796296 | 0.106990 | 22 | 0 | 0 |
| 3643 | 0.821937 | 0.099081 | 18 | 0 | 0 |
| 4371 | 0.809117 | **0.136597** | 18 | **67** | **2** |

Steps 2186-3643 track the batch-8 baseline closely; step 4371 alone shows 67 replacement
characters and 2 repeated runaways, which inflates CER by 0.038 while tol2 moves only 0.013.
Comparing final checkpoints directly would misread this as a batch-size effect. The selection
rule above avoids it by picking 3643 for that seed. Smaller batches appear more prone to this
failure, which is worth a dedicated check rather than an inference from one seed.

## Where LARGE_V2_FULL's remaining error lives

[`scripts/analysis/errprof.py`](../scripts/analysis/errprof.py), seed 42 @ 4371, 702 validation
utterances, 641 total edits over 9356 reference characters:

- **380/702 utterances are exactly correct.**
- Composition: substitutions 70.8%, deletions 15.1%, insertions 14.0% — the model mishears
  characters; it is not dropping audio or hallucinating spans.
- Not concentrated: the worst 5 utterances hold 7.6% of all edits, the worst 50 (7.1% of the
  set) hold 37.9%. This is a long tail.
- Suspected audio/text misalignment (edit distance >= 80% of reference length): **2 utterances,
  3.1% of edits.** Annotation quality is not the bottleneck.
- Against SMALL_FULL on the same surface: 268 utterances wrong in both, 125 fixed by capacity,
  54 introduced by capacity. **The jointly-wrong utterances account for 89.7% of LARGE_V2_FULL's
  total edit mass.**

Nine tenths of the residual error is insensitive to capacity, which is a concrete explanation
for the +0.005 Medium -> Large step rather than a general "data ceiling" hypothesis. It points
the next round at the language-model / decoding / tokenisation side — the text, LM and fusion
stages this round registered and has not yet run — instead of more parameters.

## Reproduction notes

**`transformers==4.57.6` is required; 5.x changes the results.** Running SMALL_FULL on 5.6.2
moved all eight recorded metrics in the favourable direction (step 4371 tol2 0.84330/0.84900
against a recorded 0.821937/0.819088). A 50-step benchmark separates the two cheaply: step-25
loss is `4.034488` on 4.57.6 against `4.03569` in the original 4090 execution log (0.03%, the
expected H100-vs-4090 numerical residual) and `2.6604` on 5.6.2. The receipts from the 5.6.2
run are kept under `receipts/p2-full-h100-transformers562-EVIDENCE/` as evidence, not as
results. The library upgrade is a real improvement lead — roughly +0.02 to +0.03 tol2 and
-0.007 CER on whisper-small for free — but it must be evaluated as a controlled change, not
folded into a pre-registered round.

**Manifests reproduce byte-for-byte.** All four frozen manifests match their recorded SHAs
(`external73` 23304 rows, `validation` 702, `template_pre` 1900, `ood_panel` 2000). Two calling
conventions matter: `prepare_manifest.py` has no `--project-root` and writes `audio_path`
verbatim, so it must be called with relative paths from the asset root; and
`build_external_training_mixes.py` derives its sampling seed from the *position* of `0.73` in
the `--external-fraction` list, so the original round's third position must be reproduced
(`--external-fraction 0.25 --external-fraction 0.5 --external-fraction 0.73`).

**Public-surface zero-shot baseline reproduces.** Original whisper-small under D0_CURRENT on
the 1900-row public surface: tol2 `0.17736842`, identical to eight decimals against the
original execution log; CER `0.39943053` against `0.39920808` (5 edit operations out of 22477
characters, fp16). Two traps: `predict.py` cannot reproduce D0_CURRENT because its
`generation_kwargs` omit `no_repeat_ngram_size` and `repetition_penalty`, and `template_pre`
audio lives under `train_raw/` in nested subdirectories, so resolution needs a recursive
basename index rather than a single-level lookup.

**`configs/rounds/raw_winner_p2_full.json.orig4090`** is the pre-rebinding config: LARGE_V2_FULL
`accum 4 / world 4` and MEDIUM_FULL `accum 4 / world 2`, versus `accum 16 / world 1` and
`accum 8 / world 1` on a single H100. Nothing else differs.

**Batch size is the cheap accelerator.** LARGE_V2_FULL at batch 1 -> 4 goes from 9.93 to 2.71
seconds per step (3.7x) at 29.3 GB of 80 GB peak, taking 4371 steps from 12.1 to 3.3 hours.
`batch * accum` is fixed, so this only trades serial micro-batches for parallel ones.

## What this round closed, and what is left

Recorded plainly because the negative results are the bulk of the output.

**Closed — measured, not inferred:**

| line | result |
|---|---|
| The `external73` data recipe | At matched size it *loses* to RAW_WINNER: −0.026 validation, −0.020 Public |
| More capacity | Medium → Large-v2 is +0.005 tol2 for double the parameters |
| Source-conditional curriculum | No net win; helps Public at Large-v2, hurts validation and OOD |
| Training longer | 5 epochs buys +0.01 tol2 with CER unchanged — cosmetic |
| Cleaning the references | 2 of 702 utterances look misaligned; the data is already clean |
| Fixing the worst utterances | The error is a flat tail; the worst 50 hold 37.9% |
| Character 5-gram rescoring | Prior round selected λ = 0.0 — no gain |
| Static fusion (MBR/ROVER) | Prior round: validation MBR trails the best single component |
| N-best reranking | Prior round: 5-best oracle ceiling is +0.0028 tol2 |

**Open, and not yet spent:**

- **The library version.** Under `transformers==5.6.2`, SMALL_FULL scores `0.843305 / 0.849003`
  against `0.826211 / 0.829060` on 4.57.6 — **+0.017 / +0.020 tol2, CER 6% lower, severe down
  from 17/17 to 11/15**, with no algorithm or parameter change. This round pinned 4.57.6 so the
  pre-registered comparison stayed valid against the original execution log; that pin protects
  the *comparison*, not any model one might ship. It has never been combined with Large-v2.
- **Per-utterance routing.** The prior round's three-model oracle is 7.5% better in CER than the
  best single system while static fusion captured none of it. The complementarity is real and
  unexploited; the obstacle is the selector, and the character LM already failed in that role.
- **A Cantonese tokenizer.** Untouched. 29-35% of reference characters tokenize to multiple
  tokens, and 70.8% of the residual error is substitution, so this is at least aimed at the
  right side of the model.
- **Augmentation.** SpecAugment is explicitly disabled and no augmentation axis was ever tested.
- **Domain vocabulary coverage.** The substitutions are real Cantonese lexical confusions
  (`容积率→溶秩率`, `熨烫平整→运动屏净`). More audio covering those words would help; that is a
  data-acquisition cost, not a method.

The honest summary: **this round did not find a better method, it found a better model, and the
difference cost 6x the parameters.** Nine method-level lines are now closed. The gains that
remain unclaimed are a library upgrade and a model-size change, neither of which is a research
contribution.

## Not in this report

- SC LARGE_V2_FULL on Public and OOD at Medium and Small capacities — only Large-v2 and Small
  were run.
- Text/LM/fusion stages and the complete finalizer. `finalize_p2_full.py` is a partial
  collector by design.
- The original 4090 round's `artifacts/raw_winner_p2_full_v1/` receipts (~62.66 GPU-hours),
  which remain only on the original training server.

The analysis behind each table is under [`scripts/analysis/`](../scripts/analysis/); point it at
a run root with `P2_RUN_ROOT=<dir>` or a first argument. Scheduler scripts are site-specific and
are not part of this repository.
