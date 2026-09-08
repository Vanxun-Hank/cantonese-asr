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

**This holds on validation only** — on Public the Large-v2 ranking reverses and on OOD it
vanishes. See the surfaces section below before drawing a conclusion about LoRA.

## Public and OOD surfaces, step 4371

D0_CURRENT decoding, public = 1900 rows of `template_pre`, OOD = the 2000-row v1 panel.
Receipts under [`reports/p2_full_converged/receipts/surfaces/`](p2_full_converged/receipts/surfaces/).

| arm | public tol2 s42/s43 | public CER s42/s43 | OOD tol2 s42/s43 | OOD CER s42/s43 |
|---|---|---|---|---|
| LARGE_V2_LORA | **0.951579 / 0.955263** | **0.050719 / 0.050985** | 0.360500 / 0.355500 | 0.325491 / 0.327482 |
| LARGE_V2_FULL | 0.946842 / 0.945263 | 0.057436 / 0.055790 | 0.358500 / 0.359000 | 0.324744 / 0.326238 |
| MEDIUM_FULL | 0.940526 / 0.926316 | 0.062642 / 0.073942 | 0.353500 / 0.347000 | 0.330595 / 0.330263 |
| SMALL_FULL | 0.872105 / 0.875263 | 0.091293 / 0.095342 | 0.321500 / 0.323500 | 0.346902 / 0.345948 |

### LoRA's validation deficit does not transfer to the other surfaces

On validation, LARGE_V2_LORA trails LARGE_V2_FULL by 0.020 tol2 (`0.877493/0.871795` against
`0.894587/0.894587`). **On Public the ranking reverses**: LoRA leads by +0.005 to +0.010 tol2
with 11% lower CER, in both seeds. On OOD the two are indistinguishable — 0.3555-0.3605 against
0.3585-0.3590, with CER equal to three decimals.

Both arms were decoded from the same step, the same decoder and the same manifest hash, so this
is a surface effect and not a checkpoint mix-up. The natural reading is that full SFT is buying
its validation margin by fitting the validation split's distribution, and that margin does not
exist off that split. The claim "LoRA does not close the gap" is therefore true *only* on
validation, and the section above should be read with that restriction.

This also matters for cost: LoRA trains 0.254% of the parameters at 6.58 GB peak.

### OOD remains the unsolved surface

Every arm sits at tol2 0.32-0.36 with CER 0.32-0.35, against 0.87-0.96 / 0.05-0.10 on Public,
and 457-529 of 2000 utterances are flagged severe. Capacity barely moves it: Small Full to
Large-v2 Full is +0.037 tol2 on OOD against +0.075 on Public. Replacement characters and
repeated runaways are essentially absent on both surfaces (at most 1), so this is accuracy on
genuinely out-of-domain audio, not decoding instability.

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

## Source-conditional curriculum: harmful, and it is not the batch confound

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
seeds and on both metrics. **The source-conditional curriculum degrades SMALL_FULL.**

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

## Not in this report

- Public and OOD for MEDIUM_LORA and SMALL_LORA — not evaluated on those surfaces.
- SC LARGE_V2_FULL — training, both seeds.
- Text/LM/fusion stages and the complete finalizer. `finalize_p2_full.py` is a partial
  collector by design.
- The original 4090 round's `artifacts/raw_winner_p2_full_v1/` receipts (~62.66 GPU-hours),
  which remain only on the original training server.

The analysis behind each table is under [`scripts/analysis/`](../scripts/analysis/); point it at
a run root with `P2_RUN_ROOT=<dir>` or a first argument. Scheduler scripts are site-specific and
are not part of this repository.
