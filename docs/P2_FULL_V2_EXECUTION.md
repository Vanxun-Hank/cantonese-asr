# Full P2 v2 execution record

## Material Passport

- Origin: academic-research-suite / experiment-agent; implementation of approved plan.
- Date: 2026-09-04.
- Base commit: `beb2396808dffdf2ffe275fd14a7777b9f99f550`.
- Round: `raw_winner_p2_full_v1`; recipe: `p2_full_four_gpu_v2`.
- Status: six training paths GPU-accepted; formal capacity runs underway.
  Full-round text/fusion/finalization remains incomplete. See dated entries below.

## Frozen design

### 2026-09-04 23:36 CST — Medium Full paired completion

Job2015 completed; seed43 endpoint and selected both4371: validation tol2
0.874643875, CER0.071184267, severe9, repetition/replacement/max0. Both Medium
seeds' actual rank0+rank1 receipts verify 69,912 exposures, three exact epoch
permutations and the final8 per epoch. Planner verifies six validation surfaces
per seed, including artifact hashes and row counts.

The paired extension decision is **false** despite mean CER improvement0.005451:
seed42 severe rises9→14 between its early/late representatives. Both Medium Full
seeds therefore stop at3 epochs under the preregistered strict stability rule.
No guardrail was relaxed based on good CER. Public/OOD do not enter this decision.
Medium43 endpoint Public/OOD queued independently; receipt
`artifacts/raw_winner_p2_full_v1/launch/endpoint_medium43_submission_2336.json`,
plan `epoch3_plan_2336.json` in the same directory. Existing arrays untouched.
Small LoRA jobs2020/2021 have started; Large LoRA2016/2017 continue. No extension
jobs submitted, and the remaining P2 tracks are not complete.

### 2026-09-04 23:28 CST — local finalizer integrity hardening

The partial collector now excludes invalid surfaces rather than attaching their
metrics after merely noting an error. It requires complete/integrity flags, split,
step, D0 decoder, frozen manifest hash, every required artifact hash, and exact
prediction/token row counts before a surface enters the matrix or selection.
Regression tests cover corrupted metrics, wrong split and wrong row counts;
14 focused finalizer/planner/core tests pass. This is local-only, not deployed
over the accepted training code. Full finalizer integration is still pending.

Live jobs2015/2016/2017 remain healthy; endpoint arrays2035/2036 are queued.

### 2026-09-04 23:05 CST — Medium Full seed42 epoch3 complete

Job2014 completed. Its six validation surfaces pass the endpoint planner's
identity, row-count and artifact-hash checks. Endpoint and selected point are
both4371: tol2=0.888888889, CER=0.072466866, severe14, true repeated runaway0,
replacement0, max-length0. Public/OOD are pending, not inferred from validation.
Array2036 queues the two deduplicated surfaces on gpu001 (concurrency1), using
`launch/epoch3_plan_2305.json` and
`launch/endpoint_medium42_submission_2305.json` under the round artifact directory.
Small array2035 remains queued; neither array was duplicated.

Medium seed43 is evaluating step3643; Large LoRA42/43 have completed validation
through1457 and are continuing training. No paired Medium extension decision is
made until seed43 completes. Disk free8.2TB. All four authorized GPUs remain in
use by this round; no unrelated jobs were changed.

Six independent original-OpenAI Small/Medium/Large-v2 Full/LoRA arms, seeds42/43.
All arms see external73 (23304 samples) exactly once per epoch. Global batch16,
last batch8, no padding/duplication/drop. Three epochs: 4371 steps/69912 exposures.
Maximum five: 7285 steps/116520 exposures. All arms use the same 7285-step cosine
horizon and365 warmup, including arms that stop at4371. Full LR1e-5, LoRA LR1e-4;
AdamW wd.01, gradient clipping1, BF16/TF32 and gradient checkpointing; no augmentation.
LoRA r8/alpha16/dropout.05, attention q/v only. FSDP1 full-shard, no CPU offload.

At epoch3, compare validation-selected representatives of steps2186/2914 against
3643/4371. Extend BOTH seeds of a family only when mean CER improves >=.001,
neither seed's CER worsens, each tol2 decline <=.005, no stability count increases,
and integrity passes. Never use Public/OOD for extension or ranking.

The primary matrix and capacity families for fusion are selected using only the
shared first3 epochs. Extended results are a separate unequal-budget supplement.
Eight or more epochs are not authorized.

## Implemented entrypoints

- `scripts/run_p2_full.py prepare`: immutable manifest/audio/model receipts, five
  epoch streams per seed, smoke streams and95/5 grouped train-only LM split.
  `--decode-audio` is required for PASS. Missing/corrupt/overlapping audio or
  oversized labels fail without modifying manifests.
- `scripts/train_p2_full.py`: explicit global optimizer batches and token-loss
  denominator, distributed FSDP, complete optimizer/RNG exports, segment receipts.
  Formal runs require separately verified GPU acceptance matching config/code.
- `scripts/p2_full_capacity_worker.py`: separate torchrun segments and external
  validation after trainer exit. Smoke:5 steps, resume to6, uninterrupted6, offline32;
  step6 uses the first eight IDs of the sixth diagnostic batch to exercise a tail
  update (88 actual draws from the frozen 96-row smoke pool; formal stream unchanged).
- `scripts/evaluate_p2_full.py`: D0 or native-five evaluation, all required error
  artifacts, validation loss, row checks, generation provenance and hash receipts.
- `scripts/train_p2_character_lm.py`: two-layer256/512 LSTM, train-only vocabulary,
  AdamW1e-3, wd.01, clip1, batch64,20 epochs, LM-dev NLL selection, separate seeds.
- `scripts/analyze_p2_full_text.py`: corrected5gram/LSTM static rerank and
  content-aligned MBR/ROVER; validation-only lambda/method selection; oracle marked.
- `scripts/finalize_p2_full.py`: **partial capacity collector**, intentionally cannot
  certify the complete round until all text/fusion/extensions/offline gates exist.

Legacy P2 artifacts are not replaced. Legacy D7 anchoring remains available to
reproduce old analyses; new round exclusively uses `P2_NATIVE5`. The corrected
LM/ROVER implementation is versioned by this branch, not presented as old evidence.

## Current verification and pending work

Local integration addition18:58 CST: `scripts/plan_p2_full_endpoints.py` constructs
validation-only selected3/epoch3 Public/OOD task lists, deduplicates equal points,
keeps every completed arm, verifies surface hashes/702 rows and records paired
extension decisions without submitting them. Missing evidence remains PARTIAL.
Two new regression tests (no premature selection; reject Public-as-validation)
plus existing selector suite:11 PASS. Not yet deployed or run against completed
epoch3 outputs; frozen training deployment remains untouched.

Local CPU tests cover epoch tails, topology ID reconstruction, global-token loss
scaling, paired extension, LSTM padding and RNG/optimizer restoration, ngram symbols,
ROVER duplicate insertion and exact native candidate retention. Local torch is not
the server CUDA build, so this is not GPU/FSDP acceptance.

Mandatory pending server gates: frozen-asset preparation, actual1/2/4-GPU
forward/backward/resume parity, all-rank GPU memory,50-step benchmarks and ETA,
formal acceptance issuance. Do not fabricate an acceptance JSON to bypass these.

Remaining completion integration: full finalizer (including extension/model-family
selection, all text analyses and packages), statistical/report/manuscript exports,
and offline research-package verification. These are NOT already completed outputs.

The main branch's old `test_raw_winner_p1_decode.py` references absent
`scripts/select_raw_winner_p1_decode.py`; record as a pre-existing collection
failure rather than importing unrelated dirty files or hiding the failure.

## Scheduling and safety

### Submitted preflight (2026-09-04)

18:45 CST Small Full epoch1(step1457) complete validation: seed42 tol2=.77350427,
CER=.10998290,severe28; seed43 tol2=.79772080,CER=.10934160,severe25.
Both repeated/replacement/max0. Both improve over their0.5epoch points; not
final selection. Training progressed to1575 and Medium Full to400, all finite.
Disk8.3TB free. No schedule mutation or extension triggered by this observation.

18:35 CST first complete adaptation validation (Small Full step729,0.5epoch):
seed42 tol2=.73076923 CER=.13413852 severe36 loss=.87824114;
seed43 tol2=.74928775 CER=.12494656 severe31 loss=.91456459.
Both receipts complete; predictions/token sidecars702/702; repeated/replacement/
max-length counts all0. These are early learning-curve points, not selected final
models or a comparison against engineered RAW69.49. Medium/Large step0 jobs2012/
2013 completed; Medium Full seed42 job2014 running, other arms queued normally.

18:25 CST: Small base job2011 completed all three surfaces. Original Small D0
step0 tol2/CER: validation0.16381766/0.39311672, Public0.17736842/0.39920808,
OOD0.1535/0.53798913 (these are original OpenAI weights, not RAW69.49).
Formal Small Full seeds42/43 jobs2018/2019 are RUNNING; both logged step25,
finite losses4.03569/4.12499 and LR6.575342e-7. No checkpoint validation yet.
Medium/Large base evaluations still running; other capacity jobs remain pending.

18:16 CST: verifier2009 COMPLETED/PASS for all six deterministic arms. Model and
optimizer tolerances unchanged; Python/NumPy/torch/CUDA RNG exact, real tail8,
per-rank Spec-off and resource evidence verified. Gate at
`launch/deterministic1/gpu_acceptance.json`, code SHA
`dd9815f73ef3141d5e0a909857027460b9a0cc89f16695b79a067fe58279e0c8`.
Measured50-step extrapolation:62.66 GPU-hours for all12 epoch3 trainings,
104.43 GPU-hours if all extend to5. Ideal four-card training-only lower bounds
15.66/26.11 hours; NOT end-to-end ETA, excludes saves, evaluations and queue gaps.

Formal submission receipt `launch/formal_epoch3/submission.json`:
base step0 validation/Public/OOD jobs2011 small,2012 medium,2013 large-v2;
Medium Full seeds42/43 jobs2014/2015; Large LoRA2016/2017;
Small Full2018/2019; Small LoRA2020/2021; Medium LoRA2022/2023;
Large Full2024/2025. Every training job depends on its model's complete step0
evaluation and uses the accepted immutable deterministic1 code/output root.
All stop at4371; NO extension jobs submitted. Failures preserve evidence;
no automatic requeue. These are submitted jobs, not claims of epoch3 completion.
Full finalizer/text/fusion/package/report integration still needs completion.

Deterministic revalidation jobs (new deployment/output suffix `-deterministic1`):
Medium Full1997/1998, Large LoRA1999/2000, Medium LoRA2001/2002,
Large Full2003/2004, Small Full2005/2006, Small LoRA2007/2008 (smoke/benchmark).
CPU verifier2009 depends on all six benchmarks. Receipt/logs under
`artifacts/raw_winner_p2_full_v1/launch/deterministic1/`; verifier output
`gpu_acceptance.json`. No formal jobs submitted. Do not duplicate these jobs.

17:44 CST numerical reproducibility revision: new immutable deployment
`worktrees/raw-winner-p2-full-v1-deterministic1`. All training phases now explicitly
enable deterministic algorithms, cuDNN deterministic and benchmark=false, cuBLAS
workspace=:4096:8 before CUDA initialization. BF16/TF32 and all statistical recipe
settings remain unchanged. Resource receipts record these switches; verifier
requires them with the original tolerances. Local23 tests PASS. Because the
execution setting affects every arm, all six paths require new smoke/benchmark
evidence; old work is retained as diagnostic, not reused as a deterministic PASS.
This is one evidence-driven revalidation, not an infinite retry policy.

Job1996 COMPLETED: `model_exact=true`, `optimizer_exact=true` with torch.equal.
Deterministic CUDA settings eliminate the first3-step independent-run difference
in this Small Full diagnostic. This supports a numerical reproducibility cause,
not proof that all resume paths are correct. Next: carry explicit deterministic
execution settings into a new version, then test full resume/tail paths with
unchanged tolerances; still no formal training authorization receipt.

17:35 CST: deterministic diagnostic1995 COMPLETED. Both independent runs now
report exactly the same step3 loss6.104610443115234. CPU-only job1996 compares
model and optimizer tensors using torch.equal (no loosened tolerances); log
`launch/determinism-compare-1996.log`. Matching loss alone is not acceptance;
resume/tail/LoRA/FSDP under these settings still need verification if confirmed.

17:34 CST: bounded numerical diagnostic job1995 submitted on gpu001, one GPU,
30-minute limit, no requeue. Two independent SMALL_FULL seed42 first3-step runs
under deterministic CUDA algorithms, cuDNN deterministic/benchmark false, cuBLAS
workspace `:4096:8`; BF16/TF32, samples and LR unchanged. Wrapper is separate
`artifacts/raw_winner_p2_full_v1/launch/diagnose_p2_determinism.py`; it logs its own
SHA and the unchanged retry1 trainer SHA. Outputs in
`outputs/raw-winner-p2-determinism-diagnostic-v1/{a,b}`; log
`artifacts/raw_winner_p2_full_v1/launch/determinism-1995.log`.
This is diagnostic evidence only, not the approved formal gate or a tolerance
change. Compare first3 losses and model/optimizer tensors before any next action.

17:23 CST diagnostic refinement: Large1989/1990 completed, verifier1993 failed
optimizer comparison (first reported tensor absolute delta1.8057e-4). Read-only
comparison of all six arms shows step6 global IDs/token denominators identical
and saved torch/CUDA RNG equal on every rank. More importantly, independent
interrupted and uninterrupted runs ALREADY differ before saving/resume: Small
Full/Small LoRA/Medium Full/Large Full losses match at steps1–2 and diverge at
step3. Thus the current comparison does not isolate a checkpoint restoration bug;
baseline run-to-run numerical reproducibility must be investigated first.
Next bounded diagnostic should compare independent short runs with deterministic
CUDA algorithms/workspace settings, preserving BF16/TF32 and all data/LR choices,
before changing checkpoint code or relaxing tolerances. No formal training
submitted; no GPU acceptance issued. No further retry was launched in this check.

17:14 CST acceptance failure:1991 stopped on SMALL_FULL optimizer-state comparison
after model tensor comparisons passed. First failing tensor max absolute delta
6.8593e-6 (configured atol1e-6, rtol1e-5).1992 likewise failed Medium Full
optimizer comparison, first tensor max absolute delta0.00140055. These values are
for the first reported failing tensor, NOT a global maximum. No acceptance JSON
was issued. Other single-GPU arms were not reached by1991; do not mark them failed
or accepted. Investigate resumed vs uninterrupted optimizer state/RNG/order and
numeric execution before any formal training. Do not silently loosen tolerances.

17:12 CST: Medium retry1987/1988 COMPLETED; Large retry1989 RUNNING beyond
the staging race, benchmark1990 dependent. Independent CPU-only acceptance jobs:
1991 verifies four single-GPU arms against their original immutable code;
1992 verifies Medium Full against retry1;1993 verifies Large Full after1990.
Verifier snapshot: `artifacts/raw_winner_p2_full_v1/launch/verify_smoke_subset_v1.py`.
Outputs: `launch/acceptance-{single,medium,large}-v1.json` with explicit subsets,
evidence code roots and hashes; a subset PASS is not full-round acceptance.
Logs: `launch/verify-{single,medium,large}-%j.log`. Do not duplicate these jobs.
No formal training submitted; tensor/optimizer/RNG parity still pending.

17:05 CST bounded recovery: added rank0-only exclusive staging creation with
collective status broadcast; late peers no longer re-check rank0's directory.
Two regression tests cover the original race and collective preservation of
pre-existing evidence. Local selected suite23 PASS; server focused suite14 PASS.
New immutable deployment `worktrees/raw-winner-p2-full-v1-retry1`, new output root
`outputs/raw-winner-p2-full-v1-retry1`. Only the failed Full arms were resubmitted:
Medium smoke1987/benchmark1988; Large smoke1989/benchmark1990. Submission receipt
`artifacts/raw_winner_p2_full_v1/launch/retry1/preflight_submission.json`.
Old deployment/results are untouched. No formal training submitted. Acceptance
must retain per-arm code provenance (old single-GPU evidence vs new Full retry);
never forge one common fingerprint or label job completion as parity PASS.

Heartbeat observation at16:51 CST:1973 completed/PASS. Small Full and all three
LoRA smoke/benchmark jobs completed (1976–1979,1982–1985); completion is not yet
resume-parity acceptance. Medium Full1974 and Large Full1980 failed with
`FileExistsError` at trainer line173: every rank checks the staging path while
rank0 can already have created it. Medium reached the first five updates and
failed during resume setup; Large failed during initial setup. This is a
distributed directory-initialization race, not evidence of CUDA OOM. Their
dependent benchmarks1975/1981 were cancelled by the invalid-dependency policy.
All evidence is retained. No formal training or replacement jobs submitted in
this heartbeat. Fix requires rank0-only initialization and collective status
propagation, a regression test, and a new isolated retry version/output root;
do not delete staging or bypass checkpoint protection.

- CPU-only asset/audio audit: job1973, observed RUNNING on gpu001, zero GPUs.
- Medium Full smoke/benchmark:1974/1975 (two GPUs).
- Large-v2 LoRA:1976/1977 (one GPU).
- Medium LoRA:1978/1979 (one GPU).
- Large-v2 Full:1980/1981 (four GPUs exclusively).
- Small Full:1982/1983 (one GPU).
- Small LoRA:1984/1985 (one GPU).

All smoke jobs depend on1973 success; each benchmark depends on its own smoke.
Slurm enforces the four-GPU capacity. Invalid dependencies terminate the dependent
preflight instead of launching without evidence; there is no automatic requeue.
Submission receipt: `/home/bolin/cantonese-asr/artifacts/raw_winner_p2_full_v1/launch/preflight_submission.json`.
Formal 3-epoch jobs have NOT been submitted. CPU tests:21 PASS on both Mac and
server; no GPU acceptance has yet been issued. Heartbeat `p2` checks every10minutes
and reports meaningful changes only. Full report/package integration remains pending.

The unrelated four-GPU job1882 subsequently left gpu001; latest allocation at
submission was12 CPUs/24GiB and zero GPUs, from CPU-only jobs1962 and1973.
No unrelated job was modified by this task.

Only one authorized four-GPU node. Live inspection on2026-09-04 found gpu001's
four GPUs allocated to unrelated job1882; do not cancel/hold/release it. Inspect
the current queue again before submission. GPU availability is not yet granted
by a successful SSH connection. CPU-only preparation may use Slurm with zero GPUs.

`slurm/p2_full_capacity.slurm` requires explicit `P2_CODE_ROOT`, `P2_ASSET_ROOT`,
`P2_PREPARATION`, `P2_RUN_ROOT`, `P2_ARM`, `P2_SEED`, `P2_PHASE`, plus sbatch node/GRES.
It does not self-submit or infer permission to use another node. All GPU jobs are
offline. Formal training is gated; failures keep staging/logs and are not silently retried.

Training-only extrapolation131–218 GPU-hours is provisional, not end-to-end ETA.
Do not submit to the competition or push GitHub/Hugging Face. Do not overwrite69.49.
# 2026-09-04 19:28 CST — first paired epoch3 completion

Small Full jobs 2018/2019 completed. The observed rank0 receipts contain exactly
4371 optimizer steps and 69,912 draws per seed. Each of the three epochs contains
all 23,304 indices exactly once; the final eight samples are retained.

| Seed | Epoch3 tol2 | Epoch3 CER | Selected step | Selected tol2 | Selected CER |
|---|---:|---:|---:|---:|---:|
| 42 | 0.821937 | 0.094378 | 4371 | 0.821937 | 0.094378 |
| 43 | 0.819088 | 0.096516 | 2914 | 0.829060 | 0.096836 |

The validation-only paired extension gate rejects Small Full: seed42 severe
increases from 18 to 19, and seed43 late-representative tol2 drops by more than
0.005. The mean CER improvement alone does not override these guards. Neither
seed will extend. This is not a claim of convergence.

Selection and surface-file hashes were checked by the new endpoint planner.
Server plan: `artifacts/raw_winner_p2_full_v1/launch/epoch3_plan_1925.json`.
Public/OOD endpoint and selected evaluations are deduplicated into six tasks.
Job array **2035**, tasks0–5, concurrency1, is queued on gpu001 only; no training
or unrelated job was interrupted. Submission receipt:
`artifacts/raw_winner_p2_full_v1/launch/endpoint_submission_1925.json`.
The independent runner verifies checkpoint identity and all model-file hashes
before inference, and refuses existing output directories. It does not alter the
accepted formal training deployment. Local planner/core tests: 12 passed.

Other ten capacity arms are running or queued. Text/LM/fusion and full finalizer
integration remain incomplete; the overall P2 experiment is not finished.
