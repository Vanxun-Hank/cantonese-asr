# Round 3 MDCC Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run four reproducible Whisper-small Full-SFT trials that keep every existing evaluation boundary fixed while increasing unique MDCC publisher-train coverage from the current 73% mixture through the complete accepted train pool.

**Architecture:** Extend the existing deterministic Manifest builder with nested, no-replacement fractions and an all-train output. Prepare the four immutable Manifests in one CPU Slurm job, then run one independent single-GPU trial per Manifest in a four-task Slurm array. Reuse the existing checkpoint evaluator, validation-only selector and reporting pipeline; only the train Manifest changes.

**Tech Stack:** Python 3.11, JSONL, pytest 8.4.2, Bash/Slurm, PyTorch 2.6, Transformers 4.57.6, four RTX 4090 GPUs.

---

## File map

- Modify `scripts/build_external_training_mixes.py`: build nested external fractions and the complete accepted publisher-train mixture.
- Modify `tests/test_build_external_training_mixes.py`: prove nesting, MDCC fill behavior, all-train coverage and deterministic reruns.
- Create `slurm/prepare_external_round3.slurm`: construct and validate Round 3 Manifests from existing QC-approved inputs.
- Create `slurm/external_round3.slurm`: map four immutable Manifests to four independent GPU trials.
- Create `slurm/report_external_round3.slurm`: generate Round 3-only and all-experiment reports.
- Create `tests/test_round3_slurm.py`: statically verify array-to-Manifest/output mapping and fixed training controls.
- Modify `slurm/final_candidate_full_ood.slurm`: include eligible Round 3 validation selections in final candidate selection.

### Task 1: Specify nested MDCC coverage

**Files:**
- Modify: `tests/test_build_external_training_mixes.py`
- Test: `tests/test_build_external_training_mixes.py`

- [ ] **Step 1: Write the failing nested-coverage test**

Append this test:

```python
def test_builds_nested_mdcc_coverage_and_all_train_mix(tmp_path: Path) -> None:
    project_root = Path(__file__).parents[1]
    official_path = tmp_path / "official.jsonl"
    cv_path = tmp_path / "cv.jsonl"
    mdcc_path = tmp_path / "mdcc.jsonl"
    official = rows("o", 10, "official")
    common_voice = rows("cv", 3, "common_voice_26_zh_HK")
    mdcc = rows("md", 30, "mdcc")
    write_jsonl(official_path, official)
    write_jsonl(cv_path, common_voice)
    write_jsonl(mdcc_path, mdcc)
    output = tmp_path / "round3"
    command = [
        sys.executable,
        "scripts/build_external_training_mixes.py",
        "--official-train", str(official_path),
        "--common-voice", str(cv_path),
        "--mdcc", str(mdcc_path),
        "--output-dir", str(output),
        "--seed", "42",
        "--external-fraction", "0.5",
        "--external-fraction", "0.6",
        "--external-fraction", "0.7",
        "--nested-fractions",
        "--include-all-external",
    ]
    subprocess.run(command, cwd=project_root, check=True, capture_output=True, text=True)

    mix50 = read_jsonl(output / "external50.jsonl")
    mix60 = read_jsonl(output / "external60.jsonl")
    mix70 = read_jsonl(output / "external70.jsonl")
    all_train = read_jsonl(output / "all_train.jsonl")
    report = json.loads((output / "mix_report.json").read_text(encoding="utf-8"))

    assert not (output / "external25.jsonl").exists()
    assert [len(mix50), len(mix60), len(mix70), len(all_train)] == [20, 25, 33, 43]
    assert report["sampling_policy"]["nested_external_fractions"] is True
    assert report["manifests"]["external50"]["sources"] == {
        "common_voice_26_zh_HK": 3, "mdcc": 7, "official": 10
    }
    assert report["manifests"]["external60"]["sources"] == {
        "common_voice_26_zh_HK": 3, "mdcc": 12, "official": 10
    }
    assert report["manifests"]["external70"]["sources"] == {
        "common_voice_26_zh_HK": 3, "mdcc": 20, "official": 10
    }

    official_ids = {str(row["id"]) for row in official}
    external_ids = [
        {str(row["id"]) for row in mix} - official_ids
        for mix in (mix50, mix60, mix70)
    ]
    assert external_ids[0] < external_ids[1] < external_ids[2]
    assert {str(row["id"]) for row in all_train} == {
        str(row["id"]) for row in official + common_voice + mdcc
    }

    first = {path.name: path.read_bytes() for path in output.glob("*.jsonl")}
    subprocess.run(command, cwd=project_root, check=True, capture_output=True, text=True)
    assert {path.name: path.read_bytes() for path in output.glob("*.jsonl")} == first
```

- [ ] **Step 2: Run the test and verify the new CLI is missing**

Run on the server environment:

```bash
/home/bolin/envs/cantonese-asr-whisper/bin/python -m pytest \
  tests/test_build_external_training_mixes.py::test_builds_nested_mdcc_coverage_and_all_train_mix -q
```

Expected: FAIL because `--nested-fractions` and `--include-all-external` are not accepted.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_build_external_training_mixes.py
git commit -m "test: specify nested MDCC coverage manifests"
```

### Task 2: Implement deterministic nested and all-train Manifests

**Files:**
- Modify: `scripts/build_external_training_mixes.py`
- Test: `tests/test_build_external_training_mixes.py`

- [ ] **Step 1: Make explicit fraction arguments replace defaults**

Use `default=None` and resolve defaults only when no flag was supplied:

```python
parser.add_argument(
    "--external-fraction",
    type=float,
    action="append",
    default=None,
    help="External fraction of final epoch rows; may be repeated.",
)
```

```python
fractions = list(dict.fromkeys(args.external_fraction or [0.25, 0.5]))
```

- [ ] **Step 2: Add the two Round 3 CLI switches**

```python
parser.add_argument(
    "--nested-fractions",
    action="store_true",
    help=(
        "Use one deterministic per-source order so every smaller external "
        "fraction is a strict subset of larger feasible fractions."
    ),
)
parser.add_argument(
    "--include-all-external",
    action="store_true",
    help="Also write all_train.jsonl with every accepted publisher-train row.",
)
```

- [ ] **Step 3: Add deterministic per-source prefix sampling**

```python
def sample_sources_prefix(
    sources: list[list[dict[str, Any]]], total: int, seed: int
) -> list[dict[str, Any]]:
    counts = balanced_counts(total, [len(source) for source in sources])
    sampled: list[dict[str, Any]] = []
    for source_index, (source, count) in enumerate(zip(sources, counts)):
        ordered = sorted(source, key=lambda row: str(row["id"]))
        random.Random(seed + 1009 * source_index).shuffle(ordered)
        sampled.extend(ordered[:count])
    return sampled
```

- [ ] **Step 4: Use one prefix order across all requested fractions**

Replace the fraction-loop sampling branch with:

```python
if args.nested_fractions:
    external = sample_sources_prefix(
        [common_voice, mdcc], external_count, args.seed + 50_000
    )
else:
    external = sample_sources(
        [common_voice, mdcc], external_count, args.seed + offset * 10_000
    )
```

- [ ] **Step 5: Write the complete accepted train mixture**

Insert before the pre-adaptation Manifest:

```python
if args.include_all_external:
    all_train_path = args.output_dir / "all_train.jsonl"
    all_train_rows = shuffled(
        official + common_voice + mdcc,
        args.seed + 80_000,
    )
    write_jsonl(all_train_path, all_train_rows)
    summary = manifest_summary(all_train_path, all_train_rows)
    summary["requested_external_fraction"] = "all"
    summary["actual_external_fraction"] = (
        len(common_voice) + len(mdcc)
    ) / len(all_train_rows)
    manifests["all_train"] = summary
```

Add these report fields:

```python
"nested_external_fractions": args.nested_fractions,
"all_external_rows_included": args.include_all_external,
```

- [ ] **Step 6: Run focused and existing regression tests**

```bash
/home/bolin/envs/cantonese-asr-whisper/bin/python -m pytest \
  tests/test_build_external_training_mixes.py -q
```

Expected: `3 passed`.

- [ ] **Step 7: Commit the implementation**

```bash
git add scripts/build_external_training_mixes.py
git commit -m "feat: build nested MDCC coverage manifests"
```

### Task 3: Prepare and validate Round 3 Manifests

**Files:**
- Create: `slurm/prepare_external_round3.slurm`

- [ ] **Step 1: Create the preparation job**

Create the file with:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=canto-ext-r3-data
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/bolin/cantonese-asr/logs/%x-%j.out

source /home/bolin/cantonese-asr/slurm/common.sh

OUTPUT=artifacts/manifests/external/round3
"$PYTHON" scripts/build_external_training_mixes.py \
  --official-train artifacts/manifests/train.jsonl \
  --common-voice artifacts/manifests/external/v1/common_voice_train.jsonl \
  --mdcc artifacts/manifests/external/v1/mdcc_train.jsonl \
  --output-dir "$OUTPUT" \
  --seed 42 \
  --external-fraction 0.73 \
  --external-fraction 0.80 \
  --external-fraction 0.85 \
  --nested-fractions \
  --include-all-external

"$PYTHON" - <<'PY'
import json
from pathlib import Path

from cantonese_asr.io import read_jsonl

root = Path("artifacts/manifests/external/round3")
report = json.loads((root / "mix_report.json").read_text(encoding="utf-8"))
names = ("external73", "external80", "external85", "all_train")
forbidden_paths = (
    Path("artifacts/manifests/validation.jsonl"),
    Path("artifacts/manifests/public_excluded.jsonl"),
    Path("artifacts/manifests/ood/v1/ood_all.jsonl"),
)
forbidden_ids = {
    str(row["id"])
    for path in forbidden_paths
    for row in read_jsonl(path)
}
ids = {}
for name in names:
    rows = read_jsonl(root / f"{name}.jsonl")
    row_ids = {str(row["id"]) for row in rows}
    if len(row_ids) != len(rows):
        raise SystemExit(f"{name} contains duplicate IDs")
    overlap = row_ids & forbidden_ids
    if overlap:
        raise SystemExit(f"{name} contains {len(overlap)} forbidden evaluation IDs")
    ids[name] = row_ids
    if report["manifests"][name]["rows"] != len(rows):
        raise SystemExit(f"{name} count disagrees with mix_report.json")

official = {str(row["id"]) for row in read_jsonl(Path("artifacts/manifests/train.jsonl"))}
external = {name: values - official for name, values in ids.items()}
if not (external["external73"] < external["external80"] < external["external85"]):
    raise SystemExit("Round 3 fractions are not strict nested subsets")
if ids["all_train"] != (
    official
    | {str(row["id"]) for row in read_jsonl(Path(
        "artifacts/manifests/external/v1/common_voice_train.jsonl"
    ))}
    | {str(row["id"]) for row in read_jsonl(Path(
        "artifacts/manifests/external/v1/mdcc_train.jsonl"
    ))}
):
    raise SystemExit("all_train does not contain every accepted train ID exactly once")
print(json.dumps(report, ensure_ascii=False, indent=2))
PY
```

- [ ] **Step 2: Check shell syntax**

```bash
bash -n slurm/prepare_external_round3.slurm
```

Expected: exit code 0 with no output.

- [ ] **Step 3: Commit**

```bash
git add slurm/prepare_external_round3.slurm
git commit -m "feat: prepare round 3 MDCC coverage data"
```

### Task 4: Define the four independent GPU trials

**Files:**
- Create: `slurm/external_round3.slurm`
- Create: `tests/test_round3_slurm.py`

- [ ] **Step 1: Write a static Slurm mapping test**

```python
from pathlib import Path


def test_round3_slurm_maps_four_fixed_manifests() -> None:
    text = Path("slurm/external_round3.slurm").read_text(encoding="utf-8")
    for value in (
        "#SBATCH --array=0-3%4",
        "external73.jsonl",
        "external80.jsonl",
        "external85.jsonl",
        "all_train.jsonl",
        "--learning-rate 2e-5",
        "--scheduler cosine",
        "--epochs 3",
        "--batch-size 8",
        "--gradient-accumulation-steps 2",
        "--seed 42",
        "--generation-max-length 225",
    ):
        assert value in text
```

- [ ] **Step 2: Run it and verify the Slurm file is missing**

```bash
/home/bolin/envs/cantonese-asr-whisper/bin/python -m pytest \
  tests/test_round3_slurm.py -q
```

Expected: FAIL with `FileNotFoundError: slurm/external_round3.slurm`.

- [ ] **Step 3: Create the array job**

```bash
#!/usr/bin/env bash
#SBATCH --job-name=canto-ext-r3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/home/bolin/cantonese-asr/logs/%x-%A_%a.out

source /home/bolin/cantonese-asr/slurm/common.sh

INDEX=${SLURM_ARRAY_TASK_ID:?Missing array task ID}
MANIFESTS=(
  artifacts/manifests/external/round3/external73.jsonl
  artifacts/manifests/external/round3/external80.jsonl
  artifacts/manifests/external/round3/external85.jsonl
  artifacts/manifests/external/round3/all_train.jsonl
)
NAMES=(
  ext-r3-external73-control
  ext-r3-external80-mdcc
  ext-r3-external85-mdcc
  ext-r3-all-train-mdcc-max
)

MANIFEST=${MANIFESTS[$INDEX]}
TRIAL=${NAMES[$INDEX]}
RUN_DIR="outputs/external-round3/$TRIAL"

"$PYTHON" train.py \
  --model artifacts/models/whisper-small \
  --train-manifest "$MANIFEST" \
  --validation-manifest artifacts/manifests/validation.jsonl \
  --train-probe-manifest artifacts/manifests/train_probe.jsonl \
  --project-root "$PROJECT_DIR" \
  --output-dir "$RUN_DIR" \
  --trial-name "$TRIAL" \
  --learning-rate 2e-5 \
  --scheduler cosine \
  --epochs 3 \
  --batch-size 8 \
  --eval-batch-size 4 \
  --gradient-accumulation-steps 2 \
  --warmup-ratio 0.05 \
  --weight-decay 0.01 \
  --logging-steps 25 \
  --num-workers 4 \
  --save-total-limit 3 \
  --seed 42 \
  --bf16 \
  --tf32

"$PYTHON" scripts/evaluate_checkpoints.py \
  --run-dir "$RUN_DIR" \
  --audio-dir "$PROJECT_DIR" \
  --validation-manifest artifacts/manifests/validation.jsonl \
  --train-probe-manifest artifacts/manifests/train_probe.jsonl \
  --ood-panel-manifest artifacts/manifests/ood/v1/ood_panel.jsonl \
  --batch-size 8 \
  --num-workers 4 \
  --generation-max-length 225

SELECTION_STATUS=0
"$PYTHON" scripts/select_best_checkpoint.py \
  --run-dir "$RUN_DIR" \
  --baseline-metrics outputs/zero-shot/validation/metrics.json \
  --max-cer 0.1163 \
  --min-sentence-accuracy 0.8219 \
  --output "$RUN_DIR/checkpoint_selection.json" || SELECTION_STATUS=$?

"$PYTHON" scripts/plot_experiments.py \
  --outputs-root "$RUN_DIR" \
  --data-report artifacts/manifests/data_report.json \
  --ood-data-report artifacts/manifests/ood/v1/data_report.json \
  --report-dir "artifacts/reports/experiments/round3/by_trial/$TRIAL"

if [[ "$SELECTION_STATUS" -ne 0 ]]; then
  echo "No checkpoint passed the fixed validation guardrails." >&2
  exit "$SELECTION_STATUS"
fi
```

- [ ] **Step 4: Run syntax and mapping tests**

```bash
bash -n slurm/external_round3.slurm
/home/bolin/envs/cantonese-asr-whisper/bin/python -m pytest \
  tests/test_round3_slurm.py -q
```

Expected: shell check exits 0 and pytest reports `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add slurm/external_round3.slurm tests/test_round3_slurm.py
git commit -m "feat: add four round 3 MDCC coverage trials"
```

### Task 5: Add Round 3 reports and final candidate inputs

**Files:**
- Create: `slurm/report_external_round3.slurm`
- Modify: `slurm/final_candidate_full_ood.slurm`
- Modify: `tests/test_round3_slurm.py`

- [ ] **Step 1: Create the report job**

```bash
#!/usr/bin/env bash
#SBATCH --job-name=canto-ext-r3-report
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/bolin/cantonese-asr/logs/%x-%j.out

source /home/bolin/cantonese-asr/slurm/common.sh

"$PYTHON" scripts/plot_experiments.py \
  --outputs-root outputs/external-round3 \
  --data-report artifacts/manifests/data_report.json \
  --ood-data-report artifacts/manifests/ood/v1/data_report.json \
  --report-dir artifacts/reports/experiments/round3

"$PYTHON" scripts/plot_experiments.py \
  --outputs-root outputs \
  --data-report artifacts/manifests/data_report.json \
  --ood-data-report artifacts/manifests/ood/v1/data_report.json \
  --report-dir artifacts/reports/experiments/all
```

- [ ] **Step 2: Add only eligible Round 3 selections to final selection**

Before the existing `select_global_candidate.py` call, add:

```bash
ROUND3_SELECTIONS=()
for path in \
  outputs/external-round3/ext-r3-external73-control/checkpoint_selection.json \
  outputs/external-round3/ext-r3-external80-mdcc/checkpoint_selection.json \
  outputs/external-round3/ext-r3-external85-mdcc/checkpoint_selection.json \
  outputs/external-round3/ext-r3-all-train-mdcc-max/checkpoint_selection.json
do
  if [[ -f "$path" ]] && "$PYTHON" - "$path" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
selected = report.get("selected")
raise SystemExit(0 if isinstance(selected, dict) and selected.get("eligible") is True else 1)
PY
  then
    ROUND3_SELECTIONS+=(--selection "$path")
  fi
done
```

Add the array expansion before `--output "$SELECTION"` in the existing command:

```bash
  "${ROUND3_SELECTIONS[@]}" \
  --output "$SELECTION"
```

This preserves every failed Round 3 selection report for diagnosis while preventing one
ineligible trial from blocking packaging of another eligible trial.

- [ ] **Step 3: Extend the static test**

```python
def test_round3_selections_feed_final_candidate() -> None:
    text = Path("slurm/final_candidate_full_ood.slurm").read_text(encoding="utf-8")
    for trial in (
        "ext-r3-external73-control",
        "ext-r3-external80-mdcc",
        "ext-r3-external85-mdcc",
        "ext-r3-all-train-mdcc-max",
    ):
        assert f"outputs/external-round3/{trial}/checkpoint_selection.json" in text
```

- [ ] **Step 4: Run checks**

```bash
bash -n slurm/report_external_round3.slurm
bash -n slurm/final_candidate_full_ood.slurm
/home/bolin/envs/cantonese-asr-whisper/bin/python -m pytest \
  tests/test_round3_slurm.py -q
```

Expected: both shell checks exit 0 and pytest reports `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add \
  slurm/report_external_round3.slurm \
  slurm/final_candidate_full_ood.slurm \
  tests/test_round3_slurm.py
git commit -m "feat: report and select round 3 candidates"
```

### Task 6: Validate, deploy and launch

**Files:**
- Verify: all files listed above

- [ ] **Step 1: Run local syntax checks**

```bash
python3 -m compileall -q scripts tests
bash -n slurm/prepare_external_round3.slurm
bash -n slurm/external_round3.slurm
bash -n slurm/report_external_round3.slurm
bash -n slurm/final_candidate_full_ood.slurm
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 2: Synchronize code**

```bash
./scripts/sync_server.sh
```

Expected: code arrives under `/home/bolin/cantonese-asr` while data, models, outputs and logs remain server-only.

- [ ] **Step 3: Run the full server test suite**

```bash
ssh pavb \
  'cd /home/bolin/cantonese-asr && /home/bolin/envs/cantonese-asr-whisper/bin/python -m pytest -q'
```

Expected: every test passes.

- [ ] **Step 4: Read authoritative external train counts**

```bash
ssh pavb 'cd /home/bolin/cantonese-asr && python - <<'"'"'PY'"'"'
import json
from pathlib import Path

report = json.loads(Path(
    "artifacts/manifests/external/v1/data_report.json"
).read_text(encoding="utf-8"))
print(json.dumps(report, ensure_ascii=False, indent=2))
PY'
```

Expected: accepted Common Voice and MDCC train counts are present; any difference from 8,451/64,779 is reported before training.

- [ ] **Step 5: Check capacity and submit Manifest preparation**

```bash
ssh pavb \
  'cd /home/bolin/cantonese-asr && df -h . && squeue -p gpu && sbatch slurm/prepare_external_round3.slurm'
```

Expected: sufficient disk and a new Slurm job ID.

- [ ] **Step 6: Inspect the completed mix report**

```bash
ssh pavb \
  'cd /home/bolin/cantonese-asr && cat artifacts/manifests/external/round3/mix_report.json'
```

Expected: four target Manifests have unique IDs, strict nested fractions, actual source counts and SHA-256 values.

- [ ] **Step 7: Launch four independent trials**

```bash
ssh pavb \
  'cd /home/bolin/cantonese-asr && sbatch slurm/external_round3.slurm'
```

Expected: one array job with tasks `0-3`, each requesting one RTX 4090.

- [ ] **Step 8: Monitor every epoch**

```bash
ssh pavb \
  'cd /home/bolin/cantonese-asr && squeue -u "$USER" && tail -n 80 logs/canto-ext-r3-*.out'
```

Expected: each task records train/validation loss, validation accuracy/CER, current LR, gradient norm, throughput, runtime and GPU memory without OOM or NaN/Inf.

- [ ] **Step 9: Generate Round 3 reports**

```bash
ssh pavb \
  'cd /home/bolin/cantonese-asr && sbatch slurm/report_external_round3.slurm'
```

Expected: `artifacts/reports/experiments/round3/report.html` plus PNG/CSV/JSON artifacts.

### Task 7: Select and deliver the new candidate

**Files:**
- Reuse: `slurm/final_candidate_full_ood.slurm`
- Reuse: `scripts/package_submission.py`
- Reuse: `scripts/verify_submission.py`

- [ ] **Step 1: Compare guarded validation winners**

Run:

```bash
ssh pavb \
  'cd /home/bolin/cantonese-asr && for f in outputs/external-round3/*/checkpoint_selection.json; do echo "$f"; cat "$f"; done'
```

Expected: only eligible checkpoints with accuracy at least 0.8219 and CER at most 0.1163 are selected.

- [ ] **Step 2: Run complete OOD and package the validation winner**

```bash
ssh pavb \
  'cd /home/bolin/cantonese-asr && sbatch slurm/final_candidate_full_ood.slurm'
```

Expected: full 27,177-row OOD metrics, public/formal protocol checks, one flat `submission.zip` and successful 32-sample offline verification.

- [ ] **Step 3: Copy the verified package to a temporary Mac upload directory**

```bash
upload_dir=$(mktemp -d /tmp/cantonese-asr-round3-upload.XXXXXX)
rsync -avP \
  pavb:/home/bolin/cantonese-asr/outputs/final-candidate/submission.zip \
  "$upload_dir/submission.zip"
shasum -a 256 "$upload_dir/submission.zip"
```

Expected: Mac SHA-256 matches the server package report.

- [ ] **Step 4: Hand off manual platform upload**

Report the exact temporary path, size, SHA-256, chosen trial, epoch, fixed validation metrics and full OOD metrics. The user uploads manually. After the user confirms platform receipt, move the temporary directory to `/Users/zhangxun/.Trash/` so the server remains the canonical package location.
