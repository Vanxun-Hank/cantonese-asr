from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREP = PROJECT_ROOT / "slurm" / "prepare_round8_mdcc_source_isolation.slurm"
SMOKE = PROJECT_ROOT / "slurm" / "round8_mdcc_source_isolation_smoke.slurm"
TRAIN = PROJECT_ROOT / "slurm" / "round8_mdcc_source_isolation.slurm"


def test_round8_preparation_locks_source_counts_hashes_and_exclusions() -> None:
    script = PREP.read_text(encoding="utf-8")
    for token in (
        "scripts/build_mdcc_source_isolation_manifest.py",
        "artifacts/manifests/train.jsonl",
        "artifacts/manifests/external/v1/mdcc_train.jsonl",
        "artifacts/manifests/external/round2/external73.jsonl",
        "artifacts/manifests/validation.jsonl",
        "artifacts/manifests/public_excluded.jsonl",
        "artifacts/manifests/ood/v1/ood_all.jsonl",
        "--expected-official-rows 6292",
        "--expected-mdcc-pool-rows 64779",
        "--mdcc-sample-count 17012",
        "0d3d1df4b90f89907443ad9706f13d9ff0ae5fe66d6b32c939af99645867cb81",
        "fc2c33f6d10f2603de3aa60f474f94889e0d140e35f5dfc29ba78fedee3013bc",
        "2ba7edef6811b0d3e96b3347aa70f4237ec27735652d3d313bce62525f1cdea6",
        "--verify-audio-exists",
        '"mdcc": 17_012',
        '"official": 6_292',
        "ood_cv.jsonl",
        "ood_mdcc.jsonl",
    ):
        assert token in script


def test_round8_smoke_is_fresh_full_sft_without_specaugment() -> None:
    script = SMOKE.read_text(encoding="utf-8")
    for token in (
        "#SBATCH --gres=gpu:1",
        "official-mdcc73/smoke32.jsonl",
        "--model artifacts/models/whisper-small",
        "--freeze-encoder-layers 0",
        "--no-apply-spec-augment",
        "--generation-max-length 225",
        "scripts/evaluate_checkpoints.py",
        'metrics["sentence_accuracy_tol2"] < 0.75',
    ):
        assert token in script
    assert "--resume-from-checkpoint" not in script


def test_round8_formal_job_changes_only_training_source_mix() -> None:
    script = TRAIN.read_text(encoding="utf-8")
    for token in (
        "#SBATCH --gres=gpu:1",
        "--model artifacts/models/whisper-small",
        "--train-manifest \"$MANIFEST\"",
        "--validation-manifest artifacts/manifests/validation.jsonl",
        "--train-probe-manifest artifacts/manifests/train_probe.jsonl",
        "--freeze-encoder-layers 0",
        "--learning-rate 2e-5",
        "--scheduler cosine",
        "--epochs 3",
        "--batch-size 8",
        "--gradient-accumulation-steps 2",
        "--warmup-ratio 0.05",
        "--weight-decay 0.01",
        "--save-total-limit 3",
        "--generation-max-length 225",
        "--no-apply-spec-augment",
        "--seed 42",
        "--bf16",
        "--tf32",
        "scripts/evaluate_checkpoints.py",
        "--diagnostic-manifest ood_cv=",
        "--diagnostic-manifest ood_mdcc=",
        "scripts/select_best_checkpoint.py",
        "--min-sentence-accuracy 0.8219",
        "--max-cer 0.1163",
    ):
        assert token in script
    for forbidden in (
        "common_voice_train.jsonl",
        "--apply-spec-augment",
        "--resume-from-checkpoint",
        "speed-v1",
        "package_submission.py",
    ):
        assert forbidden not in script
