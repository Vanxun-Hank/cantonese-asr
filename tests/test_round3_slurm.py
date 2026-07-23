from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROUND3_SLURM = PROJECT_ROOT / "slurm" / "external_round3.slurm"


def test_round3_array_runs_fixed_mdcc_coverage_trials() -> None:
    script = ROUND3_SLURM.read_text(encoding="utf-8")

    assert "#SBATCH --array=0-3%4" in script
    for trial, manifest in (
        ("external73", "external73.jsonl"),
        ("external80", "external80.jsonl"),
        ("external85", "external85.jsonl"),
        ("all_train", "all_train.jsonl"),
    ):
        assert trial in script
        assert manifest in script

    required_training_tokens = (
        "--model artifacts/models/whisper-small",
        "--learning-rate 2e-5",
        "--scheduler cosine",
        "--epochs 3",
        "--batch-size 8",
        "--eval-batch-size 4",
        "--gradient-accumulation-steps 2",
        "--warmup-ratio 0.05",
        "--weight-decay 0.01",
        "--seed 42",
        "--bf16",
        "--tf32",
        "--generation-max-length 225",
        "--validation-manifest artifacts/manifests/validation.jsonl",
        "--train-probe-manifest artifacts/manifests/train_probe.jsonl",
        "--ood-panel-manifest artifacts/manifests/ood/v1/ood_panel.jsonl",
    )
    for token in required_training_tokens:
        assert token in script

    assert "Full SFT" in script
    assert "AdamW" in script
    assert "eval/save once per epoch" in script
    assert "LoRA" not in script
    assert "scripts/evaluate_checkpoints.py" in script
    assert "scripts/select_best_checkpoint.py" in script
    assert "--min-sentence-accuracy 0.8219" in script
    assert "--max-cer 0.1163" in script
    assert "scripts/plot_experiments.py" in script
    assert "outputs/external-round3/$TRIAL" in script
    assert "checkpoint_selection.json" in script
    assert "No checkpoint passed" in script
