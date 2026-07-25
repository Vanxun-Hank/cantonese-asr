from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "slurm" / "round8_source_ablation.slurm"


def test_round8_source_ablation_runs_three_parallel_data_only_controls() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    for token in (
        "#SBATCH --array=0-2%3",
        "official_mdcc57.jsonl",
        "official_cv57.jsonl",
        "external/round2/external73.jsonl",
        "official-mdcc57-fresh",
        "official-cv57-fresh",
        "round2-exact-fresh",
        "EXPECTED_ROWS=(14743 14743 23304)",
        "--model artifacts/models/whisper-small",
        "--learning-rate 2e-5",
        "--scheduler cosine",
        "--epochs 3",
        "--batch-size 8",
        "--gradient-accumulation-steps 2",
        "--weight-decay 0.01",
        "--no-apply-spec-augment",
        "--generation-max-length 225",
        "--seed 42",
        "--min-sentence-accuracy 0.8219",
        "--max-cer 0.1163",
    ):
        assert token in script
    for forbidden in (
        "--resume-from-checkpoint",
        "--apply-spec-augment",
        "speed-v1",
        "LoRA",
    ):
        assert forbidden not in script
