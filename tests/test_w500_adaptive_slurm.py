from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "slurm/w500_adaptive_train_stage.slurm"
EVALUATE = ROOT / "slurm/w500_adaptive_evaluate_stage.slurm"
SELECT = ROOT / "slurm/w500_adaptive_select_stage.slurm"
RUNBOOK = ROOT / "docs/RUNBOOK.md"


def test_training_keeps_four_independent_single_gpu_arms_on_gpu001() -> None:
    text = TRAIN.read_text(encoding="utf-8")
    assert "#SBATCH --nodelist=gpu001" in text
    assert "#SBATCH --gres=gpu:4" in text
    assert "for gpu in 0 1 2 3" in text
    assert "export CUDA_VISIBLE_DEVICES=$gpu" in text
    assert '--arm-index "$gpu"' in text
    assert "torchrun" not in text
    assert "srun" not in text


def test_live_evaluator_uses_gpu002_and_matches_one_arm_per_gpu() -> None:
    text = EVALUATE.read_text(encoding="utf-8")
    assert "#SBATCH --nodelist=gpu002" in text
    assert "#SBATCH --gres=gpu:4" in text
    assert "ADAPTIVE_ARM_FILE is required for live evaluation" in text
    assert "for gpu in 0 1 2 3" in text
    assert "arm=${ARM_NAMES[$gpu]}" in text
    assert "export CUDA_VISIBLE_DEVICES=$gpu" in text
    assert 'wait_for_training_artifact "$run_config"' in text
    assert 'wait_for_training_artifact "$checkpoint/sha256sums.json"' in text
    assert "after_complete_wenet_official_cycle" in text
    assert "Atomic checkpoint receipt failed" in text
    assert "expected 16 checkpoint tasks" in text
    assert "gpu002_live_checkpoint_pipeline" in text


def test_stage_selection_requires_training_and_evaluation_receipts() -> None:
    text = SELECT.read_text(encoding="utf-8")
    assert "training_completion.json" in text
    assert "evaluation_completion.json" in text
    assert "for role,path in zip(('training','evaluation')" in text
    assert "if not value.get('passed')" in text
    assert "fixed_validation_only" in text
    assert "public_ood_used_for_ranking" in text


def test_runbook_launches_live_eval_after_train_starts_and_joins_both() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    for stage in (40, 45, 50):
        assert f"E{stage}=$(sbatch --parsable --dependency=after:$T{stage}" in text
        assert (
            f"S{stage}=$(sbatch --parsable --dependency=afterok:$T{stage}:$E{stage}"
            in text
        )
        assert (
            f"ADAPTIVE_STAGE={stage},ADAPTIVE_ARM_FILE=artifacts/"
            f"w500_adaptive_continuation_v1/stage{stage}_arms.json"
        ) in text
    assert "gpu001" in text and "gpu002" in text


def test_all_adaptive_slurm_entrypoints_have_valid_bash_syntax() -> None:
    paths = sorted((ROOT / "slurm").glob("w500_adaptive_*.slurm"))
    assert paths
    for path in paths:
        completed = subprocess.run(
            ["bash", "-n", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, f"{path}: {completed.stderr}"
