from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cantonese_asr.w500_adaptive import (
    assert_exact_prefix,
    branch_top_two,
    bucket_key,
    guardrail_failures,
    middle_loss_rows,
    nearest_aligned_boundary,
    rank_rows,
    validation_proxy,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/rounds/w500_adaptive_continuation.json"


def _eligible_row(label: str, *, cer: float = 0.09, tol2: float = 0.84) -> dict:
    return {
        "label": label,
        "hours": 40.0,
        "validation": {"cer": cer, "tol2": tol2, "severe": 17},
        "public": {
            "tol2": 0.88,
            "insertions": 100,
            "severe": 20,
            "repeated_runaway": 0,
            "replacement": 0,
            "effective_max": 0,
        },
        "ood": {"tol2": 0.33, "cer": 0.342},
        "complete": True,
    }


def test_registry_has_approved_stage40_matrix() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["start"]["weight_sha256"] == (
        "9cbc56266b7c263f93475230fa3e3c350841835f4049261c847c0bc36b04b9d6"
    )
    assert cfg["start"]["processor_checkpoint"].endswith(
        "W500_EONLY_R1TO1_ELR1E6"
    )
    assert "preprocessor_config.json" in cfg["start"]["processor_files_sha256"]
    assert [arm["name"] for arm in cfg["stage40"]["arms"]] == [
        "OLDSTREAM_HALF_LR",
        "BALANCED_ORIG_LR",
        "BALANCED_HALF_LR",
        "BALANCED_QUARTER_LR",
    ]
    assert cfg["training"]["source_cycle"] == ["wenet", "official"]
    assert cfg["training"]["apply_spec_augment"] is False
    assert cfg["selection"]["ranking_surface"] == "fixed_validation_only"


def test_proxy_weights_cer_more_than_tol2() -> None:
    assert validation_proxy(cer=0.09, tol2=0.84) == 80.5
    assert validation_proxy(cer=0.08, tol2=0.84) > validation_proxy(
        cer=0.09, tol2=0.87
    )


def test_public_can_veto_but_cannot_rerank() -> None:
    good = _eligible_row("good")
    cosmetically_better_public = copy.deepcopy(good)
    cosmetically_better_public["label"] = "same-validation"
    cosmetically_better_public["public"]["tol2"] = 0.95
    assert rank_rows([good, cosmetically_better_public])[0]["label"] == "good"
    broken = copy.deepcopy(good)
    broken["public"]["repeated_runaway"] = 1
    assert "public_repeated_runaway" in guardrail_failures(broken)
    assert rank_rows([broken, good]) == [good]


def test_ranking_uses_cer_when_proxy_difference_is_under_tie_band() -> None:
    lower_cer = _eligible_row("lower-cer", cer=0.0899, tol2=0.84)
    higher_proxy = _eligible_row("higher-proxy", cer=0.09, tol2=0.841)
    assert validation_proxy(cer=0.09, tol2=0.841) > validation_proxy(
        cer=0.0899, tol2=0.84
    )
    assert rank_rows([higher_proxy, lower_cer])[0]["label"] == "lower-cer"


def test_middle_loss_rows_excludes_extremes_then_preserves_order() -> None:
    rows = [{"id": str(index), "normalized_loss": float(index)} for index in range(100)]
    selected = middle_loss_rows(rows, low=0.10, high=0.90)
    assert selected[0]["id"] == "10"
    assert selected[-1]["id"] == "89"
    assert len(selected) == 80
    with pytest.raises(ValueError, match="finite"):
        middle_loss_rows(
            [{"id": "bad", "normalized_loss": float("nan")}], low=0.1, high=0.9
        )


def test_bucket_key_is_stable_with_missing_metadata() -> None:
    row = {"duration": 7.5, "text": "廣東話測試", "speaker_id": "s1"}
    assert bucket_key(row) == (
        "dur_5_8",
        "txt_1_11",
        "s1",
        "unknown",
        "unknown",
    )


def test_aligned_boundary_and_exact_prefix() -> None:
    rows = [{"id": str(index), "duration": 60.0} for index in range(400)]
    boundary = nearest_aligned_boundary(rows, target_seconds=5 * 3600, alignment=16)
    assert boundary % 16 == 0
    assert_exact_prefix(rows[:boundary], rows[: boundary + 16])
    with pytest.raises(ValueError, match="mismatch"):
        assert_exact_prefix(rows[:boundary], list(reversed(rows[: boundary + 16])))


def test_branch_top_two_creates_constant_and_half_lr_per_parent() -> None:
    parents = [
        {
            "label": "p1",
            "checkpoint": "/p1",
            "processor_checkpoint": "/processor-p1",
            "hours": 40.0,
            "wenet_lr": 5e-7,
            "official_lr": 2.5e-7,
            "wenet_cursor": 1200,
            "official_cursor": 8912,
        },
        {
            "label": "p2",
            "checkpoint": "/p2",
            "processor_checkpoint": "/processor-p2",
            "hours": 40.0,
            "wenet_lr": 2.5e-7,
            "official_lr": 1.25e-7,
            "wenet_cursor": 1200,
            "official_cursor": 8912,
        },
    ]
    arms = branch_top_two(parents, target_hours=45.0, manifest="/balanced45.jsonl")
    assert len(arms) == 4
    assert [arm["lr_mode"] for arm in arms] == [
        "constant",
        "half",
        "constant",
        "half",
    ]
    assert arms[1]["wenet_lr"] == 2.5e-7
    assert arms[3]["official_lr"] == 6.25e-8
    assert {arm["wenet_continuation_cursor"] for arm in arms} == {1200}
    assert [arm["processor_checkpoint"] for arm in arms] == [
        "/processor-p1",
        "/processor-p1",
        "/processor-p2",
        "/processor-p2",
    ]
