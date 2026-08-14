from __future__ import annotations

from scripts.evaluate_raw_winner_decode import (
    ARM_CONFIGS,
    anchor_sequence_score,
    generation_kwargs,
    select_mbr_candidate,
)
from scripts.select_raw_winner_p1_decode import normal_long_sentence_truncations


def test_anchor_score_prefers_matching_returned_text_then_top_proxy() -> None:
    texts = ["甲", "乙", "甲"]
    scores = [-0.1, -0.2, -0.05]
    assert anchor_sequence_score("甲", texts, scores) == -0.05
    assert anchor_sequence_score("丙", texts, scores) == -0.1


def test_normal_long_sentence_truncation_requires_clear_prefix_loss() -> None:
    baseline = [{"pred_text": "一二三四五六七八九十一二三四五六七八九十结尾"}]
    assert normal_long_sentence_truncations(
        baseline, [{"pred_text": "一二三四五六七八九十一二三四五六"}]
    ) == 1
    assert normal_long_sentence_truncations(
        baseline, [{"pred_text": "完全不同但并非前缀截断"}]
    ) == 0


def test_decode_matrix_changes_only_preregistered_axes() -> None:
    baseline = generation_kwargs("D0_CURRENT")
    assert baseline == {
        "language": "zh",
        "task": "transcribe",
        "return_timestamps": False,
        "do_sample": False,
        "num_beams": 2,
        "no_repeat_ngram_size": 4,
        "repetition_penalty": 1.05,
        "length_penalty": 1.0,
        "max_length": 225,
        "early_stopping": True,
    }
    assert generation_kwargs("D1_BEAM1")["num_beams"] == 1
    assert "early_stopping" not in generation_kwargs("D1_BEAM1")
    assert generation_kwargs("D2_BEAM4")["num_beams"] == 4
    assert generation_kwargs("D3_BEAM5")["num_beams"] == 5
    assert generation_kwargs("D4_MAX128")["max_length"] == 128
    assert generation_kwargs("D5_RP100")["repetition_penalty"] == 1.0
    assert generation_kwargs("D6_RP110")["repetition_penalty"] == 1.10
    assert ARM_CONFIGS["D7_MBR_B5"]["num_return_sequences"] == 5
    assert "max_new_tokens" not in baseline


def test_mbr_prefers_character_consensus() -> None:
    texts = ["我今日返工", "我今日返工", "我今日返工呀", "完全唔同", "另一句"]
    winner, risks = select_mbr_candidate(texts, [-1.0, -1.1, -1.2, -0.1, -0.2])
    assert winner == 0
    assert risks[0] == risks[1]
    assert risks[0] < risks[3]


def test_mbr_ties_use_model_score_then_original_rank() -> None:
    winner, _ = select_mbr_candidate(["相同", "相同"], [-2.0, -1.0])
    assert winner == 1
    winner, _ = select_mbr_candidate(["相同", "相同"], [-1.0, -1.0])
    assert winner == 0
