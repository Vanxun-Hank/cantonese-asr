from __future__ import annotations

from pathlib import Path

import pytest

from cantonese_asr.common_voice_audit import (
    canonical_text,
    classify_candidate,
    compare_rows,
    decide_admission,
    profile_split,
    profile_text,
    select_review_sample,
)


def test_text_keys_and_style_profile() -> None:
    profile = profile_text("  你今日去唔去呀？ ")
    assert profile["clean"] == "你今日去唔去呀?"
    assert profile["canonical"] == canonical_text("你今日去唔去呀？")
    assert profile["cantonese_markers"]["唔"] == 1


def test_profile_split_counts_speakers_hours_and_concentration() -> None:
    rows = [
        {"path": "a.mp3", "sentence": "你好", "client_id": "s1", "duration_s": 2.0},
        {"path": "b.mp3", "sentence": "唔該", "client_id": "s2", "duration_s": 3.0},
    ]
    result = profile_split(rows, "train")
    assert result["rows"] == 2
    assert result["speakers"] == 2
    assert result["hours"] == pytest.approx(5 / 3600)
    assert result["top10_speaker_share"] == 1.0


def test_overlap_separates_repeated_sentence_from_audio_duplicate() -> None:
    yue = [
        {
            "id": "y1",
            "text": "你好",
            "canonical": "你好",
            "client_id": "new",
            "audio_sha256": "a",
            "pcm_sha256": "pa",
        }
    ]
    existing = [
        {
            "id": "z1",
            "text": "你好",
            "canonical": "你好",
            "client_id": "old",
            "audio_sha256": "b",
            "pcm_sha256": "pb",
        }
    ]
    result = compare_rows(yue, existing, "zh-HK")
    assert result["canonical_sentence_overlap"]["count"] == 1
    assert result["audio_sha256_overlap"]["count"] == 0
    assert result["same_sentence_new_audio"]["count"] == 1


def test_protected_text_overlap_is_hard_quarantine() -> None:
    decision = classify_candidate(
        {
            "text": "受保护句",
            "canonical": "受保护句",
            "audio_sha256": "new",
            "duration_s": 2.0,
            "audio_readable": True,
            "label_tokens": 5,
        },
        protected_text={"受保护句"},
        protected_audio=set(),
    )
    assert decision["status"] == "hard_quarantine"
    assert "protected_text_overlap" in decision["reasons"]


def _review_rows() -> list[dict[str, object]]:
    return [
        {
            "id": f"yue:{index:03d}",
            "client_id": f"speaker:{index:03d}",
            "duration_s": 1.0 + index / 10,
            "same_sentence_new_audio": index < 40,
            "text_profile": {
                "written_risk_count": index % 5,
                "cantonese_marker_count": index % 7,
            },
        }
        for index in range(180)
    ]


def test_review_sample_is_deterministic() -> None:
    first = select_review_sample(_review_rows(), seed=42)
    second = select_review_sample(list(reversed(_review_rows())), seed=42)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    random_rows = [
        row for row in first if row["review_bucket"] == "random_duration_stratified"
    ]
    assert len({row["client_id"] for row in random_rows}) == len(random_rows)


def test_admission_requires_all_gates() -> None:
    result = decide_admission(
        {
            "retention_rate": 0.90,
            "new_utterances": 1200,
            "new_speakers": 150,
            "new_hours": 4.0,
            "sample_yue_rate": 0.90,
            "protected_text_overlap": 0,
            "protected_audio_overlap": 0,
            "top10_speaker_share": 0.30,
            "incremental_coverage": True,
        }
    )
    assert result["decision"] == "admit"


def test_admission_is_pending_before_acoustic_review() -> None:
    result = decide_admission(
        {
            "retention_rate": 0.90,
            "new_utterances": 1200,
            "new_speakers": 150,
            "new_hours": 4.0,
            "sample_yue_rate": None,
            "protected_text_overlap": 0,
            "protected_audio_overlap": 0,
            "top10_speaker_share": 0.30,
            "incremental_coverage": True,
        }
    )
    assert result["decision"] == "pending_acoustic_review"


def test_slurm_audit_is_cpu_only_and_fixed_dataset() -> None:
    text = Path("slurm/audit_common_voice_yue.slurm").read_text(encoding="utf-8")
    assert "--gres=gpu" not in text
    assert "cmqinjd7x00vynq07pwzo3lmp" in text
    assert "scripts/download_mdc_dataset.py" in text
    assert "scripts/audit_common_voice_yue.py" in text
    assert "train.py" not in text
    assert "set -a" in text


def test_mdc_authorization_failure_is_not_retried() -> None:
    text = Path("scripts/download_mdc_dataset.py").read_text(encoding="utf-8")
    permission_handler = text.index("except PermissionError:")
    retry_handler = text.index(
        "except (OSError, ValueError, requests.RequestException) as exc:"
    )
    assert permission_handler < retry_handler
