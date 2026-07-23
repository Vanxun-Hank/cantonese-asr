from __future__ import annotations

import json
from pathlib import Path

import pytest

from cantonese_asr.io import read_test_rows, resolve_audio_path, write_jsonl


def test_read_test_rows_preserves_jsonl_order(tmp_path: Path) -> None:
    path = tmp_path / "list.jsonl"
    write_jsonl(
        path,
        [
            {"audio_path": "b.wav", "ref_text": "乙"},
            {"audio_path": "a.wav", "ref_text": "甲"},
        ],
    )
    assert [row["audio_path"] for row in read_test_rows(path)] == ["b.wav", "a.wav"]


def test_read_test_rows_accepts_official_csv_alias(tmp_path: Path) -> None:
    path = tmp_path / "list.csv"
    path.write_text("audio:file,text:label\ntest_audio/a.wav,甲\n", encoding="utf-8")
    assert read_test_rows(path)[0]["audio_path"] == "test_audio/a.wav"


def test_resolve_audio_supports_root_and_basename(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio = audio_dir / "a.wav"
    audio.write_bytes(b"RIFF")
    assert resolve_audio_path(audio_dir, "nested/a.wav") == audio


def test_missing_audio_path_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"audio_path": ""}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing audio_path"):
        read_test_rows(path)

