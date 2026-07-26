from __future__ import annotations

import hashlib
import json
import math
import wave
from pathlib import Path

import numpy as np

from cantonese_asr.data_quality import (
    AudioFeatures,
    QC_VERSION,
    audit_rows,
    canonical_text,
    mark_near_audio_duplicates,
    parse_language,
    write_audit_outputs,
)
from cantonese_asr.io import read_jsonl, sha256_file


class FakeRunner:
    def transcribe(self, audio_path: Path) -> tuple[str, str, str]:
        name = audio_path.name
        if name == "mandarin.wav":
            return "<|zh|>普通话", "zh", "普通话"
        if name == "mismatch.wav":
            return "<|yue|>完全不同嘅另一段錄音", "yue", "完全不同嘅另一段錄音"
        if name == "valid.wav":
            return "<|yue|>獨特粵語句子可以保留", "yue", "獨特粵語句子可以保留"
        return "<|yue|>今日飲茶好開心", "yue", "今日飲茶好開心"

    def provenance(self) -> dict[str, str]:
        return {"runner": "fake", "language": "auto"}


class FakeTokenCounter:
    def count(self, text: str) -> int:
        return 226 if text == "太長" else len(text)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = 16_000) -> None:
    pcm = np.clip(samples * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def sine(
    seconds: float = 0.2, amplitude: float = 0.2, sample_rate: int = 16_000,
    frequency: float = 220,
) -> np.ndarray:
    steps = np.arange(round(seconds * sample_rate), dtype=np.float32)
    return amplitude * np.sin(2 * math.pi * frequency * steps / sample_rate)


def row(sample_id: str, name: str, text: str, source: str = "mdcc") -> dict[str, str]:
    return {
        "id": sample_id,
        "audio_path": f"audio/{name}",
        "text": text,
        "source": source,
        "scene": "external_mdcc",
    }


def test_normalization_and_language_parser() -> None:
    assert parse_language("<|yue|><|NEUTRAL|>你好") == "yue"
    assert parse_language("no tag") is None
    assert canonical_text("  飲茶， 好！ ") == canonical_text("飲茶好")


def test_audit_rules_outputs_and_determinism(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    audio.mkdir()
    write_wav(audio / "ok.wav", sine())
    (audio / "duplicate.wav").write_bytes((audio / "ok.wav").read_bytes())
    write_wav(audio / "silent.wav", np.zeros(1600, dtype=np.float32))
    write_wav(audio / "clip.wav", sine(amplitude=1.0))
    write_wav(audio / "mandarin.wav", sine())
    write_wav(audio / "mismatch.wav", sine())
    write_wav(audio / "tokens.wav", sine())
    write_wav(audio / "valid.wav", sine(frequency=330))
    records = audit_rows(
        [
            row("b", "duplicate.wav", "今日飲茶好開心"),
            row("a", "ok.wav", "今日飲茶好開心", "common_voice_26_zh_HK"),
            row("c", "silent.wav", "今日飲茶好開心"),
            row("d", "clip.wav", "今日飲茶好開心"),
            row("e", "mandarin.wav", "普通話句子"),
            row("f", "mismatch.wav", "今日食飯然後返屋企休息"),
            row("g", "tokens.wav", "太長"),
            row("h", "valid.wav", "獨特粵語句子可以保留"),
        ],
        tmp_path,
        FakeRunner(),
        FakeTokenCounter(),
    )
    by_id = {item["id"]: item for item in records}
    assert by_id["a"]["qc_status"] == "review"  # exact duplicate text is report-only
    assert "duplicate_audio_sha256_regression" in by_id["b"]["qc_hard_reasons"]
    assert "near_silence" in by_id["c"]["qc_hard_reasons"]
    assert "severe_clipping" in by_id["d"]["qc_hard_reasons"]
    assert "sensevoice_non_yue" in by_id["e"]["qc_hard_reasons"]
    assert "sensevoice_extreme_transcript_mismatch" in by_id["f"]["qc_hard_reasons"]
    assert "whisper_token_limit" in by_id["g"]["qc_hard_reasons"]
    assert by_id["h"]["qc_status"] == "accepted"
    first, second = tmp_path / "one", tmp_path / "two"
    hashes = {"common_voice_manifest": "a" * 64, "mdcc_manifest": "b" * 64}
    first_report = write_audit_outputs(first, records, FakeRunner().provenance(), hashes)
    second_report = write_audit_outputs(second, records, FakeRunner().provenance(), hashes)
    assert first_report == second_report
    assert sha256_file(first / "audit_results.jsonl") == sha256_file(second / "audit_results.jsonl")
    assert (first / "report.html").is_file()
    assert json.loads((first / "data_report.json").read_text(encoding="utf-8"))["qc_version"] == QC_VERSION
    assert all(item["qc_status"] == "accepted" for item in read_jsonl(first / "external_all.jsonl"))


def test_near_audio_duplicate_keeps_lexicographically_first_id() -> None:
    features = AudioFeatures(
        duration_s=1.0, sample_rate=16_000, channels=1, rms_dbfs=-20.0,
        active_rms_dbfs=-20.0, active_ratio=1.0, clipping_ratio=0.0,
        mfcc_mean=(1.0,) * 20, mfcc_std=(1.0,) * 20, fingerprint_bands=(1, 2),
    )
    records = [
        {"id": "z", "qc_status": "accepted", "qc_hard_reasons": [], "_features": features},
        {"id": "a", "qc_status": "accepted", "qc_hard_reasons": [], "_features": features},
    ]
    mark_near_audio_duplicates(records)
    by_id = {item["id"]: item for item in records}
    assert by_id["a"]["near_audio_duplicate_group"] == "near_audio:000001"
    assert by_id["z"]["qc_hard_reasons"] == ["near_duplicate_audio"]


def test_missing_audio_is_quarantined_without_hiding_the_input_row(tmp_path: Path) -> None:
    records = audit_rows(
        [row("missing", "not-there.wav", "今日飲茶好開心")],
        tmp_path,
        FakeRunner(),
        FakeTokenCounter(),
    )
    assert records[0]["qc_status"] == "hard_quarantine"
    assert records[0]["qc_hard_reasons"] == ["missing_audio"]
    assert records[0]["sensevoice"] is None
