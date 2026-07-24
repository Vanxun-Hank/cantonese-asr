from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from cantonese_asr.io import read_jsonl, write_jsonl
from scripts.augment_speed_manifest import BuildConfig, build


def write_tone(path: Path, sample_rate: int, frequency: float) -> None:
    duration_s = 0.25
    points = np.arange(int(sample_rate * duration_s), dtype=np.float32)
    waveform = 0.2 * np.sin(2 * math.pi * frequency * points / sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform, sample_rate)


def manifest_row(
    sample_id: str,
    audio_path: str,
    text: str,
    *,
    split: str = "train",
) -> dict[str, object]:
    return {
        "id": sample_id,
        "audio_path": audio_path,
        "text": text,
        "mandarin": text,
        "jyutping": "example",
        "scene": "1问候场景",
        "source": "official",
        "split": split,
    }


def build_config(tmp_path: Path, expected_parents: int = 2) -> BuildConfig:
    return BuildConfig(
        train_manifest=tmp_path / "manifests" / "train.jsonl",
        validation_manifest=tmp_path / "manifests" / "validation.jsonl",
        public_manifest=tmp_path / "manifests" / "public.jsonl",
        output_audio_root=tmp_path / "artifacts" / "data" / "augmented" / "official-speed-v1",
        output_manifest_dir=tmp_path / "artifacts" / "manifests" / "augmentation" / "official-speed-v1",
        project_root=tmp_path,
        expected_parents=expected_parents,
        workers=2,
        version_tag="official-speed-v1",
    )


def prepare_manifests(tmp_path: Path) -> BuildConfig:
    write_tone(tmp_path / "audio" / "one.wav", 22_050, 220.0)
    write_tone(tmp_path / "audio" / "two.wav", 44_100, 330.0)
    config = build_config(tmp_path)
    write_jsonl(
        config.train_manifest,
        [
            manifest_row("1", "audio/one.wav", "你好！"),
            manifest_row("2", "audio/two.wav", "我今日飲茶。"),
        ],
    )
    # The first validation text reproduces the existing punctuation-only
    # baseline overlap; augmentation must preserve, not increase, it.
    write_jsonl(
        config.validation_manifest,
        [
            manifest_row("100", "validation/one.wav", "你好", split="validation"),
        ],
    )
    write_jsonl(
        config.public_manifest,
        [
            manifest_row("101", "public/one.wav", "完全不同", split="public_excluded"),
        ],
    )
    return config


def test_builds_three_verified_views_and_preserves_labels(tmp_path: Path) -> None:
    config = prepare_manifests(tmp_path)

    report = build(config)

    rows = read_jsonl(config.output_manifest_dir / "all.jsonl")
    assert len(rows) == 6
    assert report["counts"] == {
        "parents": 2,
        "rows": 6,
        "0.9": 2,
        "1.0": 2,
        "1.1": 2,
    }
    assert report["checks"]["normalized_text_overlap"]["baseline"]["validation"] == ["你好"]
    assert report["checks"]["normalized_text_overlap"]["augmented"]["validation"] == ["你好"]

    by_parent: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_parent.setdefault(str(row["parent_id"]), []).append(row)
    assert set(by_parent) == {"1", "2"}
    for parent_rows in by_parent.values():
        assert {row["speed_factor"] for row in parent_rows} == {0.9, 1.0, 1.1}
        assert len({row["text"] for row in parent_rows}) == 1
        assert {row["mandarin"] for row in parent_rows}
        assert {row["jyutping"] for row in parent_rows}
        for row in parent_rows:
            if row["speed_factor"] != 1.0:
                audio_path = tmp_path / str(row["audio_path"])
                info = sf.info(str(audio_path))
                assert info.samplerate == 16_000
                assert info.channels == 1

    receipt = json.loads((config.output_manifest_dir / "receipt.json").read_text())
    assert receipt["rows"] == 6
    assert read_jsonl(config.output_manifest_dir / "quarantine.jsonl") == []

    verification = config.output_manifest_dir / "verification.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/verify_speed_augmentation.py",
            "--manifest",
            str(config.output_manifest_dir / "all.jsonl"),
            "--report",
            str(config.output_manifest_dir / "data_report.json"),
            "--quarantine",
            str(config.output_manifest_dir / "quarantine.jsonl"),
            "--project-root",
            str(tmp_path),
            "--expected-parents",
            "2",
            "--output",
            str(verification),
        ],
        cwd=Path(__file__).parents[1],
        check=True,
    )
    assert json.loads(verification.read_text())["passed"] is True


def test_refuses_to_overwrite_existing_augmentation_version(tmp_path: Path) -> None:
    config = prepare_manifests(tmp_path)
    build(config)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build(config)


def test_failed_parent_does_not_publish_final_manifest(tmp_path: Path) -> None:
    config = build_config(tmp_path, expected_parents=1)
    write_jsonl(
        config.train_manifest,
        [manifest_row("1", "audio/missing.wav", "你好！")],
    )
    write_jsonl(
        config.validation_manifest,
        [manifest_row("100", "validation/one.wav", "驗證", split="validation")],
    )
    write_jsonl(
        config.public_manifest,
        [manifest_row("101", "public/one.wav", "公開", split="public_excluded")],
    )

    with pytest.raises(RuntimeError, match="failed augmentation"):
        build(config)

    assert not config.output_audio_root.exists()
    assert not config.output_manifest_dir.exists()
