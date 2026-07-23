from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from cantonese_asr.io import read_jsonl, write_jsonl


def rows(prefix: str, count: int, source: str) -> list[dict[str, object]]:
    return [
        {
            "id": f"{prefix}-{index}",
            "audio_path": f"audio/{prefix}-{index}.wav",
            "text": f"text {prefix} {index}",
            "duration_s": 1.0,
            "source": source,
        }
        for index in range(count)
    ]


def test_builds_balanced_deterministic_mixes(tmp_path: Path) -> None:
    project_root = Path(__file__).parents[1]
    official_path = tmp_path / "official.jsonl"
    cv_path = tmp_path / "cv.jsonl"
    mdcc_path = tmp_path / "mdcc.jsonl"
    write_jsonl(official_path, rows("o", 12, "official"))
    write_jsonl(cv_path, rows("cv", 20, "common_voice_26_zh_HK"))
    write_jsonl(mdcc_path, rows("md", 30, "mdcc"))
    output = tmp_path / "mixes"
    command = [
        sys.executable,
        "scripts/build_external_training_mixes.py",
        "--official-train",
        str(official_path),
        "--common-voice",
        str(cv_path),
        "--mdcc",
        str(mdcc_path),
        "--output-dir",
        str(output),
        "--seed",
        "42",
    ]
    subprocess.run(command, cwd=project_root, check=True, capture_output=True, text=True)

    control = read_jsonl(output / "control_official.jsonl")
    mix25 = read_jsonl(output / "external25.jsonl")
    mix50 = read_jsonl(output / "external50.jsonl")
    preadapt = read_jsonl(output / "preadapt_external.jsonl")
    report = json.loads((output / "mix_report.json").read_text(encoding="utf-8"))

    assert len(control) == 12
    assert len(mix25) == 16
    assert len(mix50) == 24
    assert len(preadapt) == 24
    assert report["manifests"]["external25"]["sources"] == {
        "common_voice_26_zh_HK": 2,
        "mdcc": 2,
        "official": 12,
    }
    assert report["manifests"]["external50"]["sources"] == {
        "common_voice_26_zh_HK": 6,
        "mdcc": 6,
        "official": 12,
    }
    first = (output / "external50.jsonl").read_bytes()
    subprocess.run(command, cwd=project_root, check=True, capture_output=True, text=True)
    assert (output / "external50.jsonl").read_bytes() == first
