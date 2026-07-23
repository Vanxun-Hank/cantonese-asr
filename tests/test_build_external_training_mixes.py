from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from cantonese_asr.io import read_jsonl, write_jsonl
from scripts.build_external_training_mixes import balanced_counts


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


def test_balancing_uses_remaining_source_after_common_voice_is_exhausted() -> None:
    assert balanced_counts(17_008, [8_451, 64_779]) == [8_451, 8_557]


def test_builds_nested_mdcc_coverage_and_all_train_mix(tmp_path: Path) -> None:
    project_root = Path(__file__).parents[1]
    official_path = tmp_path / "official.jsonl"
    cv_path = tmp_path / "cv.jsonl"
    mdcc_path = tmp_path / "mdcc.jsonl"
    official = rows("o", 10, "official")
    common_voice = rows("cv", 3, "common_voice_26_zh_HK")
    mdcc = rows("md", 30, "mdcc")
    write_jsonl(official_path, official)
    write_jsonl(cv_path, common_voice)
    write_jsonl(mdcc_path, mdcc)
    output = tmp_path / "round3"
    command = [
        sys.executable,
        "scripts/build_external_training_mixes.py",
        "--official-train", str(official_path),
        "--common-voice", str(cv_path),
        "--mdcc", str(mdcc_path),
        "--output-dir", str(output),
        "--seed", "42",
        "--external-fraction", "0.5",
        "--external-fraction", "0.6",
        "--external-fraction", "0.7",
        "--nested-fractions",
        "--include-all-external",
    ]
    subprocess.run(command, cwd=project_root, check=True, capture_output=True, text=True)

    mix50 = read_jsonl(output / "external50.jsonl")
    mix60 = read_jsonl(output / "external60.jsonl")
    mix70 = read_jsonl(output / "external70.jsonl")
    all_train = read_jsonl(output / "all_train.jsonl")
    report = json.loads((output / "mix_report.json").read_text(encoding="utf-8"))

    assert not (output / "external25.jsonl").exists()
    assert [len(mix50), len(mix60), len(mix70), len(all_train)] == [20, 25, 33, 43]
    assert report["sampling_policy"]["nested_external_fractions"] is True
    assert report["manifests"]["all_train"]["unique_ids"] == len(all_train)
    assert report["manifests"]["external50"]["sources"] == {
        "common_voice_26_zh_HK": 3, "mdcc": 7, "official": 10
    }
    assert report["manifests"]["external60"]["sources"] == {
        "common_voice_26_zh_HK": 3, "mdcc": 12, "official": 10
    }
    assert report["manifests"]["external70"]["sources"] == {
        "common_voice_26_zh_HK": 3, "mdcc": 20, "official": 10
    }

    official_ids = {str(row["id"]) for row in official}
    external_ids = [
        {str(row["id"]) for row in mix} - official_ids
        for mix in (mix50, mix60, mix70)
    ]
    assert external_ids[0] < external_ids[1] < external_ids[2]
    assert {str(row["id"]) for row in all_train} == {
        str(row["id"]) for row in official + common_voice + mdcc
    }

    first = {path.name: path.read_bytes() for path in output.glob("*.jsonl")}
    subprocess.run(command, cwd=project_root, check=True, capture_output=True, text=True)
    assert {path.name: path.read_bytes() for path in output.glob("*.jsonl")} == first
