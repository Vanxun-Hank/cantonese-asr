from __future__ import annotations

import csv
import json
import subprocess
import sys
import wave
from pathlib import Path

from cantonese_asr.io import read_jsonl
from scripts.prepare_manifest import extract_audio_id


def test_audio_id_uses_fixed_five_digit_prefix() -> None:
    assert extract_audio_id(Path("1003211号嘅都得.wav")) == "10032"
    assert extract_audio_id(Path("0189642码有货.wav")) == "1896"


def write_wav(path: Path, duration: float = 0.1, sample_rate: int = 16_000) -> None:
    frames = int(duration * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)


def test_manifest_join_split_probe_and_quarantine(tmp_path: Path) -> None:
    project_root = Path(__file__).parents[1]
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    macos_metadata = audio_root / "__MACOSX"
    macos_metadata.mkdir()
    (macos_metadata / "._00001.wav").write_bytes(b"not audio")
    index_path = tmp_path / "index.csv"
    rows = []
    for sample_id in range(1, 13):
        text = "重复句" if sample_id in {1, 2} else f"粤语句子{sample_id}"
        rows.append(
            {
                "序号": sample_id,
                "粤语原文": text,
                "普通话翻译": f"普通话{sample_id}",
                "香港语言学会粤拼": f"jyut{sample_id}",
                "场景": f"{1 + sample_id % 3}场景",
            }
        )
        write_wav(audio_root / f"{sample_id:05d}{text}.wav")
    rows.append(
        {
            "序号": 13,
            "粤语原文": "没有音频",
            "普通话翻译": "没有音频",
            "香港语言学会粤拼": "mou5",
            "场景": "1场景",
        }
    )
    write_wav(audio_root / "00099额外.wav")
    write_wav(audio_root / "00014太长.wav", duration=1.1)
    rows.append(
        {
            "序号": 14,
            "粤语原文": "太长",
            "普通话翻译": "太长",
            "香港语言学会粤拼": "taai3",
            "场景": "2场景",
        }
    )
    with index_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    public_test = tmp_path / "template_pre.jsonl"
    public_test.write_text(
        json.dumps({"audio_path": "/source/00012公开测试.wav", "ref_text": "公开"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "manifests"
    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_manifest.py",
            "--index-csv",
            str(index_path),
            "--audio-root",
            str(audio_root),
            "--output-dir",
            str(output_dir),
            "--validation-ratio",
            "0.25",
            "--train-probe-size",
            "4",
            "--max-duration",
            "1.0",
            "--exclude-test-list",
            str(public_test),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    train = read_jsonl(output_dir / "train.jsonl")
    validation = read_jsonl(output_dir / "validation.jsonl")
    probe = read_jsonl(output_dir / "train_probe.jsonl")
    public_excluded = read_jsonl(output_dir / "public_excluded.jsonl")
    quarantine = read_jsonl(output_dir / "quarantine.jsonl")
    report = json.loads((output_dir / "data_report.json").read_text(encoding="utf-8"))

    assert len(train) + len(validation) == 11
    assert len(probe) == 4
    assert [row["id"] for row in public_excluded] == ["12"]
    assert "12" not in {row["id"] for row in train + validation}
    assert {row["audio_path"] for row in probe} <= {row["audio_path"] for row in train}
    assert {row["text"] for row in train}.isdisjoint({row["text"] for row in validation})
    reasons = {row["reason"] for row in quarantine}
    assert {"csv_row_without_audio", "audio_without_csv_row", "over_whisper_window"} <= reasons
    assert "missing_audio_id" not in reasons
    assert report["leakage_checks"]["normalized_text_overlap"] == 0
    assert report["leakage_checks"]["public_excluded_id_overlap"] == 0
    assert report["counts"]["public_excluded"] == 1
    assert sum(report["duration_histogram"].values()) == 11
