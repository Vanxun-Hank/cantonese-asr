from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import wave
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from cantonese_asr.io import read_jsonl


def wav_bytes(duration: float = 0.1, sample_rate: int = 16_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * int(duration * sample_rate))
    return buffer.getvalue()


def test_external_manifest_materialization_and_leakage(tmp_path: Path) -> None:
    project_root = Path(__file__).parents[1]
    cv = tmp_path / "cv"
    clips = cv / "clips"
    clips.mkdir(parents=True)
    rows = [
        {"client_id": "a", "path": "protected.mp3", "sentence": "你好"},
        {"client_id": "b", "path": "valid.mp3", "sentence": "我今日飲茶。"},
    ]
    for row in rows:
        (clips / row["path"]).write_bytes(wav_bytes())
    with (cv / "train.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["client_id", "path", "sentence"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    with (cv / "clip_durations.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["clip", "duration[ms]"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(
            [
                {"clip": "protected.mp3", "duration[ms]": 100},
                {"clip": "valid.mp3", "duration[ms]": 100},
            ]
        )

    mdcc = tmp_path / "mdcc"
    mdcc.mkdir()
    audio = wav_bytes(duration=0.2)
    table = pa.Table.from_pylist(
        [
            {
                "id": 1,
                "sex": "female",
                "duration": 0.2,
                "transcript": "今日天氣幾好",
                "audio": {"bytes": audio, "path": "one.wav"},
            },
            {
                "id": 2,
                "sex": "male",
                "duration": 0.2,
                "transcript": "另一句",
                "audio": {"bytes": audio, "path": "duplicate.wav"},
            },
        ]
    )
    pq.write_table(table, mdcc / "train-00000-of-00001.parquet")
    protected = tmp_path / "protected.jsonl"
    protected.write_text(
        json.dumps({"text": "你好！"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "manifests"
    audio_root = tmp_path / "materialized"
    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_external_manifest.py",
            "--common-voice-root",
            str(cv),
            "--mdcc-parquet-root",
            str(mdcc),
            "--mdcc-audio-root",
            str(audio_root),
            "--output-dir",
            str(output),
            "--project-root",
            str(tmp_path),
            "--protected-text-jsonl",
            str(protected),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    cv_rows = read_jsonl(output / "common_voice_train.jsonl")
    mdcc_rows = read_jsonl(output / "mdcc_train.jsonl")
    quarantine = read_jsonl(output / "quarantine.jsonl")
    report = json.loads((output / "data_report.json").read_text(encoding="utf-8"))

    assert [row["text"] for row in cv_rows] == ["我今日飲茶。"]
    assert len(mdcc_rows) == 1
    assert (tmp_path / mdcc_rows[0]["audio_path"]).is_file()
    assert {row["reason"] for row in quarantine} == {
        "protected_text_overlap",
        "duplicate_audio_sha256",
    }
    assert report["leakage_checks"] == {
        "protected_text_overlap": 0,
        "duplicate_audio_sha256": 0,
    }
