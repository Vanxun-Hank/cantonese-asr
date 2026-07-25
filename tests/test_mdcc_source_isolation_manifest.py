from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from cantonese_asr.io import read_jsonl, sha256_file, write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "build_mdcc_source_isolation_manifest.py"


def row(row_id: str, source: str, audio_hash: str) -> dict[str, object]:
    return {
        "id": row_id,
        "audio_path": f"audio/{row_id}.wav",
        "audio_sha256": audio_hash,
        "text": f"粤语 {row_id}",
        "duration_s": 1.0,
        "source": source,
        "split": "train",
    }


def run_builder(
    tmp_path: Path,
    *,
    mdcc_sample_count: int = 6,
    excluded_rows: list[dict[str, object]] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path, Path, Path]:
    official_path = tmp_path / "official.jsonl"
    mdcc_path = tmp_path / "mdcc.jsonl"
    round2_path = tmp_path / "round2.jsonl"
    excluded_path = tmp_path / "excluded.jsonl"
    output_dir = tmp_path / "output"

    official = [row(f"official:{index}", "official", f"off-{index}") for index in range(4)]
    mdcc = [row(f"mdcc:{index}", "mdcc", f"mdcc-{index}") for index in range(10)]
    round2 = official + mdcc[:3] + [
        row("cv:0", "common_voice_26_zh_HK", "cv-0")
    ]
    write_jsonl(official_path, official)
    write_jsonl(mdcc_path, mdcc)
    write_jsonl(round2_path, round2)
    write_jsonl(excluded_path, excluded_rows or [])

    command = [
        sys.executable,
        str(SCRIPT),
        "--official-train",
        str(official_path),
        "--mdcc-train",
        str(mdcc_path),
        "--round2-manifest",
        str(round2_path),
        "--excluded-manifest",
        str(excluded_path),
        "--output-dir",
        str(output_dir),
        "--seed",
        "42",
        "--expected-official-rows",
        "4",
        "--expected-mdcc-pool-rows",
        "10",
        "--mdcc-sample-count",
        str(mdcc_sample_count),
        "--expected-official-sha256",
        sha256_file(official_path),
        "--expected-mdcc-sha256",
        sha256_file(mdcc_path),
        "--expected-round2-sha256",
        sha256_file(round2_path),
    ]
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, output_dir, official_path, mdcc_path, round2_path


def test_builds_unique_deterministic_official_mdcc_mix(tmp_path: Path) -> None:
    result, output_dir, *_ = run_builder(tmp_path)

    assert result.returncode == 0, result.stderr
    rows = read_jsonl(output_dir / "train.jsonl")
    smoke = read_jsonl(output_dir / "smoke32.jsonl")
    report = json.loads(
        (output_dir / "data_report.json").read_text(encoding="utf-8")
    )
    assert len(rows) == 10
    assert len({str(item["id"]) for item in rows}) == 10
    assert Counter(str(item["source"]) for item in rows) == {
        "official": 4,
        "mdcc": 6,
    }
    assert not any("common_voice" in str(item["source"]) for item in rows)
    selected_mdcc = [item for item in rows if item["source"] == "mdcc"]
    assert sorted(int(item["sampling_rank"]) for item in selected_mdcc) == list(
        range(6)
    )
    assert all(item["sampling_seed"] == 42 for item in selected_mdcc)
    assert len(smoke) == 10
    assert report["passed"] is True
    assert report["inputs"]["mdcc_pool"]["rows"] == 10
    assert report["output"]["rows"] == 10
    assert report["output"]["unique_ids"] == 10
    assert report["output"]["sources"] == {"mdcc": 6, "official": 4}
    assert report["sampling"]["without_replacement"] is True
    assert report["sampling"]["mdcc_round2_intersection"] <= 3

    first_manifest = (output_dir / "train.jsonl").read_bytes()
    first_report = (output_dir / "data_report.json").read_bytes()
    result, output_dir, *_ = run_builder(tmp_path)
    assert result.returncode == 0, result.stderr
    assert (output_dir / "train.jsonl").read_bytes() == first_manifest
    assert (output_dir / "data_report.json").read_bytes() == first_report


def test_rejects_protected_audio_overlap(tmp_path: Path) -> None:
    excluded = [
        {
            **row("protected-copy", "mdcc", "mdcc-0"),
            "split": "validation",
        }
    ]
    result, *_ = run_builder(
        tmp_path,
        mdcc_sample_count=10,
        excluded_rows=excluded,
    )

    assert result.returncode != 0
    assert "protected overlap" in result.stderr


def test_rejects_mdcc_pool_with_duplicate_ids(tmp_path: Path) -> None:
    result, output_dir, official_path, mdcc_path, round2_path = run_builder(tmp_path)
    assert result.returncode == 0, result.stderr
    mdcc = read_jsonl(mdcc_path)
    write_jsonl(mdcc_path, mdcc + [dict(mdcc[0])])
    command = [
        sys.executable,
        str(SCRIPT),
        "--official-train",
        str(official_path),
        "--mdcc-train",
        str(mdcc_path),
        "--round2-manifest",
        str(round2_path),
        "--output-dir",
        str(output_dir),
        "--expected-official-rows",
        "4",
        "--expected-mdcc-pool-rows",
        "11",
        "--mdcc-sample-count",
        "6",
    ]
    duplicate_result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert duplicate_result.returncode != 0
    assert "duplicate ids" in duplicate_result.stderr
