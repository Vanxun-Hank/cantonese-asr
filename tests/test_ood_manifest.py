from __future__ import annotations

import json
from pathlib import Path

from cantonese_asr.io import write_jsonl
from scripts.prepare_ood_manifest import deterministic_panel, load_exclusions


def make_ood_rows() -> list[dict[str, str]]:
    rows = []
    for source, split in (
        ("common_voice_26_zh_HK", "dev"),
        ("common_voice_26_zh_HK", "test"),
        ("mdcc", "validation"),
        ("mdcc", "test"),
    ):
        for index in range(10):
            rows.append(
                {
                    "id": f"{source}:{split}:{index:02d}",
                    "source": source,
                    "publisher_split": split,
                    "duration_s": "1.0",
                    "text": f"sample {source} {split} {index}",
                    "sample_rate": "16000",
                    "channels": "1",
                }
            )
    return rows


def test_ood_panel_is_deterministic_and_capped_per_publisher_split() -> None:
    rows = make_ood_rows()
    first = deterministic_panel(rows, per_split=3, seed=42)
    second = deterministic_panel(list(reversed(rows)), per_split=3, seed=42)
    assert first == second
    assert len(first) == 12
    groups = {}
    for row in first:
        key = (row["source"], row["publisher_split"])
        groups[key] = groups.get(key, 0) + 1
    assert set(groups.values()) == {3}
    assert deterministic_panel(rows, per_split=3, seed=43) != first


def test_exclusion_boundary_uses_text_audio_hash_and_id(tmp_path: Path) -> None:
    manifest = tmp_path / "training.jsonl"
    write_jsonl(
        manifest,
        [
            {
                "id": "official:1",
                "text": "  你好！ ",
                "audio_path": "missing.wav",
                "audio_sha256": "a" * 64,
            }
        ],
    )
    texts, audio, ids, report = load_exclusions([manifest], tmp_path)
    assert texts
    assert audio == {"a" * 64}
    assert ids == {"official:1"}
    assert report == {
        "rows": 1,
        "unique_text_keys": 1,
        "unique_audio_sha256": 1,
        "unique_ids": 1,
    }
