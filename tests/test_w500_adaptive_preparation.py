from __future__ import annotations

import json
from pathlib import Path

import pytest

from cantonese_asr.io import read_jsonl
from scripts import prepare_w500_adaptive_continuation as prepare


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/rounds/w500_adaptive_continuation.json"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _audio(root: Path, name: str) -> str:
    path = root / "audio" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(name.encode("utf-8"))
    return str(path.relative_to(root))


def _fixture(tmp_path: Path) -> tuple[dict, Path]:
    cfg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    cfg["start"]["actual_wenet_hours"] = 35.0
    consumed = [
        {
            "id": f"used-{index}",
            "audio_path": _audio(tmp_path, f"used-{index}.wav"),
            "duration": 60.0,
            "text": f"參考句{index:02d}",
            "speaker_id": "speaker",
            "program_group": "program",
            "source_tar": "source",
        }
        for index in range(32)
    ]
    historical_delta = [
        {
            "id": f"old-{index}",
            "audio_path": _audio(tmp_path, f"old-{index}.wav"),
            "duration": 1125.0,
            "text": f"舊流句子{index:02d}",
            "speaker_id": "speaker",
            "program_group": "program",
            "source_tar": "source",
        }
        for index in range(16)
    ]
    candidates = [
        {
            "id": f"candidate-{index}",
            "audio_path": _audio(tmp_path, f"candidate-{index}.wav"),
            "duration": 900.0,
            "text": f"候選句{index:03d}",
            "speaker_id": "speaker",
            "program_group": "program",
            "source_tar": "source",
            "confidence": 0.5 + index / 1000,
        }
        for index in range(64)
    ]
    # These prove the audit rejects both already-consumed and malformed rows.
    candidates.extend(
        [
            dict(consumed[0]),
            {
                "id": "replacement",
                "audio_path": _audio(tmp_path, "replacement.wav"),
                "duration": 10.0,
                "text": "壞字�",
            },
        ]
    )
    paths = {
        "w500_pool": "data/pool.jsonl",
        "consumed_35h": "data/consumed.jsonl",
        "historical_40h": "data/historical.jsonl",
        "champion_w20": "data/champion.jsonl",
        "official": "data/official.jsonl",
        "validation": "data/validation.jsonl",
        "public": "data/public.jsonl",
        "ood": "data/ood.jsonl",
    }
    cfg["data"].update(paths)
    _write_jsonl(tmp_path / paths["w500_pool"], candidates)
    _write_jsonl(tmp_path / paths["consumed_35h"], consumed)
    _write_jsonl(tmp_path / paths["historical_40h"], consumed + historical_delta)
    for name in ("champion_w20", "official", "validation", "public", "ood"):
        _write_jsonl(tmp_path / paths[name], [])
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    return cfg, config_path


def _fake_audio_check(path: Path, declared_duration: float) -> tuple[str, dict]:
    return f"sha-{path.stem}", {
        "decoded_duration_s": declared_duration,
        "sample_rate": 16000,
        "channels": 1,
        "frames": int(declared_duration * 16000),
    }


def test_audit_then_freeze_writes_atomic_nested_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, _ = _fixture(tmp_path)
    monkeypatch.setattr(prepare, "audio_check_and_hash", _fake_audio_check)
    out = tmp_path / "adaptive"
    audit = prepare.audit_candidates(tmp_path, cfg, out)
    assert audit["passed"] is True
    assert audit["overlaps"]["protected"] == 0
    assert audit["rejections"]["protected_id"] == 1
    assert audit["rejections"]["abnormal_unicode"] == 1

    eligible_path = out / "eligible_candidates.jsonl"
    eligible_sha = prepare.sha256_file(eligible_path)
    losses_path = tmp_path / "candidate_losses.jsonl"
    loss_rows = [
        {
            "id": row["id"],
            "normalized_loss": index / 10.0,
            "eligible_manifest_sha256": eligible_sha,
        }
        for index, row in enumerate(read_jsonl(eligible_path))
    ]
    _write_jsonl(losses_path, loss_rows)
    receipt = prepare.freeze_manifests(tmp_path, cfg, out, losses_path)

    assert (out / "MANIFESTS_FROZEN").read_text().strip() == "PASS"
    assert receipt["passed"] is True
    assert receipt["overlaps"]["protected"] == 0
    assert receipt["boundaries"]["40.0"]["rows"] % 16 == 0
    assert receipt["boundaries"]["45.0"]["prefix_of_50h"] is True
    rows40 = read_jsonl(Path(receipt["boundaries"]["40.0"]["path"]))
    rows45 = read_jsonl(Path(receipt["boundaries"]["45.0"]["path"]))
    rows50 = read_jsonl(Path(receipt["boundaries"]["50.0"]["path"]))
    assert rows45[: len(rows40)] == rows40
    assert rows50[: len(rows45)] == rows45
    arms = json.loads((out / "stage40_arms.json").read_text(encoding="utf-8"))["arms"]
    assert [arm["name"] for arm in arms] == [
        "OLDSTREAM_HALF_LR",
        "BALANCED_ORIG_LR",
        "BALANCED_HALF_LR",
        "BALANCED_QUARTER_LR",
    ]
    assert {arm["wenet_continuation_cursor"] for arm in arms} == {0}
    assert {arm["official_sample_cursor"] for arm in arms} == {7712}


def test_freeze_rejects_incomplete_or_stale_loss_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, _ = _fixture(tmp_path)
    monkeypatch.setattr(prepare, "audio_check_and_hash", _fake_audio_check)
    out = tmp_path / "adaptive"
    prepare.audit_candidates(tmp_path, cfg, out)
    eligible = read_jsonl(out / "eligible_candidates.jsonl")
    bad_losses = tmp_path / "bad_losses.jsonl"
    _write_jsonl(
        bad_losses,
        [
            {
                "id": row["id"],
                "normalized_loss": 1.0,
                "eligible_manifest_sha256": "stale",
            }
            for row in eligible[:-1]
        ],
    )
    with pytest.raises(SystemExit, match="exact eligible manifest"):
        prepare.freeze_manifests(tmp_path, cfg, out, bad_losses)
    assert not (out / "MANIFESTS_FROZEN").exists()
    assert not (out / "manifests").exists()
