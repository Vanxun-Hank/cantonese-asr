from __future__ import annotations

import json
from collections import Counter

from cantonese_asr.training_sampling import (
    EpochEncodedSampler,
    avoid_batch_duplicates,
    build_speed_assignments,
    exp004_wenet_source_group,
    largest_remainder_counts,
    round11_source_group,
    round12_source_group,
    source_group,
)


def exp004_rows() -> list[dict[str, str]]:
    rows = [
        {"id": f"o-{index}", "source": "official"}
        for index in range(6_292)
    ]
    rows.extend(
        {
            "id": f"c-{index}",
            "source": "common_voice_26_zh_HK",
        }
        for index in range(8_451)
    )
    rows.extend(
        {"id": f"m-{index}", "source": "mdcc"}
        for index in range(8_561)
    )
    return rows


def decoded_indices(sampler: EpochEncodedSampler) -> list[int]:
    width = len(sampler.rows)
    return [encoded % width for encoded in sampler]


def test_source_group_handles_yue_as_cv() -> None:
    assert source_group({"source": "official"}) == "official"
    assert source_group({"source": "common_voice_26_yue"}) == "cv"
    assert source_group({"source": "mdcc"}) == "mdcc"


def test_round11_source_group_distinguishes_cv_variants() -> None:
    assert round11_source_group({"source": "official"}) == "official"
    assert (
        round11_source_group({"source": "common_voice_26_zh_HK"})
        == "cv_zh_hk"
    )
    assert (
        round11_source_group({"source": "common_voice_26_yue"})
        == "cv_yue"
    )
    assert round11_source_group({"source": "mdcc"}) == "mdcc"


def test_round12_source_group_distinguishes_wenet_and_official() -> None:
    assert round12_source_group({"source": "official"}) == "official"
    assert round12_source_group({"source": "WenetSpeech_Yue"}) == "wenet"


def test_speed_assignments_are_exact_balanced_and_change_by_epoch() -> None:
    rows = exp004_rows()
    assignments = build_speed_assignments(rows, seed=42, epochs=3)
    for assignment in assignments:
        assert Counter(assignment.values()) == {"0.9": 2_098, "1.0": 2_097, "1.1": 2_097}
    assert assignments[0] != assignments[1]


def test_official_speed_sampler_preserves_exp004_pool(tmp_path) -> None:
    rows = exp004_rows()
    sampler = EpochEncodedSampler(
        rows,
        policy="official_speed",
        seed=42,
        batch_size=8,
        receipt_dir=tmp_path,
    )
    first = decoded_indices(sampler)
    sampler.set_epoch(1)
    second = decoded_indices(sampler)
    assert len(first) == len(second) == 23_304
    assert len(set(first)) == len(set(second)) == 23_304
    assert first != second
    receipt = json.loads((tmp_path / "epoch_2.json").read_text())
    assert receipt["source_draws"] == {"cv": 8_451, "mdcc": 8_561, "official": 6_292}


def test_source_balanced_sampler_exact_counts_rotates_and_has_no_batch_duplicates(
    tmp_path,
) -> None:
    rows = exp004_rows()
    sampler = EpochEncodedSampler(
        rows,
        policy="source_balanced",
        seed=42,
        batch_size=8,
        receipt_dir=tmp_path,
    )
    first = decoded_indices(sampler)
    first_groups = Counter(source_group(rows[index]) for index in first)
    sampler.set_epoch(1)
    second = decoded_indices(sampler)
    second_groups = Counter(source_group(rows[index]) for index in second)
    assert first_groups == second_groups == {
        "official": 7_768,
        "cv": 7_768,
        "mdcc": 7_768,
    }
    assert len(first) == len(second) == 23_304
    assert all(
        len(batch) == len(set(batch))
        for batch in (
            first[start : start + 8] for start in range(0, len(first), 8)
        )
    )
    first_cv = {index for index in first if source_group(rows[index]) == "cv"}
    second_cv = {index for index in second if source_group(rows[index]) == "cv"}
    assert first_cv != second_cv
    assert len(first_cv | second_cv) > len(first_cv)
    receipt = json.loads((tmp_path / "epoch_2.json").read_text())
    assert receipt["within_minibatch_duplicates"] == 0
    assert receipt["total_draws"] == 23_304


def test_round11_four_source_sampler_uses_fixed_exp004_budget(tmp_path) -> None:
    rows = [
        {"id": f"o-{index}", "source": "official"}
        for index in range(6_292)
    ]
    rows.extend(
        {"id": f"h-{index}", "source": "common_voice_26_zh_HK"}
        for index in range(8_451)
    )
    rows.extend(
        {"id": f"y-{index}", "source": "common_voice_26_yue"}
        for index in range(1_688)
    )
    rows.extend(
        {"id": f"m-{index}", "source": "mdcc"}
        for index in range(64_779)
    )
    sampler = EpochEncodedSampler(
        rows,
        policy="round11_source_balanced",
        seed=42,
        batch_size=8,
        receipt_dir=tmp_path,
    )
    sampled = decoded_indices(sampler)
    assert len(sampled) == 23_304
    assert Counter(round11_source_group(rows[index]) for index in sampled) == {
        "official": 5_826,
        "cv_zh_hk": 5_826,
        "cv_yue": 5_826,
        "mdcc": 5_826,
    }
    receipt = json.loads((tmp_path / "epoch_1.json").read_text())
    assert receipt["within_minibatch_duplicates"] == 0


def test_round12_sampler_has_exact_80_20_windows_and_no_batch_duplicates(
    tmp_path,
) -> None:
    rows = [
        {"id": f"w-{index}", "source": "WenetSpeech_Yue"}
        for index in range(101)
    ]
    rows.extend(
        {"id": f"o-{index}", "source": "official"}
        for index in range(31)
    )
    sampler = EpochEncodedSampler(
        rows,
        policy="wenet_official_80_20",
        seed=42,
        batch_size=8,
        receipt_dir=tmp_path,
        round12_stream_length=6_400,
    )
    sampled = decoded_indices(sampler)
    assert len(sampled) == 6_400
    assert all(
        Counter(round12_source_group(rows[index]) for index in sampled[start : start + 6_400])
        == {"wenet": 5_120, "official": 1_280}
        for start in range(0, len(sampled), 6_400)
    )
    assert all(
        len(batch) == len(set(batch))
        for batch in (
            sampled[start : start + 8] for start in range(0, len(sampled), 8)
        )
    )
    receipt = json.loads((tmp_path / "epoch_1.json").read_text())
    assert receipt["source_draws"] == {"official": 1_280, "wenet": 5_120}


def test_exp004_wenet_sampler_preserves_nested_ratios_and_audit_windows(
    tmp_path,
) -> None:
    for ratio in (0.0, 0.05, 0.10, 0.20):
        rows = exp004_rows()
        if ratio:
            rows.extend(
                {"id": f"w-{index}", "source": "WenetSpeech_Yue"}
                for index in range(22_636)
            )
        receipt_dir = tmp_path / str(ratio)
        sampler = EpochEncodedSampler(
            rows,
            policy="exp004_wenet_mix",
            seed=42,
            batch_size=8,
            receipt_dir=receipt_dir,
            exp004_wenet_ratio=ratio,
            exp004_wenet_stream_length=3_200,
            exp004_wenet_window_length=1_600,
        )
        sampled = decoded_indices(sampler)
        assert len(sampled) == 3_200
        assert all(
            len(batch) == len(set(batch))
            for batch in (
                sampled[start : start + 8]
                for start in range(0, len(sampled), 8)
            )
        )
        for start in range(0, len(sampled), 1_600):
            groups = Counter(
                exp004_wenet_source_group(rows[index])
                for index in sampled[start : start + 1_600]
            )
            assert groups.get("wenet", 0) == int(1_600 * ratio)
            assert sum(groups[group] for group in ("official", "cv", "mdcc")) == int(
                1_600 * (1 - ratio)
            )
        overall = Counter(
            exp004_wenet_source_group(rows[index]) for index in sampled
        )
        expected_exp004 = largest_remainder_counts(
            3_200 - int(3_200 * ratio),
            {"official": 6_292, "cv": 8_451, "mdcc": 8_561},
        )
        assert {
            group: overall[group]
            for group in ("official", "cv", "mdcc")
        } == expected_exp004
        assert overall.get("wenet", 0) == int(3_200 * ratio)
        receipt = json.loads(
            (receipt_dir / "epoch_1.json").read_text(encoding="utf-8")
        )
        assert receipt["within_minibatch_duplicates"] == 0
        assert receipt["target_wenet_ratio"] == ratio


def test_exp004_wenet_sampler_supports_exact_additive_quotas(tmp_path) -> None:
    rows = exp004_rows()
    rows.extend(
        {"id": f"w-{index}", "source": "WenetSpeech_Yue"}
        for index in range(22_636)
    )
    arms = {
        "baseline": (160, 32, 11_200),
        "add_1_5x": (176, 48, 12_320),
        "add_2_0x": (192, 64, 13_440),
        "add_2_5x": (208, 80, 14_560),
    }
    for arm, (window_length, wenet_per_window, stream_length) in arms.items():
        receipt_dir = tmp_path / arm
        sampler = EpochEncodedSampler(
            rows,
            policy="exp004_wenet_mix",
            seed=42,
            batch_size=8,
            receipt_dir=receipt_dir,
            exp004_wenet_stream_length=stream_length,
            exp004_wenet_window_length=window_length,
            exp004_wenet_per_window=wenet_per_window,
        )
        sampled = decoded_indices(sampler)
        assert len(sampled) == stream_length
        for start in range(0, len(sampled), window_length):
            groups = Counter(
                exp004_wenet_source_group(rows[index])
                for index in sampled[start : start + window_length]
            )
            assert groups["wenet"] == wenet_per_window
            assert sum(groups[group] for group in ("official", "cv", "mdcc")) == 128
        overall = Counter(
            exp004_wenet_source_group(rows[index]) for index in sampled
        )
        assert overall["official"] + overall["cv"] + overall["mdcc"] == 8_960
        assert overall["wenet"] == 70 * wenet_per_window
        assert all(
            len(batch) == len(set(batch))
            for batch in (
                sampled[start : start + 8]
                for start in range(0, len(sampled), 8)
            )
        )
        receipt = json.loads(
            (receipt_dir / "epoch_1.json").read_text(encoding="utf-8")
        )
        assert receipt["quota_mode"] == "exact_counts"
        assert receipt["wenet_per_audit_window"] == wenet_per_window
        assert receipt["exp004_per_audit_window"] == 128


def test_fixed_wenet_manifest_order_preserves_frozen_rows(tmp_path) -> None:
    rows = exp004_rows()
    rows.extend(
        {"id": f"w-{index}", "source": "WenetSpeech_Yue"}
        for index in range(16)
    )
    sampler = EpochEncodedSampler(
        rows,
        policy="exp004_wenet_mix",
        seed=42,
        batch_size=8,
        receipt_dir=tmp_path,
        exp004_wenet_stream_length=80,
        exp004_wenet_window_length=80,
        exp004_wenet_per_window=16,
        exp004_wenet_fixed_manifest_order=True,
    )
    indices = decoded_indices(sampler)
    wenet_indices = [
        index
        for index in indices
        if exp004_wenet_source_group(rows[index]) == "wenet"
    ]
    assert wenet_indices == list(range(23_304, 23_320))
    receipt = json.loads((tmp_path / "epoch_1.json").read_text())
    assert receipt["wenet_fixed_manifest_order"] is True


def test_avoid_batch_duplicates_swaps_repeated_indices() -> None:
    fixed = avoid_batch_duplicates([1, 1, 2, 3, 4, 5], batch_size=3)
    assert len(fixed) == 6
    assert sorted(fixed) == [1, 1, 2, 3, 4, 5]
    assert len(set(fixed[:3])) == 3
