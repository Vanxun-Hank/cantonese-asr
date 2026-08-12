from __future__ import annotations

import torch

from cantonese_asr.training_sampling import (
    FixedExposureSampler,
    fixed_exposure_topology_receipt,
)
from scripts.convert_openai_whisper_checkpoint import renamed_state_dict


def test_fixed_exposure_sampler_is_sequential() -> None:
    assert list(FixedExposureSampler(5)) == [0, 1, 2, 3, 4]


def test_fixed_exposure_global_steps_match_topologies() -> None:
    ids = [f"id-{index}" for index in range(32)]
    receipts = [
        fixed_exposure_topology_receipt(
            ids,
            world_size=world,
            per_device_batch=batch,
            gradient_accumulation_steps=accum,
        )
        for world, batch, accum in ((1, 8, 2), (2, 2, 4), (4, 1, 4))
    ]
    assert {item["global_effective_batch"] for item in receipts} == {16}
    assert len({item["global_step_ids_sha256"] for item in receipts}) == 1
    assert all(item["optimizer_steps"] == 2 for item in receipts)
    assert all(item["steps"][0]["global_ids"] == ids[:16] for item in receipts)


def test_fixed_exposure_rejects_partial_global_batch() -> None:
    try:
        fixed_exposure_topology_receipt(
            ["a", "b"],
            world_size=1,
            per_device_batch=8,
            gradient_accumulation_steps=2,
        )
    except ValueError as error:
        assert "not divisible" in str(error)
    else:
        raise AssertionError("partial global batch must fail")


def test_openai_whisper_key_conversion() -> None:
    tensor = torch.ones(1)
    converted = renamed_state_dict(
        {
            "encoder.blocks.0.attn.query.weight": tensor,
            "decoder.token_embedding.weight": tensor,
        }
    )
    assert set(converted) == {
        "encoder.layers.0.self_attn.q_proj.weight",
        "decoder.embed_tokens.weight",
    }
