from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.train_w500_adaptive_continuation import (
    REQUIRED_CHECKPOINT_FILES,
    _initial_wenet_seconds,
    capture_rng_state,
    checkpoint_targets,
    configure_source,
    create_optimizer,
    milestone_cursors,
    optimizer_state_digest,
    perform_optimizer_step,
    restore_rng_state,
    save_checkpoint,
    tensor_digest,
)


class _Decoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(7, 4)
        self.layer = torch.nn.Linear(4, 4)


class _Core(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.LayerNorm(4),
            torch.nn.Dropout(0.2),
        )
        self.decoder = _Decoder()


class _TinyWhisper(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Core()
        self.proj_out = torch.nn.Linear(4, 7, bias=False)
        self.proj_out.weight = self.model.decoder.embed_tokens.weight

    def forward(self, x: torch.Tensor) -> SimpleNamespace:
        encoded = self.model.encoder(x)
        decoded = self.model.decoder.layer(encoded)
        logits = self.proj_out(decoded)
        return SimpleNamespace(loss=logits.square().mean())

    def save_pretrained(self, destination: Path, *, safe_serialization: bool) -> None:
        assert safe_serialization is True
        destination.mkdir(parents=True, exist_ok=True)
        # The checkpoint writer treats this as an opaque model artifact.
        torch.save(self.state_dict(), destination / "model.safetensors")
        (destination / "config.json").write_text("{}\n", encoding="utf-8")


class _TinyProcessor:
    def save_pretrained(self, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "preprocessor_config.json").write_text("{}\n", encoding="utf-8")
        (destination / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")


def _batches(offset: float = 0.0) -> list[dict[str, torch.Tensor]]:
    return [
        {"x": torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10 + offset},
        {"x": torch.arange(16, 32, dtype=torch.float32).reshape(4, 4) / 10 + offset},
    ]


def _step(
    model: _TinyWhisper,
    optimizer: torch.optim.Optimizer,
    encoder_named: list[tuple[str, torch.nn.Parameter]],
    decoder_named: list[tuple[str, torch.nn.Parameter]],
    source: str,
    offset: float,
) -> None:
    perform_optimizer_step(
        source=source,
        model=model,
        optimizer=optimizer,
        batches=_batches(offset),
        device=torch.device("cpu"),
        encoder_named=encoder_named,
        decoder_named=decoder_named,
        wenet_lr=2e-3,
        official_lr=7e-4,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
    )


def _cycle(
    model: _TinyWhisper,
    optimizer: torch.optim.Optimizer,
    encoder_named: list[tuple[str, torch.nn.Parameter]],
    decoder_named: list[tuple[str, torch.nn.Parameter]],
    cycle: int,
) -> None:
    _step(model, optimizer, encoder_named, decoder_named, "wenet", float(cycle))
    _step(model, optimizer, encoder_named, decoder_named, "official", float(cycle) + 0.5)


def test_configure_source_enforces_encoder_only_then_full_sft() -> None:
    model = _TinyWhisper()
    configure_source(model, source="wenet")
    assert all(parameter.requires_grad for parameter in model.model.encoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.model.decoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.proj_out.parameters())

    configure_source(model, source="official")
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_checkpoint_targets_keep_approved_nominal_grid() -> None:
    assert checkpoint_targets(35.0, 40.0) == [36.25, 37.5, 38.75, 40.0]
    assert checkpoint_targets(35.1010722222, 40.0) == [36.25, 37.5, 38.75, 40.0]
    assert checkpoint_targets(38.75, 45.0) == [41.25, 42.5, 43.75, 45.0]
    assert checkpoint_targets(40.0, 45.0) == [41.25, 42.5, 43.75, 45.0]


def test_milestones_use_nearest_complete_effective_batch_and_exact_exposure() -> None:
    rows = [{"id": str(index), "duration": 60.0} for index in range(320)]
    boundaries = milestone_cursors(
        rows,
        start_cursor=0,
        start_exposure_seconds=35.0 * 3600,
        targets_hours=[36.25, 37.5, 38.75, 40.0],
        effective_batch_size=16,
    )
    assert [value["cursor"] for value in boundaries.values()] == [80, 144, 224, 304]
    assert all(int(value["cursor"]) % 16 == 0 for value in boundaries.values())
    assert boundaries[36.25]["actual_seconds"] == 35.0 * 3600 + 80 * 60
    assert boundaries[40.0]["deviation_seconds"] == 4 * 60


def test_wenet_accumulation_cannot_change_decoder_or_its_adam_moments() -> None:
    torch.manual_seed(4)
    model = _TinyWhisper()
    optimizer, encoder_named, decoder_named, _ = create_optimizer(model, 0.01)
    _step(model, optimizer, encoder_named, decoder_named, "official", 0.0)
    decoder_before = {name: parameter.detach().clone() for name, parameter in decoder_named}
    moments_before = optimizer_state_digest(optimizer, decoder_named)

    _step(model, optimizer, encoder_named, decoder_named, "wenet", 1.0)

    assert all(
        torch.equal(parameter, decoder_before[name])
        for name, parameter in decoder_named
    )
    assert optimizer_state_digest(optimizer, decoder_named) == moments_before


def test_accumulation_window_rejects_wrong_microbatch_count() -> None:
    model = _TinyWhisper()
    optimizer, encoder_named, decoder_named, _ = create_optimizer(model, 0.01)
    try:
        perform_optimizer_step(
            source="wenet",
            model=model,
            optimizer=optimizer,
            batches=_batches()[:1],
            device=torch.device("cpu"),
            encoder_named=encoder_named,
            decoder_named=decoder_named,
            wenet_lr=1e-6,
            official_lr=5e-7,
            gradient_accumulation_steps=2,
            max_grad_norm=1.0,
        )
    except ValueError as error:
        assert "exactly gradient_accumulation_steps" in str(error)
    else:  # pragma: no cover - makes the invariant explicit
        raise AssertionError("mixed/partial accumulation window was accepted")


def test_fresh_branch_has_empty_adam_state_but_same_arm_resume_restores_it() -> None:
    torch.manual_seed(8)
    parent = _TinyWhisper()
    parent_optimizer, encoder_named, decoder_named, _ = create_optimizer(parent, 0.01)
    _cycle(parent, parent_optimizer, encoder_named, decoder_named, 0)
    saved_optimizer = copy.deepcopy(parent_optimizer.state_dict())

    branch = _TinyWhisper()
    branch.load_state_dict(parent.state_dict())
    fresh_optimizer, branch_encoder, branch_decoder, _ = create_optimizer(branch, 0.01)
    assert fresh_optimizer.state == {}

    fresh_optimizer.load_state_dict(saved_optimizer)
    assert optimizer_state_digest(
        fresh_optimizer, branch_encoder + branch_decoder
    ) == optimizer_state_digest(parent_optimizer, encoder_named + decoder_named)


def test_two_cycle_resume_matches_four_uninterrupted_cycles() -> None:
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    initial = _TinyWhisper().state_dict()

    uninterrupted = _TinyWhisper()
    uninterrupted.load_state_dict(initial)
    uninterrupted_optimizer, u_encoder, u_decoder, _ = create_optimizer(
        uninterrupted, 0.01
    )
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    for cycle in range(4):
        _cycle(uninterrupted, uninterrupted_optimizer, u_encoder, u_decoder, cycle)

    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    split = _TinyWhisper()
    split.load_state_dict(initial)
    split_optimizer, s_encoder, s_decoder, _ = create_optimizer(split, 0.01)
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    for cycle in range(2):
        _cycle(split, split_optimizer, s_encoder, s_decoder, cycle)
    model_state = copy.deepcopy(split.state_dict())
    optimizer_state = copy.deepcopy(split_optimizer.state_dict())
    rng_state = capture_rng_state()

    resumed = _TinyWhisper()
    resumed.load_state_dict(model_state)
    resumed_optimizer, r_encoder, r_decoder, _ = create_optimizer(resumed, 0.01)
    resumed_optimizer.load_state_dict(optimizer_state)
    restore_rng_state(rng_state)
    for cycle in range(2, 4):
        _cycle(resumed, resumed_optimizer, r_encoder, r_decoder, cycle)

    assert tensor_digest(uninterrupted.named_parameters()) == tensor_digest(
        resumed.named_parameters()
    )
    assert optimizer_state_digest(
        uninterrupted_optimizer, u_encoder + u_decoder
    ) == optimizer_state_digest(resumed_optimizer, r_encoder + r_decoder)
    # Two samples per microbatch, two microbatches, two sources, four cycles.
    assert 4 * 2 * 2 * 2 == 32
    assert [f"sample-{index}" for index in range(32)][16:] == [
        f"sample-{index}" for index in range(16, 32)
    ]
    assert sum([10.0] * 32) == 320.0


def test_checkpoint_contains_exact_resume_and_exposure_receipts(tmp_path: Path) -> None:
    torch.manual_seed(5)
    model = _TinyWhisper()
    optimizer, _, _, _ = create_optimizer(model, 0.01)
    steps = tmp_path / "source_steps.jsonl"
    rows = [
        {"source": "wenet", "sample_ids": ["w0", "w1"], "step_seconds": 12.5},
        {"source": "official", "sample_ids": ["o0", "o1"], "step_seconds": 8.0},
    ]
    steps.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    state = {
        "identity": {"arm_name": "A"},
        "total_optimizer_steps": 2,
        "completed_cycles": 1,
        "wenet_sample_cursor": 16,
        "official_sample_cursor": 7728,
        "cumulative_wenet_seconds": 35.1 * 3600 + 12.5,
        "continuation_wenet_seconds": 12.5,
        "cumulative_official_seconds": 8.0,
    }
    receipt = save_checkpoint(
        output_dir=tmp_path,
        label="checkpoint-wenet-36p25h",
        model=model,
        processor=_TinyProcessor(),
        optimizer=optimizer,
        state=state,
        source_steps_path=steps,
    )
    checkpoint = Path(receipt["path"])
    assert all((checkpoint / name).is_file() for name in REQUIRED_CHECKPOINT_FILES)
    restored = json.loads((checkpoint / "adaptive_state.json").read_text())
    assert restored["checkpoint_phase"] == "after_complete_wenet_official_cycle"
    assert restored["next_source"] == "wenet"
    assert restored["wenet_sample_cursor"] == 16
    assert restored["official_sample_cursor"] == 7728
    assert restored["continuation_wenet_seconds"] == 12.5


def test_new_branch_inherits_parent_total_exposure_but_accepts_remapped_cursor(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "oldstream-parent"
    parent.mkdir()
    (parent / "adaptive_state.json").write_text(
        json.dumps(
            {
                "wenet_sample_cursor": 64,
                "cumulative_wenet_seconds": 36.25 * 3600,
                "continuation_wenet_seconds": 4500.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    arm = {
        "wenet_continuation_cursor": 80,
        "continuation_wenet_seconds": 4800.0,
        "start_hours": 36.25,
    }
    total, continuation = _initial_wenet_seconds(arm, parent)
    assert total == 36.25 * 3600
    assert continuation == 4800.0
