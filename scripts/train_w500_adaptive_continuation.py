#!/usr/bin/env python3
"""Train one branch of the adaptive W500 continuation tournament.

Each optimizer step is source-homogeneous.  Wenet steps update only Whisper's
Encoder, while Official steps perform full-model SFT.  Checkpoints are emitted
only after a complete Wenet -> Official cycle and carry enough state to resume
the *same* arm exactly.  Loading a selected parent starts a new branch and
therefore deliberately creates a fresh AdamW optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Sampler
from transformers import WhisperForConditionalGeneration, WhisperProcessor, set_seed
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

from cantonese_asr.io import read_jsonl


METADATA_KEYS = (
    "_augmentation_speed_factor",
    "_augmentation_noise_applied",
    "_augmentation_snr_db",
    "_training_source_code",
)
REQUIRED_CHECKPOINT_FILES = (
    "adaptive_state.json",
    "optimizer.pt",
    "trainer_state.json",
    "rng_state.pth",
    "source_steps.jsonl",
    "model.safetensors",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--arm-file", type=Path, required=True)
    parser.add_argument("--arm-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke-cycles", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def _resolve(root: Path, path: Path | str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FixedSequenceSampler(Sampler[int]):
    def __init__(self, indices: list[int]) -> None:
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def deterministic_official_indices(rows: int, total: int, seed: int) -> list[int]:
    if rows <= 0:
        raise ValueError("Official manifest must not be empty")
    result: list[int] = []
    epoch = 0
    while len(result) < total:
        generator = torch.Generator()
        generator.manual_seed(int(seed) + epoch)
        result.extend(torch.randperm(rows, generator=generator).tolist())
        epoch += 1
    return result[:total]


def tensor_digest(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def optimizer_state_digest(
    optimizer: torch.optim.Optimizer,
    named_parameters: list[tuple[str, torch.nn.Parameter]],
) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(named_parameters, key=lambda item: item[0]):
        digest.update(name.encode("utf-8"))
        state = optimizer.state.get(parameter, {})
        for key in sorted(state):
            value = state[key]
            digest.update(str(key).encode("utf-8"))
            if torch.is_tensor(value):
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def create_optimizer(
    model: torch.nn.Module, weight_decay: float
) -> tuple[
    torch.optim.AdamW,
    list[tuple[str, torch.nn.Parameter]],
    list[tuple[str, torch.nn.Parameter]],
    dict[str, Any],
]:
    """Create the stable AdamW grouping used by the proven alternating trainer."""

    encoder_ids = {id(parameter) for parameter in model.model.encoder.parameters()}
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    encoder_named = [
        (name, parameter) for name, parameter in named if id(parameter) in encoder_ids
    ]
    decoder_named = [
        (name, parameter) for name, parameter in named if id(parameter) not in encoder_ids
    ]
    decay_names = set(get_parameter_names(model, ALL_LAYERNORM_LAYERS))
    decay_names = {name for name in decay_names if "bias" not in name}
    groups: list[dict[str, Any]] = []
    for component, values in (("encoder", encoder_named), ("decoder", decoder_named)):
        for decay in (True, False):
            parameters = [
                parameter for name, parameter in values if (name in decay_names) == decay
            ]
            if parameters:
                groups.append(
                    {
                        "params": parameters,
                        "lr": 0.0,
                        "weight_decay": float(weight_decay) if decay else 0.0,
                        "source_component": component,
                        "decay": decay,
                    }
                )
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
    receipt = {
        "encoder_parameter_names": [name for name, _ in encoder_named],
        "decoder_parameter_names": [name for name, _ in decoder_named],
        "encoder_parameter_count": sum(parameter.numel() for _, parameter in encoder_named),
        "decoder_parameter_count": sum(parameter.numel() for _, parameter in decoder_named),
        "total_optimizer_parameter_count": sum(parameter.numel() for _, parameter in named),
        "proj_out_tied_to_decoder_embedding": (
            model.proj_out.weight is model.model.decoder.embed_tokens.weight
        ),
        "parameter_groups": [
            {
                "source_component": group["source_component"],
                "decay": group["decay"],
                "weight_decay": group["weight_decay"],
                "parameter_count": sum(parameter.numel() for parameter in group["params"]),
            }
            for group in groups
        ],
    }
    return optimizer, encoder_named, decoder_named, receipt


def checkpoint_targets(start_hours: float, target_hours: float) -> list[float]:
    """Return the four preregistered quarter-stage exposure targets.

    The root's measured exposure is 35.101... hours, but the approved targets
    remain 36.25/37.5/38.75/40.  Likewise, a Top-2 parent may be an earlier
    checkpoint.  Five-hour tournament endpoints therefore define the nominal
    stage grid; generic intervals retain the straightforward quarter split.
    """

    start = float(start_hours)
    target = float(target_hours)
    if not math.isfinite(start) or not math.isfinite(target) or target <= start:
        raise ValueError(f"Invalid stage interval: {start_hours} -> {target_hours}")
    if target in {40.0, 45.0, 50.0} and start < target:
        nominal_start = target - 5.0
    else:
        nominal_start = start
    increment = (target - nominal_start) / 4.0
    return [round(nominal_start + increment * index, 6) for index in range(1, 5)]


def configure_source(model: torch.nn.Module, *, source: str) -> None:
    """Apply the approved Wenet Encoder-only or Official Full-SFT topology."""

    if source not in {"wenet", "official"}:
        raise ValueError(f"Unknown source: {source}")
    for parameter in model.model.encoder.parameters():
        parameter.requires_grad_(True)
    decoder_enabled = source == "official"
    for parameter in model.model.decoder.parameters():
        parameter.requires_grad_(decoder_enabled)
    for parameter in model.proj_out.parameters():
        parameter.requires_grad_(decoder_enabled)


def set_source_mode(
    source: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    wenet_lr: float,
    official_lr: float,
) -> None:
    """Switch topology and all optimizer group LRs at one step boundary."""

    configure_source(model, source=source)
    for group in optimizer.param_groups:
        component = group.get("source_component")
        if component not in {"encoder", "decoder"}:
            raise RuntimeError(f"Optimizer group lacks source component: {component!r}")
        if source == "wenet":
            group["lr"] = float(wenet_lr) if component == "encoder" else 0.0
        else:
            group["lr"] = float(official_lr)


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    clean = {key: value for key, value in batch.items() if key not in METADATA_KEYS}
    return {key: value.to(device, non_blocking=True) for key, value in clean.items()}


def perform_optimizer_step(
    *,
    source: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: Sequence[dict[str, torch.Tensor]],
    device: torch.device,
    encoder_named: list[tuple[str, torch.nn.Parameter]],
    decoder_named: list[tuple[str, torch.nn.Parameter]],
    wenet_lr: float,
    official_lr: float,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
) -> dict[str, float]:
    """Run exactly one source-homogeneous accumulated optimizer step."""

    if len(batches) != int(gradient_accumulation_steps):
        raise ValueError(
            "One optimizer step must contain exactly gradient_accumulation_steps "
            f"microbatches; got {len(batches)}"
        )
    set_source_mode(
        source,
        model,
        optimizer,
        wenet_lr=wenet_lr,
        official_lr=official_lr,
    )
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    for raw_batch in batches:
        batch = move_batch(raw_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            loss = model(**batch).loss
        losses.append(float(loss.detach().cpu()))
        (loss / int(gradient_accumulation_steps)).backward()

    if source == "wenet":
        offenders = [name for name, parameter in decoder_named if parameter.grad is not None]
        if offenders:
            raise RuntimeError(f"Inactive gradients during wenet: {offenders[:20]}")
    else:
        encoder_has_grad = any(parameter.grad is not None for _, parameter in encoder_named)
        decoder_has_grad = any(parameter.grad is not None for _, parameter in decoder_named)
        if not encoder_has_grad or not decoder_has_grad:
            raise RuntimeError(
                "Official Full-SFT gradient audit failed: "
                f"encoder={encoder_has_grad} decoder={decoder_has_grad}"
            )
    active = [
        parameter for parameter in model.parameters() if parameter.grad is not None
    ]
    grad_norm_value = clip_grad_norm_(active, float(max_grad_norm))
    grad_norm = float(torch.as_tensor(grad_norm_value).detach().cpu())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {"loss": statistics.mean(losses), "grad_norm": grad_norm}


def row_duration_seconds(row: dict[str, Any]) -> float:
    value = row.get("duration_s", row.get("duration", row.get("seconds")))
    if value is None:
        raise ValueError(f"Manifest row lacks duration: {row.get('id', '<unknown>')}")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError(f"Invalid duration {value!r}: {row.get('id', '<unknown>')}")
    return seconds


def row_identifier(row: dict[str, Any], index: int) -> str:
    value = row.get("id", row.get("audio_path"))
    if value is None or not str(value).strip():
        raise ValueError(f"Manifest row {index} lacks id and audio_path")
    return str(value)


def milestone_cursors(
    rows: list[dict[str, Any]],
    *,
    start_cursor: int,
    start_exposure_seconds: float,
    targets_hours: Iterable[float],
    effective_batch_size: int,
) -> dict[float, dict[str, float | int]]:
    """Choose the complete-step boundary nearest each future exposure target."""

    if start_cursor < 0 or start_cursor > len(rows):
        raise ValueError(f"Invalid Wenet cursor: {start_cursor}")
    if effective_batch_size <= 0 or start_cursor % effective_batch_size:
        raise ValueError("Wenet cursor must be aligned to the effective batch size")
    candidates: list[tuple[int, float]] = []
    total = float(start_exposure_seconds)
    for cursor in range(start_cursor + effective_batch_size, len(rows) + 1, effective_batch_size):
        total += sum(
            row_duration_seconds(row)
            for row in rows[cursor - effective_batch_size : cursor]
        )
        candidates.append((cursor, total))
    result: dict[float, dict[str, float | int]] = {}
    used_cursor = start_cursor
    for target in targets_hours:
        target = float(target)
        if target * 3600.0 <= start_exposure_seconds:
            continue
        available = [item for item in candidates if item[0] > used_cursor]
        if not available:
            raise ValueError(f"Manifest cannot reach {target:g} Wenet hours")
        cursor, seconds = min(
            available,
            key=lambda item: (abs(item[1] - target * 3600.0), item[0]),
        )
        result[target] = {
            "cursor": cursor,
            "actual_seconds": seconds,
            "actual_hours": seconds / 3600.0,
            "deviation_seconds": seconds - target * 3600.0,
        }
        used_cursor = cursor
    return result


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older server torch
        return torch.load(path, map_location="cpu")


def _checkpoint_hashes(staging: Path) -> dict[str, str]:
    names = [name for name in REQUIRED_CHECKPOINT_FILES if (staging / name).is_file()]
    names.extend(
        name
        for name in (
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
        )
        if (staging / name).is_file()
    )
    return {name: sha256_file(staging / name) for name in sorted(set(names))}


def save_checkpoint(
    *,
    output_dir: Path,
    label: str,
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
    source_steps_path: Path,
) -> dict[str, Any]:
    destination = output_dir / label
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite checkpoint: {destination}")
    staging = destination.with_name(destination.name + ".staging")
    if staging.exists():
        raise RuntimeError(f"Stale checkpoint staging directory: {staging}")
    staging.mkdir(parents=True)
    try:
        model.save_pretrained(staging, safe_serialization=True)
        processor.save_pretrained(staging)
        torch.save(optimizer.state_dict(), staging / "optimizer.pt")
        torch.save(capture_rng_state(), staging / "rng_state.pth")
        shutil.copy2(source_steps_path, staging / "source_steps.jsonl")
        saved_state = dict(state)
        saved_state.update(
            {
                "checkpoint_phase": "after_complete_wenet_official_cycle",
                "next_source": "wenet",
                "model_safetensors_sha256": sha256_file(staging / "model.safetensors"),
                "optimizer_state_sha256": sha256_file(staging / "optimizer.pt"),
                "source_steps_sha256": sha256_file(staging / "source_steps.jsonl"),
            }
        )
        write_json(staging / "adaptive_state.json", saved_state)
        write_json(
            staging / "trainer_state.json",
            {
                "global_step": int(saved_state["total_optimizer_steps"]),
                "completed_cycles": int(saved_state["completed_cycles"]),
                "wenet_sample_cursor": int(saved_state["wenet_sample_cursor"]),
                "official_sample_cursor": int(saved_state["official_sample_cursor"]),
                "is_local_process_zero": True,
                "is_world_process_zero": True,
            },
        )
        hashes = _checkpoint_hashes(staging)
        missing = [name for name in REQUIRED_CHECKPOINT_FILES if name not in hashes]
        if missing:
            raise RuntimeError(f"Incomplete checkpoint staging directory: {missing}")
        write_json(staging / "sha256sums.json", hashes)
        staging.rename(destination)
    except BaseException:
        # Preserve a failed staging directory as evidence; never silently reuse it.
        raise
    return {
        "path": str(destination.resolve()),
        "model_safetensors_sha256": hashes["model.safetensors"],
        "adaptive_state_sha256": hashes["adaptive_state.json"],
        "optimizer_state_sha256": hashes["optimizer.pt"],
    }


def _identity_diff(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    keys = sorted(set(expected) | set(actual))
    return [key for key in keys if expected.get(key) != actual.get(key)]


def _initial_wenet_seconds(
    arm: dict[str, Any], start_checkpoint: Path
) -> tuple[float, float]:
    """Return total and continuation Wenet exposure at the branch point."""

    cursor = int(arm["wenet_continuation_cursor"])
    parent_state_path = start_checkpoint / "adaptive_state.json"
    if parent_state_path.is_file():
        parent = read_json(parent_state_path)
        parent_cursor = int(parent["wenet_sample_cursor"])
        if "continuation_wenet_seconds" in arm:
            continuation_seconds = float(arm["continuation_wenet_seconds"])
        elif parent_cursor == cursor:
            continuation_seconds = float(parent.get("continuation_wenet_seconds", 0.0))
        else:
            raise SystemExit(
                "A cross-stream branch must provide continuation_wenet_seconds "
                "from its cursor-remap receipt"
            )
        return (
            float(parent["cumulative_wenet_seconds"]),
            continuation_seconds,
        )
    return float(arm["start_hours"]) * 3600.0, float(
        arm.get("continuation_wenet_seconds", 0.0)
    )


def _official_prefix_seconds(
    rows: list[dict[str, Any]], *, cursor: int, seed: int
) -> float:
    if cursor == 0:
        return 0.0
    indices = deterministic_official_indices(len(rows), cursor, seed)
    return sum(row_duration_seconds(rows[index]) for index in indices)


def _take_microbatches(
    iterator: Iterator[dict[str, torch.Tensor]], count: int, source: str
) -> list[dict[str, torch.Tensor]]:
    result: list[dict[str, torch.Tensor]] = []
    for _ in range(count):
        try:
            result.append(next(iterator))
        except StopIteration as error:
            raise RuntimeError(f"{source} manifest exhausted mid-optimizer-step") from error
    return result


def main() -> None:
    # The production dataset stack imports optional training integrations.  Keep
    # it lazy so pure topology/resume tests do not require those server extras.
    from train import ManifestDataset, SpeechSeq2SeqCollator

    args = parse_args()
    if args.arm_index < 0:
        raise SystemExit("--arm-index must be non-negative")
    if args.smoke_cycles < 0:
        raise SystemExit("--smoke-cycles must be non-negative")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit(
            "This entrypoint requires exactly one visible CUDA GPU; "
            f"found {torch.cuda.device_count()}"
        )

    root = args.project_root.resolve()
    config_path = _resolve(root, args.config)
    preparation_path = _resolve(root, args.preparation)
    arm_file_path = _resolve(root, args.arm_file)
    output = _resolve(root, args.output_dir)
    resume = _resolve(root, args.resume) if args.resume is not None else None
    config = read_json(config_path)
    preparation = read_json(preparation_path)
    if not preparation.get("passed"):
        raise SystemExit("Preparation receipt did not pass")
    arm_payload = read_json(arm_file_path)
    arms = arm_payload.get("arms")
    if not isinstance(arms, list) or args.arm_index >= len(arms):
        raise SystemExit(f"Arm index {args.arm_index} is absent from {arm_file_path}")
    arm = dict(arms[args.arm_index])
    required_arm = {
        "name",
        "start_checkpoint",
        "start_hours",
        "target_hours",
        "wenet_manifest",
        "wenet_continuation_cursor",
        "official_sample_cursor",
        "wenet_lr",
        "official_lr",
    }
    missing_arm = sorted(required_arm - arm.keys())
    if missing_arm:
        raise SystemExit(f"Arm lacks required fields: {missing_arm}")

    training = config["training"]
    if list(training["source_cycle"]) != ["wenet", "official"]:
        raise SystemExit("Adaptive trainer requires source_cycle=['wenet', 'official']")
    batch_size = int(training["per_device_batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    effective_batch = batch_size * accumulation
    configured_effective = int(training.get("global_effective_batch_size", effective_batch))
    if effective_batch != configured_effective:
        raise SystemExit(
            f"Effective batch mismatch: runtime={effective_batch} config={configured_effective}"
        )
    if bool(training.get("gradient_checkpointing", False)):
        raise SystemExit("Gradient checkpointing must remain disabled")

    start_checkpoint = _resolve(root, arm["start_checkpoint"])
    wenet_manifest = _resolve(root, arm["wenet_manifest"])
    official_manifest = _resolve(root, config["data"]["official"])
    for path, label in (
        (start_checkpoint, "start checkpoint"),
        (wenet_manifest, "Wenet manifest"),
        (official_manifest, "Official manifest"),
    ):
        if not path.exists():
            raise SystemExit(f"Missing {label}: {path}")
    start_weight_hash = sha256_file(start_checkpoint / "model.safetensors")
    expected_start_hash = arm.get("parent_model_sha256", arm.get("start_model_sha256"))
    if expected_start_hash is None and start_checkpoint == _resolve(root, config["start"]["checkpoint"]):
        expected_start_hash = config["start"]["weight_sha256"]
    if expected_start_hash is not None and start_weight_hash != expected_start_hash:
        raise SystemExit("Starting checkpoint weight hash mismatch")

    identity = {
        "arm_name": str(arm["name"]),
        "arm_index": int(args.arm_index),
        "config_sha256": sha256_file(config_path),
        "preparation_sha256": sha256_file(preparation_path),
        "arm_file_sha256": sha256_file(arm_file_path),
        "trainer_sha256": sha256_file(Path(__file__).resolve()),
        "start_checkpoint": str(start_checkpoint),
        "start_weight_sha256": start_weight_hash,
        "wenet_manifest": str(wenet_manifest),
        "wenet_manifest_sha256": sha256_file(wenet_manifest),
        "official_manifest": str(official_manifest),
        "official_manifest_sha256": sha256_file(official_manifest),
    }

    if output.exists() and resume is None:
        raise SystemExit(f"Refusing to overwrite output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_steps_path = output / "source_steps.jsonl"
    if resume is not None:
        if not resume.is_dir():
            raise SystemExit(f"Resume checkpoint does not exist: {resume}")
        for name in REQUIRED_CHECKPOINT_FILES:
            if not (resume / name).is_file():
                raise SystemExit(f"Resume checkpoint is incomplete: missing {name}")
        resume_state = read_json(resume / "adaptive_state.json")
        differences = _identity_diff(identity, resume_state.get("identity", {}))
        if differences:
            raise SystemExit(f"Resume identity mismatch: {differences}")
        if resume_state.get("next_source") != "wenet":
            raise SystemExit("Resume is not at a complete source-cycle boundary")
        if source_steps_path.exists():
            if sha256_file(source_steps_path) != resume_state["source_steps_sha256"]:
                raise SystemExit("Output source-step log differs from resume checkpoint")
        else:
            shutil.copy2(resume / "source_steps.jsonl", source_steps_path)
    elif source_steps_path.exists():
        raise SystemExit(f"Refusing existing source-step log: {source_steps_path}")
    else:
        source_steps_path.touch()

    set_seed(int(training["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = bool(training["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(training["tf32"])
    device = torch.device("cuda:0")
    model_source = resume if resume is not None else start_checkpoint
    processor_source = Path(arm.get("processor_checkpoint", start_checkpoint))
    processor_source = _resolve(root, processor_source)
    processor = WhisperProcessor.from_pretrained(processor_source, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_source, local_files_only=True
    )
    model.config.apply_spec_augment = False
    model.config.mask_time_prob = 0.0
    model.config.mask_feature_prob = 0.0
    model.config.use_cache = False
    if (
        model.config.apply_spec_augment
        or model.config.mask_time_prob
        or model.config.mask_feature_prob
    ):
        raise SystemExit("SpecAugment disable guard failed")
    model.to(device)
    model.train()
    optimizer, encoder_named, decoder_named, parameter_receipt = create_optimizer(
        model, float(training["weight_decay"])
    )
    if not parameter_receipt["proj_out_tied_to_decoder_embedding"]:
        raise SystemExit("Unexpected Whisper proj_out tying behavior")

    wenet_rows = read_jsonl(wenet_manifest)
    official_rows = read_jsonl(official_manifest)
    initial_wenet_cursor = int(arm["wenet_continuation_cursor"])
    initial_official_cursor = int(arm["official_sample_cursor"])
    initial_total_wenet_seconds, initial_continuation_seconds = _initial_wenet_seconds(
        arm, start_checkpoint
    )
    initial_official_seconds = float(
        arm.get(
            "cumulative_official_seconds",
            _official_prefix_seconds(
                official_rows,
                cursor=initial_official_cursor,
                seed=int(training["data_seed"]),
            ),
        )
    )
    state: dict[str, Any] = {
        "schema_version": 1,
        "arm": str(arm["name"]),
        "identity": identity,
        "optimizer_mode": "fresh_adamw_at_branch",
        "total_optimizer_steps": 0,
        "wenet_optimizer_steps": 0,
        "official_optimizer_steps": 0,
        "wenet_sample_cursor": initial_wenet_cursor,
        "official_sample_cursor": initial_official_cursor,
        "continuation_wenet_seconds": initial_continuation_seconds,
        "cumulative_wenet_seconds": initial_total_wenet_seconds,
        "cumulative_official_seconds": initial_official_seconds,
        "completed_cycles": 0,
        "next_source": "wenet",
        "gradient_audits": {
            "wenet_encoder_only": False,
            "post_official_wenet_decoder_and_moments_unchanged": False,
            "official_full_sft": False,
        },
    }
    if resume is not None:
        state = resume_state
        optimizer.load_state_dict(_load_torch(resume / "optimizer.pt"))

    targets = checkpoint_targets(float(arm["start_hours"]), float(arm["target_hours"]))
    boundaries = milestone_cursors(
        wenet_rows,
        start_cursor=int(state["wenet_sample_cursor"]),
        start_exposure_seconds=float(state["cumulative_wenet_seconds"]),
        targets_hours=targets,
        effective_batch_size=effective_batch,
    )
    final_cursor = max(
        (int(value["cursor"]) for value in boundaries.values()),
        default=int(state["wenet_sample_cursor"]),
    )
    if args.smoke_cycles:
        final_cursor = min(
            final_cursor,
            int(state["wenet_sample_cursor"]) + args.smoke_cycles * effective_batch,
        )
    remaining_cycles = (final_cursor - int(state["wenet_sample_cursor"])) // effective_batch
    official_needed = int(state["official_sample_cursor"]) + remaining_cycles * effective_batch
    official_indices = deterministic_official_indices(
        len(official_rows), official_needed, int(training["data_seed"])
    )[int(state["official_sample_cursor"]) :]
    wenet_indices = list(range(int(state["wenet_sample_cursor"]), final_cursor))

    collator = SpeechSeq2SeqCollator(processor)
    wenet_loader = DataLoader(
        ManifestDataset(wenet_manifest, processor, root),
        batch_size=batch_size,
        sampler=FixedSequenceSampler(wenet_indices),
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    official_loader = DataLoader(
        ManifestDataset(official_manifest, processor, root),
        batch_size=batch_size,
        sampler=FixedSequenceSampler(official_indices),
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    wenet_iterator = iter(wenet_loader)
    official_iterator = iter(official_loader)
    if resume is not None:
        restore_rng_state(_load_torch(resume / "rng_state.pth"))

    run_receipt = {
        "schema_version": 1,
        "identity": identity,
        "arm": arm,
        "training": training,
        "milestone_boundaries": {str(key): value for key, value in boundaries.items()},
        "optimizer_reinitialized_at_new_branch": resume is None,
        "same_arm_resume": resume is not None,
        "parameter_receipt": parameter_receipt,
        "environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
    }
    run_config_path = output / "run_config.json"
    if run_config_path.exists():
        if read_json(run_config_path)["identity"] != identity:
            raise SystemExit("Existing output run identity mismatch")
    else:
        write_json(run_config_path, run_receipt)
    processor.save_pretrained(output)

    completion: dict[str, Any] = {
        "schema_version": 1,
        "passed": False,
        "arm": str(arm["name"]),
        "checkpoints": {},
        "smoke_cycles": int(args.smoke_cycles),
    }
    saved_audits = state.get("gradient_audits", {})
    first_wenet_audit = bool(saved_audits.get("wenet_encoder_only", False))
    post_official_wenet_audit = bool(
        saved_audits.get("post_official_wenet_decoder_and_moments_unchanged", False)
    )
    first_official_audit = bool(saved_audits.get("official_full_sft", False))
    started = time.monotonic()

    def source_step(source: str) -> dict[str, Any]:
        nonlocal first_wenet_audit, post_official_wenet_audit, first_official_audit
        iterator = wenet_iterator if source == "wenet" else official_iterator
        cursor_key = f"{source}_sample_cursor"
        cursor = int(state[cursor_key])
        rows = wenet_rows if source == "wenet" else official_rows
        if source == "wenet":
            indices = list(range(cursor, cursor + effective_batch))
        else:
            indices = deterministic_official_indices(
                len(official_rows), cursor + effective_batch, int(training["data_seed"])
            )[cursor:]
        sample_rows = [rows[index] for index in indices]
        sample_ids = [row_identifier(row, index) for index, row in zip(indices, sample_rows)]
        step_seconds = sum(row_duration_seconds(row) for row in sample_rows)

        audit_wenet = source == "wenet" and (
            not first_wenet_audit
            or (int(state["official_optimizer_steps"]) > 0 and not post_official_wenet_audit)
        )
        decoder_before = tensor_digest(decoder_named) if audit_wenet else None
        optimizer_before = (
            optimizer_state_digest(optimizer, decoder_named) if audit_wenet else None
        )
        batches = _take_microbatches(iterator, accumulation, source)
        result = perform_optimizer_step(
            source=source,
            model=model,
            optimizer=optimizer,
            batches=batches,
            device=device,
            encoder_named=encoder_named,
            decoder_named=decoder_named,
            wenet_lr=float(arm["wenet_lr"]),
            official_lr=float(arm["official_lr"]),
            gradient_accumulation_steps=accumulation,
            max_grad_norm=float(training["max_grad_norm"]),
        )
        if audit_wenet:
            if tensor_digest(decoder_named) != decoder_before:
                raise RuntimeError("Decoder weights changed during Wenet Encoder-only step")
            if optimizer_state_digest(optimizer, decoder_named) != optimizer_before:
                raise RuntimeError("Decoder optimizer moments changed during Wenet step")
            if not first_wenet_audit:
                first_wenet_audit = True
            else:
                post_official_wenet_audit = True
        if source == "official":
            first_official_audit = True
        state["gradient_audits"] = {
            "wenet_encoder_only": first_wenet_audit,
            "post_official_wenet_decoder_and_moments_unchanged": post_official_wenet_audit,
            "official_full_sft": first_official_audit,
        }

        state["total_optimizer_steps"] = int(state["total_optimizer_steps"]) + 1
        state[f"{source}_optimizer_steps"] = int(state[f"{source}_optimizer_steps"]) + 1
        state[cursor_key] = cursor + effective_batch
        state[f"cumulative_{source}_seconds"] = (
            float(state[f"cumulative_{source}_seconds"]) + step_seconds
        )
        if source == "wenet":
            state["continuation_wenet_seconds"] = (
                float(state["continuation_wenet_seconds"]) + step_seconds
            )
        record = {
            "event": "optimizer_step",
            "source": source,
            "source_homogeneous_accumulation": True,
            "total_optimizer_step": int(state["total_optimizer_steps"]),
            "wenet_optimizer_step": int(state["wenet_optimizer_steps"]),
            "official_optimizer_step": int(state["official_optimizer_steps"]),
            "sample_cursor_after": int(state[cursor_key]),
            "sample_ids": sample_ids,
            "samples": len(sample_ids),
            "step_seconds": step_seconds,
            "cumulative_wenet_seconds": float(state["cumulative_wenet_seconds"]),
            "cumulative_official_seconds": float(state["cumulative_official_seconds"]),
            "wenet_lr": float(arm["wenet_lr"]) if source == "wenet" else float(arm["official_lr"]),
            "decoder_lr": 0.0 if source == "wenet" else float(arm["official_lr"]),
            "loss": result["loss"],
            "grad_norm": result["grad_norm"],
            "elapsed_seconds": time.monotonic() - started,
        }
        append_jsonl(source_steps_path, record)
        return record

    checkpoint_by_cursor = {
        int(values["cursor"]): (target, values) for target, values in boundaries.items()
    }
    while int(state["wenet_sample_cursor"]) < final_cursor:
        source_step("wenet")
        source_step("official")
        state["completed_cycles"] = int(state["completed_cycles"]) + 1
        state["next_source"] = "wenet"
        current_cursor = int(state["wenet_sample_cursor"])
        if args.smoke_cycles or current_cursor not in checkpoint_by_cursor:
            continue
        target, boundary = checkpoint_by_cursor[current_cursor]
        state["wenet_hours_target"] = float(target)
        state["actual_wenet_hours"] = float(state["cumulative_wenet_seconds"]) / 3600.0
        state["target_deviation_seconds"] = (
            float(state["cumulative_wenet_seconds"]) - float(target) * 3600.0
        )
        label = f"checkpoint-wenet-{target:g}h"
        completion["checkpoints"][str(target)] = save_checkpoint(
            output_dir=output,
            label=label,
            model=model,
            processor=processor,
            optimizer=optimizer,
            state=state,
            source_steps_path=source_steps_path,
        )
        if abs(float(boundary["actual_seconds"]) - float(state["cumulative_wenet_seconds"])) > 1e-6:
            raise RuntimeError("Runtime Wenet exposure differs from frozen milestone boundary")

    if args.smoke_cycles and int(state["completed_cycles"]) > 0:
        completion["checkpoints"]["smoke"] = save_checkpoint(
            output_dir=output,
            label=f"checkpoint-smoke-{int(state['completed_cycles'])}-cycles",
            model=model,
            processor=processor,
            optimizer=optimizer,
            state=state,
            source_steps_path=source_steps_path,
        )

    configure_source(model, source="official")
    completion.update(
        {
            "passed": (
                first_wenet_audit
                and first_official_audit
                and (post_official_wenet_audit or int(state["completed_cycles"]) == 1)
                and int(state["wenet_sample_cursor"]) == final_cursor
            ),
            "final_state": state,
            "gradient_audits": {
                "wenet_encoder_only": first_wenet_audit,
                "post_official_wenet_decoder_and_moments_unchanged": post_official_wenet_audit,
                "official_full_sft": first_official_audit,
            },
            "source_steps_sha256": sha256_file(source_steps_path),
        }
    )
    write_json(output / "training_completion.json", completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2))
    if not completion["passed"]:
        raise SystemExit(10)


if __name__ == "__main__":
    main()
