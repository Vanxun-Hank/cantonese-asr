#!/usr/bin/env python3
"""One resumable P2 training segment; external evaluation runs after exit.

Use torchrun for multi-GPU arms.  No automatic Slurm submission or retries.
The formal path requires a separately verified GPU acceptance receipt.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import partial
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, FullStateDictConfig,
                                   FullOptimStateDictConfig, StateDictType, MixedPrecision,
                                   ShardingStrategy)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import WhisperProcessor, set_seed
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer, WhisperDecoderLayer

from cantonese_asr.io import sha256_file
from cantonese_asr.model_loading import load_whisper_model
from cantonese_asr.p2_full import atomic_json, cosine_factor, rank_microbatches, code_fingerprint
from cantonese_asr.training_sampling import source_group
from train import ManifestDataset, SpeechSeq2SeqCollator, configure_lora



def apply_source_mode(model, mode: str) -> None:
    """W500 source-conditional topology: external audio may only move the Encoder.

    W500_ADAPTIVE_METHOD.md: updating the whole model on external batches teaches
    useful acoustics but also changes Cantonese word choice, insertion behaviour and
    end-of-sequence behaviour. So the Decoder and proj_out are frozen on external
    steps and trainable on official steps.

    proj_out.weight is tied to decoder.embed_tokens.weight, so they freeze together.
    Toggling requires_grad is sufficient and safe with AdamW: parameters whose .grad
    is None are skipped by step(), leaving their Adam moments and weight decay
    untouched -- which is exactly the "Encoder moments remain continuous" property
    the W500 recipe relies on.
    """
    if mode not in {"official", "external"}:
        raise ValueError(f"unknown source_mode {mode!r}")
    core = getattr(model, "module", model)
    decoder_enabled = mode == "official"
    for parameter in core.model.encoder.parameters():
        parameter.requires_grad_(True)
    for parameter in core.model.decoder.parameters():
        parameter.requires_grad_(decoder_enabled)
    for parameter in core.proj_out.parameters():
        parameter.requires_grad_(decoder_enabled)


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--asset-root", type=Path, required=True)
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--seed", type=int, choices=(42, 43), required=True)
    p.add_argument("--until-step", type=int, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--phase", choices=("smoke", "benchmark", "train"), required=True)
    p.add_argument("--acceptance", type=Path)
    p.add_argument("--extension-receipt", type=Path)
    p.add_argument("--batch", type=int, help="Only approved OOM reduction; global batch stays 16")
    return p


def initialize_checkpoint_paths(checkpoint, staging, rank, world, broadcast=None):
    """Rank zero owns exclusive creation; all ranks receive the same outcome."""
    status = [None]
    if rank == 0:
        try:
            if checkpoint.exists() or staging.exists():
                raise FileExistsError("checkpoint already exists; preserve evidence and use a new attempt")
            staging.mkdir()
        except OSError as error:
            status[0] = f"{type(error).__name__}: {error}"
    if world > 1:
        (broadcast or dist.broadcast_object_list)(status, src=0)
    if status[0] is not None:
        raise RuntimeError(f"collective checkpoint initialization failed: {status[0]}")


def main():
    a = args_parser().parse_args()
    # Required by the GPU reproducibility diagnostic (1995/1996). Configure
    # before any CUDA context is initialized; do not silently downgrade errors.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cfg = json.loads(a.config.read_text())
    arm = cfg["arms"][a.arm]
    prep = json.loads((a.preparation / "preflight.json").read_text())
    if prep["status"] != "PASS" or prep["config_sha256"] != sha256_file(a.config):
        raise ValueError("preparation/config mismatch")
    if a.phase == "train":
        if a.acceptance is None:
            raise ValueError("formal training requires GPU acceptance receipt")
        gate = json.loads(a.acceptance.read_text())
        if (gate.get("status") != "PASS" or gate.get("config_sha256") != sha256_file(a.config)
                or gate.get("code_sha256") != code_fingerprint(ROOT)
                or a.arm not in gate.get("accepted_arms", [])):
            raise ValueError("arm not accepted for formal training")
        if a.until_step > 4371:
            if a.extension_receipt is None:
                raise ValueError("extension requires paired validation decision")
            decision = json.loads(a.extension_receipt.read_text())
            if decision.get("arm") != a.arm or decision.get("extend_both_seeds") is not True:
                raise ValueError("extension not authorized by evidence")
    elif a.until_step > (6 if a.phase == "smoke" else 50):
        raise ValueError("smoke/benchmark step limit exceeded")
    if not 1 <= a.until_step <= 7285:
        raise ValueError("invalid stop step")
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world != arm["world_size"] or not torch.cuda.is_available():
        raise ValueError("actual topology must match registered CUDA arm")
    batch = a.batch or arm["batch"]
    if batch < 1 or arm["batch"] % batch or (arm["batch"] // batch) & ((arm["batch"] // batch) - 1):
        raise ValueError("only power-of-two batch reductions permitted")
    if 16 % (batch * world) or 8 % (batch * world):
        raise ValueError("invalid global/tail batch topology")
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl")
    device = torch.device("cuda", local)
    set_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    manifest = a.asset_root / cfg["external73"]["path"]
    if sha256_file(manifest) != cfg["external73"]["sha256"]:
        raise ValueError("training manifest changed")
    source_model = a.asset_root / cfg["models"][arm["model"]]
    model_receipt_path = a.preparation / "model_hashes.json"
    if sha256_file(model_receipt_path) != prep["model_hashes_sha256"]:
        raise ValueError("model provenance receipt changed")
    model_receipt = json.loads(model_receipt_path.read_text())[arm["model"]]
    for item in model_receipt["files"]:
        if sha256_file(source_model / item["path"]) != item["sha256"]:
            raise ValueError("original model asset changed after freeze")
    processor = WhisperProcessor.from_pretrained(source_model, language="zh", task="transcribe", local_files_only=True)
    previous = None
    if a.resume:
        previous = json.loads((a.resume / "COMPLETE.json").read_text())
        if previous["arm"] != a.arm or previous["seed"] != a.seed or previous["world_size"] != world:
            raise ValueError("resume identity/topology mismatch")
        if previous["config_sha256"] != sha256_file(a.config):
            raise ValueError("resume config changed")
        if previous.get("code_sha256") != code_fingerprint(ROOT):
            raise ValueError("resume implementation changed; do not mix code versions")
        for item in previous["files"]:
            if sha256_file(a.resume / item["path"]) != item["sha256"]:
                raise ValueError(f"checkpoint corrupt: {item['path']}")
    start = previous["step"] if previous else 0
    if start >= a.until_step:
        raise ValueError("resume does not precede stop step")
    model = load_whisper_model(a.resume / "model" if a.resume else source_model,
                               dtype=torch.float32, adapter_trainable=True)
    model.config.apply_spec_augment = False
    model.config.mask_time_prob = 0.0
    model.config.mask_feature_prob = 0.0
    model.config.use_cache = False
    model.generation_config.language = "zh"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    if arm["method"] == "lora" and not a.resume:
        model, _ = configure_lora(model, argparse.Namespace(lora=True,
            lora_target_scope="all_attention_qv", lora_target_modules="", lora_r=8,
            lora_alpha=16, lora_dropout=.05))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    backbone = model.get_base_model() if arm["method"] == "lora" else model
    whisper = backbone.model
    original_mask = whisper._mask_input_features
    mask_observations = {"calls": 0, "unchanged": 0}
    def verify_no_spec(input_features, attention_mask=None):
        if whisper.config.apply_spec_augment or whisper.config.mask_time_prob or whisper.config.mask_feature_prob:
            raise ValueError("SpecAugment unexpectedly enabled in forward")
        result = original_mask(input_features, attention_mask=attention_mask)
        mask_observations["calls"] += 1
        if result is not input_features and not torch.equal(result, input_features):
            raise ValueError("input features changed by SpecAugment path")
        mask_observations["unchanged"] += 1
        return result
    whisper._mask_input_features = verify_no_spec
    trainable = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    if arm["method"] == "lora" and any(".lora_" not in n for n in trainable):
        raise ValueError("non-adapter trainable parameter")
    dataset = ManifestDataset(manifest, processor, a.asset_root, seed=a.seed, epochs=5)
    collate = SpeechSeq2SeqCollator(processor)
    frozen_stream = a.preparation / f"{'smoke' if a.phase == 'smoke' else 'stream'}_s{a.seed}.json"
    expected_stream = prep["streams"][str(a.seed)]["smoke_sha256" if a.phase == "smoke" else "sha256"]
    if sha256_file(frozen_stream) != expected_stream:
        raise ValueError("frozen sample stream changed")
    stream = json.loads(frozen_stream.read_text())
    if a.phase == "smoke":
        # Exercise a real tail update and resume across it; never change the
        # formal stream. The immutable 96-row diagnostic pool supplies 88 draws.
        if len(stream) != 6 or len(stream[-1]["indices"]) != 16:
            raise ValueError("unexpected smoke pool layout")
        stream[-1]["indices"] = stream[-1]["indices"][:8]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = a.output_dir / f"checkpoint-{a.until_step}"
    staging = a.output_dir / f"checkpoint-{a.until_step}.staging"
    initialize_checkpoint_paths(checkpoint, staging, rank, world)
    if rank == 0:
        atomic_json(a.output_dir / f"run_config_segment_{start}_{a.until_step}.json", {
            "arm": a.arm, "seed": a.seed, "config": cfg, "actual_batch": batch,
            "world_size": world, "start_step": start, "until_step": a.until_step,
            "phase": a.phase, "trainable": trainable, "code_sha256": code_fingerprint(ROOT),
            "trainable_parameters": sum(trainable.values()),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "config_sha256": sha256_file(a.config), "stream_sha256": expected_stream})
    if world > 1:
        dist.barrier()
        model = FSDP(model, device_id=device, use_orig_params=True,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=partial(transformer_auto_wrap_policy,
                transformer_layer_cls={WhisperEncoderLayer, WhisperDecoderLayer}),
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                           buffer_dtype=torch.bfloat16))
    else:
        model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=arm["lr"], weight_decay=.01)
    if a.resume:
        osd = torch.load(a.resume / "optimizer.pt", map_location="cpu", weights_only=False)
        if world > 1:
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                osd = FSDP.optim_state_dict_to_load(model, optimizer, osd)
        optimizer.load_state_dict(osd)
        del osd
        rng = torch.load(a.resume / f"rng_rank{rank}.pt", map_location="cpu", weights_only=False)
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"], device)
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    processed = 0
    for entry in stream[start:a.until_step]:
        step = entry["step"]
        if "source_mode" in entry:
            # Source-conditional stream. Toggling trainability under FSDP is not
            # sound, but this variant runs world_size 1 so there is no wrapper.
            if world > 1:
                raise ValueError("source-conditional freezing requires world_size 1")
            apply_source_mode(model, entry["source_mode"])
        partitions = rank_microbatches(entry["indices"], world, batch)
        microbatches = [collate([dataset[i] for i in micro[rank]]) for micro in partitions]
        observed = [m["_training_row_index"].tolist() for m in microbatches]
        if observed != [micro[rank] for micro in partitions]:
            raise ValueError("actual collated sample IDs differ from frozen topology")
        with (a.output_dir / f"observed_rank{rank}_segment_{start}_{a.until_step}.jsonl").open("a") as handle:
            handle.write(json.dumps({"step": step, "rank": rank, "microbatch_indices": observed}) + "\n")
        valid = sum(int(m["labels"].ne(-100).sum()) for m in microbatches)
        valid_tensor = torch.tensor(valid, device=device, dtype=torch.long)
        if world > 1:
            dist.all_reduce(valid_tensor)
        total_tokens = int(valid_tensor.item())
        if total_tokens == 0:
            raise ValueError("empty global labels")
        learning_rate = arm["lr"] * cosine_factor(step - 1)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        local_loss_sum = torch.zeros((), device=device)
        for micro in microbatches:
            tensors = {k: v.to(device) for k, v in micro.items() if not k.startswith("_")}
            if tensors["labels"].shape[1] > model.config.max_target_positions:
                raise ValueError("label exceeds decoder capacity; no silent truncation")
            if not torch.isfinite(tensors["input_features"]).all():
                raise ValueError("nonfinite input features")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(**tensors)
                count = tensors["labels"].ne(-100).sum()
                # FSDP averages gradients across ranks; compensate once. No
                # accumulation divisor: the denominator is the global token count.
                loss = result.loss * count * world / total_tokens
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            loss.backward()
            local_loss_sum += result.loss.detach().float() * count
        norm = model.clip_grad_norm_(1.0) if world > 1 else torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite gradient norm")
        optimizer.step()
        if world > 1:
            dist.all_reduce(local_loss_sum)
        processed += len(entry["indices"])
        if rank == 0:
            record = {**entry, "sample_ids": [prep["sample_ids"][i] for i in entry["indices"]],
                "sources": dict(Counter(source_group(dataset.rows[i]) for i in entry["indices"])),
                    "source_mode": entry.get("source_mode"),
                "global_samples": len(entry["indices"]), "valid_tokens": total_tokens,
                "loss": float(local_loss_sum / total_tokens), "grad_norm": float(norm),
                "lr": learning_rate, "elapsed_seconds": time.perf_counter() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
            with (a.output_dir / f"training_segment_{start}_{a.until_step}.jsonl").open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            if step % 25 == 0 or step == a.until_step:
                print(json.dumps({k: record[k] for k in ("step", "loss", "lr", "elapsed_seconds")}), flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    atomic_json(staging / f"resources_rank{rank}.json", {"rank": rank,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "device_name": torch.cuda.get_device_name(device), "elapsed_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "specaugment_forward_observations": mask_observations})
    torch.save({"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device)},
               staging / f"rng_rank{rank}.pt")
    if world > 1:
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
        base = model.module
    else:
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        optim_state = optimizer.state_dict()
        base = model
    if rank == 0:
        base.save_pretrained(staging / "model", state_dict=state, safe_serialization=True,
                             max_shard_size="4GB")
        processor.save_pretrained(staging / "model")
        torch.save(optim_state, staging / "optimizer.pt")
        atomic_json(staging / "scheduler.json", {"last_completed_step": a.until_step,
                    "horizon": 7285, "warmup_steps": 365, "next_lr": arm["lr"] * cosine_factor(a.until_step)})
    if world > 1:
        dist.barrier()
    if rank == 0:
        files = [{"path": str(p.relative_to(staging)), "sha256": sha256_file(p), "bytes": p.stat().st_size}
                 for p in sorted(staging.rglob("*")) if p.is_file()]
        atomic_json(staging / "COMPLETE.json", {"arm": a.arm, "seed": a.seed,
            "step": a.until_step, "world_size": world, "config_sha256": sha256_file(a.config),
            "stream_sha256": expected_stream, "files": files, "phase": a.phase, "code_sha256": code_fingerprint(ROOT),
            "next_stream_position": a.until_step, "segment_seconds": elapsed,
            "segment_samples": processed, "samples_per_second": processed / elapsed,
            "optimizer_steps_per_second": (a.until_step - start) / elapsed,
            "training_gpu_hours": elapsed * world / 3600})
        staging.rename(checkpoint)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
