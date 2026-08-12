#!/usr/bin/env python3
"""Reproducible full fine-tuning entry point for Cantonese Whisper-small."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from peft import LoraConfig, TaskType
from scipy.signal import butter, resample_poly, sosfilt
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    set_seed,
)

from cantonese_asr.io import read_jsonl, sha256_file, sha256_text_files
from cantonese_asr.decoding_guard import detect_runaway
from cantonese_asr.metrics import compute_diagnostic_metrics, compute_official_metrics
from cantonese_asr.model_loading import WhisperPeftModelForSeq2SeqLM
from cantonese_asr.training_sampling import (
    EpochEncodedSampler,
    FixedExposureSampler,
    build_speed_assignments,
    exp004_wenet_source_group,
    source_group,
    speed_assignment_report,
    stable_seed,
)


def runtime_training_world_size() -> int:
    """Return the global data-parallel width used for effective-batch checks."""
    distributed_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if distributed_world_size > 1:
        return distributed_world_size
    if torch.cuda.is_available():
        return max(1, int(torch.cuda.device_count()))
    return 1


def training_source_code(row: dict[str, Any]) -> int:
    """Encode auditable training sources without changing manifest rows."""
    try:
        group = exp004_wenet_source_group(row)
    except ValueError:
        return -1
    return {
        "official": 0,
        "wenet": 1,
        "cv": 2,
        "mdcc": 3,
    }[group]


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        processor: WhisperProcessor,
        project_root: Path,
        max_samples: int | None = None,
        official_mix_manifest: Path | None = None,
        wenet_mix_manifest: Path | None = None,
        sampling_policy: str = "uniform",
        speed_manifest: Path | None = None,
        seed: int = 42,
        epochs: int = 3,
        sampling_receipt_dir: Path | None = None,
        online_speed_perturbation: bool = False,
        online_gaussian_noise: bool = False,
        online_gaussian_noise_probability: float = 1.0,
        noise_snr_min_db: float = 15.0,
        noise_snr_max_db: float = 25.0,
        online_channel_bandlimit: bool = False,
        channel_bandlimit_probability: float = 0.5,
    ) -> None:
        self.processor = processor
        self.project_root = project_root
        self.rows = read_jsonl(manifest)
        if official_mix_manifest is not None:
            self.rows.extend(read_jsonl(official_mix_manifest))
        if wenet_mix_manifest is not None:
            self.rows.extend(read_jsonl(wenet_mix_manifest))
        if max_samples is not None:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise ValueError(f"Empty manifest: {manifest}")
        self.sampling_policy = sampling_policy
        self.seed = int(seed)
        self.online_speed_perturbation = bool(online_speed_perturbation)
        self.online_gaussian_noise = bool(online_gaussian_noise)
        self.online_gaussian_noise_probability = float(
            online_gaussian_noise_probability
        )
        self.noise_snr_min_db = float(noise_snr_min_db)
        self.noise_snr_max_db = float(noise_snr_max_db)
        self.online_channel_bandlimit = bool(online_channel_bandlimit)
        self.channel_bandlimit_probability = float(channel_bandlimit_probability)
        self._augmentation_rngs: dict[str, np.random.Generator] = {}
        self._augmentation_worker_seed: int | None = None
        self.speed_views: dict[str, dict[str, str]] = {}
        self.speed_assignments: list[dict[int, str]] = []
        if sampling_policy == "official_speed":
            if speed_manifest is None:
                raise ValueError("official_speed policy requires --official-speed-manifest")
            for speed_row in read_jsonl(speed_manifest):
                parent_id = str(speed_row.get("parent_id", "")).strip()
                factor = f"{float(speed_row.get('speed_factor')):.1f}"
                audio_path = str(speed_row.get("audio_path", "")).strip()
                if not parent_id or factor not in {"0.9", "1.0", "1.1"} or not audio_path:
                    raise ValueError(f"Invalid speed-view row: {speed_row}")
                self.speed_views.setdefault(parent_id, {})[factor] = audio_path
            official_ids = {
                str(row["id"])
                for row in self.rows
                if source_group(row) == "official"
            }
            missing = sorted(
                parent
                for parent in official_ids
                if set(self.speed_views.get(parent, {})) != {"0.9", "1.0", "1.1"}
            )
            if missing:
                raise ValueError(f"Missing complete speed views for Official rows: {missing[:5]}")
            self.speed_assignments = build_speed_assignments(
                self.rows, seed=seed, epochs=epochs
            )
            if sampling_receipt_dir is not None:
                sampling_receipt_dir.mkdir(parents=True, exist_ok=True)
                (sampling_receipt_dir / "speed_assignments.json").write_text(
                    json.dumps(
                        speed_assignment_report(self.speed_assignments),
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )

    def __len__(self) -> int:
        return len(self.rows)

    def _augmentation_rng(self, stream: str) -> np.random.Generator:
        """Return a deterministic, worker-local RNG for one augmentation stream."""
        worker_seed = int(torch.initial_seed())
        if self._augmentation_worker_seed != worker_seed:
            self._augmentation_rngs = {}
            self._augmentation_worker_seed = worker_seed
        if stream not in self._augmentation_rngs:
            self._augmentation_rngs[stream] = np.random.default_rng(
                stable_seed(self.seed, worker_seed, stream)
            )
        return self._augmentation_rngs[stream]

    def _apply_online_augmentation(
        self,
        audio: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, float, bool]:
        audio = np.asarray(audio, dtype=np.float32)
        speed_factor = 1.0
        if self.online_speed_perturbation:
            speed_factor = float(
                self._augmentation_rng("speed").choice((0.9, 1.0, 1.1))
            )
            if speed_factor != 1.0:
                audio = librosa.effects.time_stretch(
                    np.asarray(audio, dtype=np.float32),
                    rate=speed_factor,
                )

        noise_applied = False
        snr_db = float("nan")
        if self.online_gaussian_noise:
            noise_rng = self._augmentation_rng("noise")
            if noise_rng.random() < self.online_gaussian_noise_probability:
                snr_db = float(
                    noise_rng.uniform(self.noise_snr_min_db, self.noise_snr_max_db)
                )
                signal_rms = float(
                    np.sqrt(np.mean(np.square(audio, dtype=np.float64)))
                )
                if signal_rms > 1e-8:
                    noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
                    noise = noise_rng.normal(
                        loc=0.0,
                        scale=noise_rms,
                        size=audio.shape,
                    ).astype(np.float32)
                    audio = audio + noise
                    noise_applied = True

        channel_applied = False
        if self.online_channel_bandlimit:
            channel_rng = self._augmentation_rng("channel")
            if channel_rng.random() < self.channel_bandlimit_probability:
                original_length = len(audio)
                bandpass = butter(
                    4,
                    (300.0, 3400.0),
                    btype="bandpass",
                    fs=16_000,
                    output="sos",
                )
                audio = sosfilt(bandpass, audio).astype(np.float32)
                audio = resample_poly(audio, up=1, down=2).astype(np.float32)
                audio = resample_poly(audio, up=2, down=1).astype(np.float32)
                if len(audio) < original_length:
                    audio = np.pad(audio, (0, original_length - len(audio)))
                audio = np.asarray(audio[:original_length], dtype=np.float32)
                channel_applied = True
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        return (
            audio.astype(np.float32, copy=False),
            speed_factor,
            noise_applied,
            snr_db,
            channel_applied,
        )

    def __getitem__(self, encoded_index: int) -> dict[str, Any]:
        epoch, index = divmod(int(encoded_index), len(self.rows))
        row = self.rows[index]
        audio_path = Path(str(row["audio_path"]))
        if self.sampling_policy == "official_speed" and source_group(row) == "official":
            if epoch >= len(self.speed_assignments):
                raise IndexError(f"No speed assignment for epoch {epoch}")
            factor = self.speed_assignments[epoch][index]
            audio_path = Path(self.speed_views[str(row["id"])][factor])
        if not audio_path.is_absolute():
            audio_path = self.project_root / audio_path
        audio, _ = librosa.load(audio_path, sr=16_000, mono=True)
        (
            audio,
            speed_factor,
            noise_applied,
            snr_db,
            channel_applied,
        ) = self._apply_online_augmentation(audio)
        processed_audio = self.processor.feature_extractor(
            audio,
            sampling_rate=16_000,
            return_attention_mask=True,
        )
        input_features = processed_audio.input_features[0]
        attention_mask = processed_audio.attention_mask[0]
        labels = self.processor.tokenizer(str(row["text"])).input_ids
        return {
            "input_features": input_features,
            "attention_mask": attention_mask,
            "labels": labels,
            "_augmentation_speed_factor": speed_factor,
            "_augmentation_noise_applied": noise_applied,
            "_augmentation_snr_db": snr_db,
            "_augmentation_channel_applied": channel_applied,
            "_training_source_code": training_source_code(row),
            "_training_row_index": int(index),
        }


class ControlledSamplingTrainer(Seq2SeqTrainer):
    """Use an explicit epoch-aware sampler only for preregistered policies."""

    augmentation_metadata_keys = (
        "_augmentation_speed_factor",
        "_augmentation_noise_applied",
        "_augmentation_snr_db",
        "_augmentation_channel_applied",
        "_training_source_code",
        "_training_row_index",
    )

    def __init__(
        self,
        *args: Any,
        sampling_policy: str = "uniform",
        sampling_seed: int = 42,
        sampling_batch_size: int = 8,
        sampling_receipt_dir: Path | None = None,
        sampling_wenet_ratio: float = 0.0,
        sampling_stream_length: int = 25_600,
        sampling_window_length: int = 1_600,
        sampling_wenet_per_window: int | None = None,
        sampling_wenet_fixed_manifest_order: bool = False,
        scheduler_total_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.controlled_sampling_policy = sampling_policy
        self.controlled_sampling_seed = sampling_seed
        self.controlled_sampling_batch_size = sampling_batch_size
        self.controlled_sampling_receipt_dir = sampling_receipt_dir
        self.controlled_sampling_wenet_ratio = float(sampling_wenet_ratio)
        self.controlled_sampling_stream_length = int(sampling_stream_length)
        self.controlled_sampling_window_length = int(sampling_window_length)
        self.controlled_sampling_wenet_per_window = (
            int(sampling_wenet_per_window)
            if sampling_wenet_per_window is not None
            else None
        )
        self.controlled_sampling_wenet_fixed_manifest_order = bool(
            sampling_wenet_fixed_manifest_order
        )
        self.fixed_scheduler_total_steps = scheduler_total_steps
        self.scheduler_steps_requested_by_trainer: int | None = None
        self.augmentation_receipt: dict[str, Any] = {
            "training_microbatches": 0,
            "epochs": {},
        }
        self.source_exposure: dict[str, Any] = {
            "training_microbatches": 0,
            "total_samples": 0,
            "source_counts": {
                "official": 0,
                "cv": 0,
                "mdcc": 0,
                "wenet": 0,
            },
            "windows": {},
        }
        self.fixed_exposure_microbatch = 0
        super().__init__(*args, **kwargs)
        rows = getattr(self.train_dataset, "rows", [])
        self.source_pool_sizes = {
            source: sum(training_source_code(row) == code for row in rows)
            for source, code in {
                "official": 0,
                "wenet": 1,
                "cv": 2,
                "mdcc": 3,
            }.items()
        }

    def _record_augmentation_batch(
        self,
        speed_factors: torch.Tensor,
        noise_applied: torch.Tensor,
        snr_db: torch.Tensor,
        channel_applied: torch.Tensor,
    ) -> None:
        epoch_index = int(float(self.state.epoch or 0.0)) + 1
        epoch = self.augmentation_receipt["epochs"].setdefault(
            str(epoch_index),
            {
                "samples": 0,
                "speed_factor_counts": {"0.9": 0, "1.0": 0, "1.1": 0},
                "noise_applied": 0,
                "noise_skipped_silence": 0,
                "channel_applied": 0,
                "snr_db_count": 0,
                "snr_db_sum": 0.0,
                "snr_db_min": None,
                "snr_db_max": None,
                "snr_db_bins": {
                    "15-17": 0,
                    "17-19": 0,
                    "19-21": 0,
                    "21-23": 0,
                    "23-25": 0,
                },
            },
        )
        factors = speed_factors.detach().float().cpu().tolist()
        applied = noise_applied.detach().bool().cpu().tolist()
        snrs = snr_db.detach().float().cpu().tolist()
        channels = channel_applied.detach().bool().cpu().tolist()
        for factor, did_apply, snr, did_channel in zip(
            factors, applied, snrs, channels
        ):
            factor_key = f"{float(factor):.1f}"
            if factor_key not in epoch["speed_factor_counts"]:
                raise ValueError(f"Unexpected online speed factor: {factor}")
            epoch["samples"] += 1
            epoch["speed_factor_counts"][factor_key] += 1
            if np.isfinite(snr):
                epoch["snr_db_count"] += 1
                epoch["snr_db_sum"] += float(snr)
                epoch["snr_db_min"] = (
                    float(snr)
                    if epoch["snr_db_min"] is None
                    else min(epoch["snr_db_min"], float(snr))
                )
                epoch["snr_db_max"] = (
                    float(snr)
                    if epoch["snr_db_max"] is None
                    else max(epoch["snr_db_max"], float(snr))
                )
                bin_index = min(4, max(0, int((float(snr) - 15.0) // 2.0)))
                bin_key = tuple(epoch["snr_db_bins"])[bin_index]
                epoch["snr_db_bins"][bin_key] += 1
                if did_apply:
                    epoch["noise_applied"] += 1
                else:
                    epoch["noise_skipped_silence"] += 1
            if did_channel:
                epoch["channel_applied"] += 1
        self.augmentation_receipt["training_microbatches"] += 1
        if self.augmentation_receipt["training_microbatches"] % 100 == 0:
            self.write_augmentation_receipt()

    def write_augmentation_receipt(self) -> None:
        if self.controlled_sampling_receipt_dir is None:
            return
        self.controlled_sampling_receipt_dir.mkdir(parents=True, exist_ok=True)
        payload = json.loads(json.dumps(self.augmentation_receipt))
        for epoch in payload["epochs"].values():
            count = int(epoch.pop("snr_db_count"))
            total = float(epoch.pop("snr_db_sum"))
            epoch["snr_db_mean"] = total / count if count else None
            epoch["noise_trigger_rate"] = (
                epoch["noise_applied"] / epoch["samples"]
                if epoch["samples"]
                else 0.0
            )
            epoch["channel_trigger_rate"] = (
                epoch["channel_applied"] / epoch["samples"]
                if epoch["samples"]
                else 0.0
            )
        (self.controlled_sampling_receipt_dir / "online_augmentation.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def _record_source_batch(self, source_codes: torch.Tensor) -> None:
        labels = {0: "official", 1: "wenet", 2: "cv", 3: "mdcc"}
        window_steps = (
            100
            if self.controlled_sampling_policy == "exp004_wenet_mix"
            else 400
        )
        window_index = int(self.state.global_step) // window_steps
        window = self.source_exposure["windows"].setdefault(
            str(window_index + 1),
            {
                "optimizer_step_start": window_index * window_steps + 1,
                "optimizer_step_end": (window_index + 1) * window_steps,
                "samples": 0,
                "source_counts": {
                    "official": 0,
                    "cv": 0,
                    "mdcc": 0,
                    "wenet": 0,
                },
            },
        )
        for code in source_codes.detach().cpu().tolist():
            source = labels.get(int(code))
            if source is None:
                continue
            self.source_exposure["total_samples"] += 1
            self.source_exposure["source_counts"][source] += 1
            window["samples"] += 1
            window["source_counts"][source] += 1
        self.source_exposure["training_microbatches"] += 1
        if self.source_exposure["training_microbatches"] % 100 == 0:
            self.write_source_receipt()

    def source_exposure_snapshot(self) -> dict[str, Any]:
        payload = json.loads(json.dumps(self.source_exposure))
        counts = payload["source_counts"]
        total = int(payload["total_samples"])
        payload["source_rates"] = {
            source: (int(count) / total if total else 0.0)
            for source, count in counts.items()
        }
        payload["source_pool_sizes"] = self.source_pool_sizes
        payload["equivalent_epochs"] = {
            source: (
                int(counts[source]) / int(pool_size)
                if int(pool_size) > 0
                else None
            )
            for source, pool_size in self.source_pool_sizes.items()
        }
        return payload

    def write_source_receipt(self) -> None:
        if self.controlled_sampling_receipt_dir is None:
            return
        self.controlled_sampling_receipt_dir.mkdir(parents=True, exist_ok=True)
        (
            self.controlled_sampling_receipt_dir / "source_sampling.json"
        ).write_text(
            json.dumps(
                self.source_exposure_snapshot(),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> Any:
        speed_factors = inputs.pop("_augmentation_speed_factor", None)
        noise_applied = inputs.pop("_augmentation_noise_applied", None)
        snr_db = inputs.pop("_augmentation_snr_db", None)
        channel_applied = inputs.pop("_augmentation_channel_applied", None)
        source_codes = inputs.pop("_training_source_code", None)
        training_row_indices = inputs.pop("_training_row_index", None)
        if (
            model.training
            and speed_factors is not None
            and noise_applied is not None
            and snr_db is not None
            and channel_applied is not None
        ):
            self._record_augmentation_batch(
                speed_factors,
                noise_applied,
                snr_db,
                channel_applied,
            )
        if model.training and source_codes is not None:
            self._record_source_batch(source_codes)
        if (
            model.training
            and training_row_indices is not None
            and self.controlled_sampling_policy == "fixed_exposure"
        ):
            self._record_fixed_exposure_batch(training_row_indices)
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def _record_fixed_exposure_batch(self, indices: torch.Tensor) -> None:
        """Write rank-local draws so the finalizer can prove global parity."""
        if self.controlled_sampling_receipt_dir is None:
            return
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        self.controlled_sampling_receipt_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "rank": rank,
            "microbatch": self.fixed_exposure_microbatch,
            "optimizer_step": int(self.state.global_step) + 1,
            "row_indices": [int(value) for value in indices.detach().cpu().tolist()],
        }
        path = self.controlled_sampling_receipt_dir / f"fixed_exposure_rank_{rank}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.fixed_exposure_microbatch += 1

    def evaluate(
        self,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        if self.controlled_sampling_receipt_dir is not None:
            snapshot = self.source_exposure_snapshot()
            record = {
                "global_step": int(self.state.global_step),
                "epoch": float(self.state.epoch or 0.0),
                "metric_key_prefix": metric_key_prefix,
                "examples_seen": snapshot["total_samples"],
                "source_examples_seen": snapshot["source_counts"],
                "source_rates": snapshot["source_rates"],
                "equivalent_epochs": snapshot["equivalent_epochs"],
            }
            exposure_path = (
                self.controlled_sampling_receipt_dir
                / "evaluation_exposure.jsonl"
            )
            exposure_path.parent.mkdir(parents=True, exist_ok=True)
            with exposure_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )
        return metrics

    def prediction_step(
        self,
        model: Any,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
        **gen_kwargs: Any,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        inputs = dict(inputs)
        for key in self.augmentation_metadata_keys:
            inputs.pop(key, None)
        return super().prediction_step(
            model,
            inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys,
            **gen_kwargs,
        )

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: Any | None = None,
    ) -> Any:
        """Optionally decouple the LR horizon from this arm's stopping step."""
        self.scheduler_steps_requested_by_trainer = int(num_training_steps)
        scheduler_steps = (
            self.fixed_scheduler_total_steps
            if self.fixed_scheduler_total_steps is not None
            else num_training_steps
        )
        return super().create_scheduler(scheduler_steps, optimizer)

    def _get_train_sampler(self, train_dataset: Dataset | None = None) -> Any:
        if self.controlled_sampling_policy == "uniform":
            try:
                return super()._get_train_sampler(train_dataset)
            except TypeError:
                return super()._get_train_sampler()
        dataset = train_dataset or self.train_dataset
        if dataset is None or not hasattr(dataset, "rows"):
            raise ValueError("Controlled sampling requires ManifestDataset")
        if self.controlled_sampling_policy == "fixed_exposure":
            return FixedExposureSampler(len(dataset.rows))
        return EpochEncodedSampler(
            dataset.rows,
            policy=self.controlled_sampling_policy,
            seed=self.controlled_sampling_seed,
            batch_size=self.controlled_sampling_batch_size,
            receipt_dir=self.controlled_sampling_receipt_dir,
            exp004_wenet_ratio=self.controlled_sampling_wenet_ratio,
            exp004_wenet_stream_length=self.controlled_sampling_stream_length,
            exp004_wenet_window_length=self.controlled_sampling_window_length,
            exp004_wenet_per_window=self.controlled_sampling_wenet_per_window,
            exp004_wenet_fixed_manifest_order=(
                self.controlled_sampling_wenet_fixed_manifest_order
            ),
        )


@dataclass
class SpeechSeq2SeqCollator:
    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        audio_features = [
            {
                "input_features": feature["input_features"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            audio_features, return_tensors="pt"
        )
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        batch["_augmentation_speed_factor"] = torch.tensor(
            [feature["_augmentation_speed_factor"] for feature in features],
            dtype=torch.float32,
        )
        batch["_augmentation_noise_applied"] = torch.tensor(
            [feature["_augmentation_noise_applied"] for feature in features],
            dtype=torch.bool,
        )
        batch["_augmentation_snr_db"] = torch.tensor(
            [feature["_augmentation_snr_db"] for feature in features],
            dtype=torch.float32,
        )
        batch["_augmentation_channel_applied"] = torch.tensor(
            [feature["_augmentation_channel_applied"] for feature in features],
            dtype=torch.bool,
        )
        batch["_training_source_code"] = torch.tensor(
            [feature["_training_source_code"] for feature in features],
            dtype=torch.int8,
        )
        batch["_training_row_index"] = torch.tensor(
            [feature["_training_row_index"] for feature in features],
            dtype=torch.int64,
        )
        return batch


class JsonlMetricsCallback(TrainerCallback):
    """Persist Trainer log/eval/save events in a plotting-friendly schema."""

    def __init__(self, path: Path, trial_name: str) -> None:
        self.path = path
        self.trial_name = trial_name
        self.started_at = time.monotonic()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, event: str, state: Any, values: dict[str, Any]) -> None:
        record: dict[str, Any] = {
            "event": event,
            "trial": self.trial_name,
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "global_step": int(state.global_step),
            "elapsed_seconds": time.monotonic() - self.started_at,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if torch.cuda.is_available():
            record["gpu_peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
            record["gpu_peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
        for key, value in values.items():
            if isinstance(value, (np.floating, np.integer)):
                value = value.item()
            if isinstance(value, (str, int, float, bool)) or value is None:
                record[key] = value
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self.path.exists() and state.global_step == 0:
            self.path.unlink()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._write("train_begin", state, {})

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._write("log", state, logs or {})

    def on_evaluate(
        self,
        args: Any,
        state: Any,
        control: Any,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._write("evaluation", state, metrics or {})

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._write(
            "checkpoint",
            state,
            {"checkpoint": str(Path(args.output_dir) / f"checkpoint-{state.global_step}")},
        )

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._write("train_end", state, {"status": "completed"})


class ExactCheckpointStepsCallback(TrainerCallback):
    """Override periodic Trainer flow with an explicit save/evaluation schedule."""

    def __init__(self, steps: list[int], *, evaluate: bool) -> None:
        self.steps = frozenset(steps)
        self.evaluate = evaluate

    def on_step_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> Any:
        scheduled = int(state.global_step) in self.steps
        control.should_save = scheduled
        control.should_evaluate = scheduled and self.evaluate
        return control


class ValidationEarlyStoppingCallback(EarlyStoppingCallback):
    """Apply patience only to the named validation metric in multi-eval runs."""

    def on_evaluate(
        self,
        args: Any,
        state: Any,
        control: Any,
        metrics: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        if metric_to_check not in metrics:
            return control
        return super().on_evaluate(args, state, control, metrics, **kwargs)


def parse_checkpoint_steps(value: str) -> list[int]:
    try:
        steps = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--checkpoint-steps must be comma-separated integers"
        ) from error
    if not steps or any(step < 1 for step in steps):
        raise argparse.ArgumentTypeError(
            "--checkpoint-steps must contain positive integers"
        )
    if steps != sorted(set(steps)):
        raise argparse.ArgumentTypeError(
            "--checkpoint-steps must be unique and strictly increasing"
        )
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument(
        "--official-mix-manifest",
        type=Path,
        help=(
            "Append the fixed Official train manifest in memory. This is only "
            "valid with the Round 12 Wenet/Official 80/20 sampling policy."
        ),
    )
    parser.add_argument(
        "--wenet-mix-manifest",
        type=Path,
        help=(
            "Append a fixed WenetSpeech-Yue manifest in memory. This is only "
            "valid with the EXP004/Wenet source-aware sampling policy."
        ),
    )
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--train-probe-manifest", type=Path)
    parser.add_argument("--final-refit", action="store_true")
    parser.add_argument(
        "--freeze-encoder-layers",
        type=int,
        default=0,
        help=(
            "Freeze the encoder convolutional frontend and the first N encoder "
            "Transformer blocks. N=12 freezes the complete Whisper-small encoder."
        ),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--trial-name", default="baseline")
    parser.add_argument(
        "--dependency-lock",
        type=Path,
        help=(
            "Optional immutable dependency receipt used to resolve this run. "
            "Its path and SHA-256 are recorded in run_config.json."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--scheduler", choices=["linear", "cosine", "constant"], default="linear"
    )
    parser.add_argument(
        "--sampling-policy",
        choices=[
            "uniform",
            "fixed_exposure",
            "official_speed",
            "source_balanced",
            "round11_source_balanced",
            "wenet_official_80_20",
            "exp004_wenet_mix",
        ],
        default="uniform",
    )
    parser.add_argument(
        "--sampling-wenet-ratio",
        type=float,
        default=0.0,
        help="Target Wenet exposure ratio for exp004_wenet_mix.",
    )
    parser.add_argument(
        "--sampling-stream-length",
        type=int,
        default=25_600,
        help="Number of source-aware sample draws in one sampler epoch.",
    )
    parser.add_argument(
        "--sampling-window-length",
        type=int,
        default=1_600,
        help="Source-ratio audit window in sample draws.",
    )
    parser.add_argument(
        "--sampling-wenet-per-window",
        type=int,
        help=(
            "Optional exact Wenet draw count in each source audit window. "
            "This replaces the floating ratio calculation for "
            "exp004_wenet_mix while preserving deterministic interleaving."
        ),
    )
    parser.add_argument(
        "--sampling-wenet-fixed-manifest-order",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Consume the Wenet mix manifest in frozen row order. Its row count "
            "must exactly equal total requested Wenet exposure."
        ),
    )
    parser.add_argument("--official-speed-manifest", type=Path)
    parser.add_argument(
        "--evaluation-strategy", choices=["epoch", "steps"], default="epoch"
    )
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument(
        "--checkpoint-steps",
        type=parse_checkpoint_steps,
        help=(
            "Comma-separated exact global steps at which to evaluate and save. "
            "Requires --evaluation-strategy steps and replaces periodic "
            "--eval-steps/--save-steps."
        ),
    )
    parser.add_argument(
        "--scheduler-total-steps",
        type=int,
        help=(
            "Optional fixed optimizer-step horizon for scheduler and warmup "
            "construction, independent of --max-steps."
        ),
    )
    parser.add_argument(
        "--optimizer-state-from",
        type=Path,
        help="Load AdamW moments from optimizer.pt without restoring old Trainer step state.",
    )
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-seed",
        type=int,
        help="Dataset/augmentation seed; defaults to --seed.",
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        help="Source sampler seed; defaults to --seed.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--generation-max-length", type=int, default=225)
    parser.add_argument("--generation-num-beams", type=int, default=1)
    parser.add_argument("--generation-no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--generation-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument(
        "--apply-spec-augment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Whisper's native training-only SpecAugment masking.",
    )
    parser.add_argument("--mask-time-prob", type=float, default=0.05)
    parser.add_argument("--mask-time-length", type=int, default=10)
    parser.add_argument("--mask-time-min-masks", type=int, default=2)
    parser.add_argument("--mask-feature-prob", type=float, default=0.0)
    parser.add_argument("--mask-feature-length", type=int, default=10)
    parser.add_argument("--mask-feature-min-masks", type=int, default=0)
    parser.add_argument(
        "--online-speed-perturbation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--online-gaussian-noise",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--online-gaussian-noise-probability", type=float, default=1.0
    )
    parser.add_argument("--noise-snr-min-db", type=float, default=15.0)
    parser.add_argument("--noise-snr-max-db", type=float, default=25.0)
    parser.add_argument(
        "--online-channel-bandlimit",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--channel-bandlimit-probability", type=float, default=0.5)
    parser.add_argument("--lora", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="",
        help="Comma-separated PEFT target-module suffixes.",
    )
    parser.add_argument(
        "--lora-target-scope",
        choices=[
            "encoder_self_qv",
            "decoder_cross_qv",
            "encoder_self_decoder_cross_qv",
            "all_attention_qv",
        ],
        help=(
            "Resolve an exact, architecture-aware LoRA module set. This is "
            "mutually exclusive with --lora-target-modules."
        ),
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--max-train-probe-samples", type=int)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fsdp-full-shard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable full-shard FSDP with Whisper layer auto-wrap.",
    )
    return parser.parse_args()


def configure_spec_augment(
    model: WhisperForConditionalGeneration,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Validate, apply, and record Whisper's native SpecAugment policy."""
    if not 0.0 <= args.mask_time_prob <= 1.0:
        raise ValueError("--mask-time-prob must be within [0, 1]")
    if args.mask_time_length < 1:
        raise ValueError("--mask-time-length must be positive")
    if args.mask_time_min_masks < 0:
        raise ValueError("--mask-time-min-masks must be non-negative")
    if not 0.0 <= args.mask_feature_prob <= 1.0:
        raise ValueError("--mask-feature-prob must be within [0, 1]")
    if args.mask_feature_length < 1:
        raise ValueError("--mask-feature-length must be positive")
    if args.mask_feature_min_masks < 0:
        raise ValueError("--mask-feature-min-masks must be non-negative")

    model.config.apply_spec_augment = bool(args.apply_spec_augment)
    model.config.mask_time_prob = float(args.mask_time_prob)
    model.config.mask_time_length = int(args.mask_time_length)
    model.config.mask_time_min_masks = int(args.mask_time_min_masks)
    model.config.mask_feature_prob = float(args.mask_feature_prob)
    model.config.mask_feature_length = int(args.mask_feature_length)
    model.config.mask_feature_min_masks = int(args.mask_feature_min_masks)
    return {
        "enabled": bool(model.config.apply_spec_augment),
        "implementation": "transformers_whisper_native",
        "training_only": True,
        "mask_time_prob": float(model.config.mask_time_prob),
        "mask_time_length": int(model.config.mask_time_length),
        "mask_time_min_masks": int(model.config.mask_time_min_masks),
        "mask_feature_prob": float(model.config.mask_feature_prob),
        "mask_feature_length": int(model.config.mask_feature_length),
        "mask_feature_min_masks": int(model.config.mask_feature_min_masks),
        "global_application_probability": None,
        "global_application_note": (
            "Transformers Whisper native SpecAugment has no global probability "
            "gate; mask probabilities describe masked-axis coverage."
        ),
    }


def parse_lora_target_modules(value: str) -> list[str]:
    targets = [item.strip() for item in value.split(",") if item.strip()]
    if len(targets) != len(set(targets)):
        raise ValueError("--lora-target-modules contains duplicates")
    allowed = {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"}
    unsupported = sorted(set(targets) - allowed)
    if unsupported:
        raise ValueError(f"Unsupported LoRA target modules: {unsupported}")
    return targets


LORA_SCOPE_REGIONS = {
    "encoder_self_qv": frozenset({"encoder_self"}),
    "decoder_cross_qv": frozenset({"decoder_cross"}),
    "encoder_self_decoder_cross_qv": frozenset(
        {"encoder_self", "decoder_cross"}
    ),
    "all_attention_qv": frozenset(
        {"encoder_self", "decoder_self", "decoder_cross"}
    ),
}


def classify_lora_module(name: str) -> str | None:
    """Classify one exact Whisper attention projection module."""
    parts = name.split(".")
    if len(parts) < 6 or parts[-1] not in {"q_proj", "v_proj"}:
        return None
    if "encoder" in parts and "layers" in parts and "self_attn" in parts:
        return "encoder_self"
    if "decoder" in parts and "layers" in parts:
        if "encoder_attn" in parts:
            return "decoder_cross"
        if "self_attn" in parts:
            return "decoder_self"
    return None


def resolve_lora_scope_modules(
    model: WhisperForConditionalGeneration,
    scope: str,
) -> tuple[list[str], dict[str, list[str]]]:
    """Resolve a preregistered LoRA scope to exact model module names."""
    expected_regions = LORA_SCOPE_REGIONS.get(scope)
    if expected_regions is None:
        raise ValueError(f"Unsupported LoRA target scope: {scope}")

    by_region: dict[str, list[str]] = {
        "encoder_self": [],
        "decoder_self": [],
        "decoder_cross": [],
    }
    for name, module in model.named_modules():
        region = classify_lora_module(name)
        if region is not None and isinstance(module, torch.nn.Linear):
            by_region[region].append(name)
    for names in by_region.values():
        names.sort()

    missing = sorted(region for region in expected_regions if not by_region[region])
    if missing:
        raise ValueError(
            f"LoRA scope {scope} did not resolve required regions: {missing}"
        )
    targets = sorted(
        name
        for region in expected_regions
        for name in by_region[region]
    )
    return targets, {
        region: names
        for region, names in by_region.items()
        if region in expected_regions
    }


def configure_lora(
    model: WhisperForConditionalGeneration,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    target_scope = getattr(args, "lora_target_scope", None)
    raw_targets = getattr(args, "lora_target_modules", "")
    if target_scope and raw_targets.strip():
        raise ValueError(
            "--lora-target-scope is mutually exclusive with "
            "--lora-target-modules"
        )
    resolved_by_region: dict[str, list[str]] = {}
    if target_scope:
        targets, resolved_by_region = resolve_lora_scope_modules(
            model, target_scope
        )
    else:
        targets = parse_lora_target_modules(raw_targets)
    if not args.lora:
        if targets or target_scope:
            raise ValueError("LoRA targets require --lora")
        return model, {
            "used": False,
            "enabled": False,
            "target_scope": None,
            "target_modules": [],
            "matched_modules": [],
            "matched_module_count": 0,
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "total_parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        }
    if args.lora_r < 1 or args.lora_alpha < 1:
        raise ValueError("--lora-r and --lora-alpha must be positive")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("--lora-dropout must be within [0, 1)")
    if not targets:
        raise ValueError("--lora-target-modules is required with --lora")

    wrapped = WhisperPeftModelForSeq2SeqLM(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=targets,
        ),
    )
    trainable = [
        (name, parameter.numel())
        for name, parameter in wrapped.named_parameters()
        if parameter.requires_grad
    ]
    non_lora_trainable = [
        name for name, _ in trainable if ".lora_" not in name
    ]
    if non_lora_trainable:
        raise ValueError(
            "Only LoRA parameters may be trainable; found "
            f"{non_lora_trainable[:5]}"
        )
    matched_modules = sorted(
        {
            name.rsplit(".lora_", 1)[0]
            for name, _ in wrapped.named_parameters()
            if ".lora_" in name
        }
    )
    matched_by_suffix = {
        target: sum(name.endswith(f".{target}") for name in matched_modules)
        for target in targets
    }
    if not matched_modules or any(count == 0 for count in matched_by_suffix.values()):
        raise ValueError(
            f"LoRA targets did not all match model modules: {matched_by_suffix}"
        )
    matched_by_region: dict[str, list[str]] = {
        "encoder_self": [],
        "decoder_self": [],
        "decoder_cross": [],
        "other": [],
    }
    for name in matched_modules:
        region = classify_lora_module(name) or "other"
        matched_by_region[region].append(name)
    matched_by_region = {
        region: names
        for region, names in matched_by_region.items()
        if names
    }
    if target_scope:
        expected_regions = LORA_SCOPE_REGIONS[target_scope]
        unexpected_regions = sorted(set(matched_by_region) - expected_regions)
        if unexpected_regions:
            raise ValueError(
                f"LoRA scope {target_scope} matched unexpected regions: "
                f"{unexpected_regions}"
            )
        expected_target_count = sum(
            len(names) for names in resolved_by_region.values()
        )
        if len(matched_modules) != expected_target_count:
            raise ValueError(
                f"LoRA scope {target_scope} matched {len(matched_modules)} "
                f"modules; expected {expected_target_count}"
            )
    trainable_parameter_count = sum(count for _, count in trainable)
    total_parameter_count = sum(
        parameter.numel() for parameter in wrapped.parameters()
    )
    policy = {
        "used": True,
        "enabled": True,
        "implementation": "huggingface_peft",
        "task_type": "SEQ_2_SEQ_LM",
        "bias": "none",
        "rank_r": args.lora_r,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
        "target_scope": target_scope,
        "target_modules": targets,
        "matched_modules": matched_modules,
        "matched_module_count": len(matched_modules),
        "matched_by_suffix": matched_by_suffix,
        "matched_by_region": matched_by_region,
        "resolved_by_region": resolved_by_region,
        "trainable_parameters": trainable_parameter_count,
        "total_parameters": total_parameter_count,
        "trainable_percentage": (
            100.0 * trainable_parameter_count / total_parameter_count
        ),
        "trainable_tensor_count": len(trainable),
        "trainable_parameter_names": [name for name, _ in trainable],
        "only_lora_trainable": True,
    }
    print(json.dumps({"lora_setup": policy}, ensure_ascii=False, indent=2))
    return wrapped, policy


def resolve_resume(value: str | None, output_dir: Path) -> str | bool | None:
    if not value:
        return None
    if value != "latest":
        path = Path(value)
        if not path.is_dir():
            raise ValueError(f"Checkpoint not found: {path}")
        return str(path)
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    if not checkpoints:
        raise ValueError(f"No checkpoints below {output_dir}")
    return str(checkpoints[-1])


def apply_encoder_freeze(
    model: WhisperForConditionalGeneration,
    freeze_layers: int,
) -> dict[str, Any]:
    """Apply and describe a deterministic bottom-up Whisper encoder freeze."""
    encoder = model.model.encoder
    total_layers = len(encoder.layers)
    if not 0 <= freeze_layers <= total_layers:
        raise ValueError(
            f"--freeze-encoder-layers must be within [0, {total_layers}], "
            f"got {freeze_layers}"
        )

    frozen_components: list[str] = []
    if freeze_layers:
        for component_name in ("conv1", "conv2"):
            component = getattr(encoder, component_name)
            for parameter in component.parameters():
                parameter.requires_grad = False
            frozen_components.append(f"model.encoder.{component_name}")
        for index, layer in enumerate(encoder.layers[:freeze_layers]):
            for parameter in layer.parameters():
                parameter.requires_grad = False
            frozen_components.append(f"model.encoder.layers.{index}")
        if freeze_layers == total_layers:
            for parameter in encoder.layer_norm.parameters():
                parameter.requires_grad = False
            frozen_components.append("model.encoder.layer_norm")

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "requested_encoder_layers": freeze_layers,
        "total_encoder_layers": total_layers,
        "convolutional_frontend_frozen": bool(freeze_layers),
        "encoder_final_layer_norm_frozen": freeze_layers == total_layers,
        "frozen_components": frozen_components,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "frozen_parameters": total_parameters - trainable_parameters,
    }


def write_run_config(
    args: argparse.Namespace,
    model: Any,
    output_dir: Path,
    freeze_policy: dict[str, Any],
    spec_augment_policy: dict[str, Any],
    lora_policy: dict[str, Any],
) -> None:
    code_paths = [Path(__file__), *sorted((Path(__file__).parent / "cantonese_asr").glob("*.py"))]
    manifest_paths = [
        args.train_manifest,
        args.official_mix_manifest,
        args.wenet_mix_manifest,
        args.validation_manifest,
        args.train_probe_manifest,
        args.official_speed_manifest,
    ]
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    frozen_parameters = [
        {"name": name, "num_parameters": parameter.numel()}
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    ]
    # Hugging Face Whisper intentionally keeps the encoder's sinusoidal position
    # table fixed. Full SFT means every learnable parameter is enabled; the fixed
    # table is part of the upstream architecture and must not be made learnable.
    expected_fixed = {"model.encoder.embed_positions.weight"}
    full_sft = {item["name"] for item in frozen_parameters} <= expected_fixed
    source_config = args.model / "config.json"
    source_weight = args.model / "model.safetensors"
    model_source: dict[str, Any] = {
        "path": str(args.model.resolve()),
        "config_sha256": sha256_file(source_config)
        if source_config.is_file()
        else None,
        "model_safetensors_sha256": sha256_file(source_weight)
        if source_weight.is_file()
        else None,
    }
    if source_config.is_file():
        config = json.loads(source_config.read_text(encoding="utf-8"))
        model_source["config"] = {
            key: config.get(key)
            for key in (
                "model_type",
                "architectures",
                "d_model",
                "encoder_layers",
                "decoder_layers",
                "encoder_attention_heads",
                "decoder_attention_heads",
                "encoder_ffn_dim",
                "decoder_ffn_dim",
                "vocab_size",
            )
        }
    payload = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model": {
            "architecture": model.__class__.__name__,
            "total_parameters": total,
            "trainable_parameters": trainable,
            "all_parameters_trainable": trainable == total,
            "full_sft": full_sft,
            "fine_tuning_method": "lora" if lora_policy["used"] else "full_sft",
            "source": model_source,
            "encoder_freeze": freeze_policy,
            "spec_augment": spec_augment_policy,
            "lora": lora_policy,
            "frozen_parameters": frozen_parameters,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "visible_gpu_count": torch.cuda.device_count(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "visible_gpu_count": (
                int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
            ),
            "training_world_size": runtime_training_world_size(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "manifest_sha256": {
            path.name: sha256_file(path)
            for path in manifest_paths
            if path is not None and path.is_file()
        },
        "dependency_lock": (
            {
                "path": str(args.dependency_lock.resolve()),
                "sha256": sha256_file(args.dependency_lock),
            }
            if args.dependency_lock is not None
            else None
        ),
        "code_sha256": sha256_text_files(code_paths),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.fp16 and args.bf16:
        raise SystemExit("Choose only one of --fp16 and --bf16")
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise SystemExit("Batch size and gradient accumulation must be positive")
    if args.warmup_steps < 0:
        raise SystemExit("--warmup-steps must be non-negative")
    if args.warmup_steps and args.warmup_ratio:
        raise SystemExit("Choose only one of --warmup-steps and --warmup-ratio")
    if args.early_stopping_patience < 0 or args.save_total_limit < 1:
        raise SystemExit("Early stopping patience must be non-negative and save limit positive")
    if not 0 <= args.freeze_encoder_layers <= 12:
        raise SystemExit("--freeze-encoder-layers must be within [0, 12]")
    if args.lora and args.freeze_encoder_layers:
        raise SystemExit("LoRA experiments do not allow manual encoder freezing")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.fsdp_full_shard and world_size < 2:
        raise SystemExit("--fsdp-full-shard requires WORLD_SIZE >= 2")
    if args.fsdp_full_shard and args.lora:
        raise SystemExit("P2 FSDP full-shard is only registered for Full SFT")
    if args.sampling_policy == "fixed_exposure":
        effective_batch_size = (
            args.batch_size * args.gradient_accumulation_steps * world_size
        )
        if args.max_steps < 1:
            raise SystemExit("fixed_exposure requires a positive --max-steps")
        expected_rows = args.max_steps * effective_batch_size
        actual_rows = sum(1 for _ in args.train_manifest.open(encoding="utf-8"))
        if actual_rows != expected_rows:
            raise SystemExit(
                "fixed exposure row count must equal max_steps * global batch: "
                f"{actual_rows} != {expected_rows}"
            )
    if args.sampling_policy == "official_speed" and args.official_speed_manifest is None:
        raise SystemExit("--official-speed-manifest is required for official_speed")
    if args.sampling_policy != "official_speed" and args.official_speed_manifest is not None:
        raise SystemExit("--official-speed-manifest is only valid for official_speed")
    if args.sampling_policy == "wenet_official_80_20":
        if args.official_mix_manifest is None:
            raise SystemExit(
                "--official-mix-manifest is required for wenet_official_80_20"
            )
    elif args.official_mix_manifest is not None:
        raise SystemExit(
            "--official-mix-manifest is only valid for wenet_official_80_20"
        )
    if args.sampling_policy == "exp004_wenet_mix":
        if not 0.0 <= args.sampling_wenet_ratio < 1.0:
            raise SystemExit("--sampling-wenet-ratio must be within [0, 1)")
        if (
            args.sampling_wenet_per_window is not None
            and args.sampling_wenet_ratio != 0.0
        ):
            raise SystemExit(
                "Choose only one of --sampling-wenet-ratio and "
                "--sampling-wenet-per-window"
            )
        if (
            args.sampling_wenet_per_window is not None
            and not 0
            <= args.sampling_wenet_per_window
            < args.sampling_window_length
        ):
            raise SystemExit(
                "--sampling-wenet-per-window must be within "
                "[0, sampling window length)"
            )
        has_wenet_draws = (
            args.sampling_wenet_per_window > 0
            if args.sampling_wenet_per_window is not None
            else args.sampling_wenet_ratio > 0
        )
        if has_wenet_draws and args.wenet_mix_manifest is None:
            raise SystemExit(
                "--wenet-mix-manifest is required when Wenet exposure is positive"
            )
        if not has_wenet_draws and args.wenet_mix_manifest is not None:
            raise SystemExit(
                "A zero-Wenet arm must not receive --wenet-mix-manifest"
            )
        effective_batch_size = (
            args.batch_size
            * args.gradient_accumulation_steps
            * runtime_training_world_size()
        )
        if args.sampling_stream_length != args.max_steps * effective_batch_size:
            raise SystemExit(
                "EXP004/Wenet stream length must equal max_steps * effective batch size"
            )
    elif (
        args.wenet_mix_manifest is not None
        or args.sampling_wenet_ratio != 0
        or args.sampling_wenet_per_window is not None
    ):
        raise SystemExit(
            "Wenet mix sampler options are only valid for exp004_wenet_mix"
        )
    if args.sampling_stream_length < 1 or args.sampling_window_length < 1:
        raise SystemExit("Sampling stream/window lengths must be positive")
    if args.generation_no_repeat_ngram_size < 0:
        raise SystemExit("--generation-no-repeat-ngram-size must be non-negative")
    if args.generation_repetition_penalty <= 0:
        raise SystemExit("--generation-repetition-penalty must be positive")
    if not 0.0 <= args.online_gaussian_noise_probability <= 1.0:
        raise SystemExit("--online-gaussian-noise-probability must be within [0, 1]")
    if not 0.0 <= args.channel_bandlimit_probability <= 1.0:
        raise SystemExit("--channel-bandlimit-probability must be within [0, 1]")
    augmentation_axes = sum(
        bool(value)
        for value in (
            args.apply_spec_augment,
            args.online_speed_perturbation,
            args.online_gaussian_noise,
            args.online_channel_bandlimit,
        )
    )
    if augmentation_axes > 1:
        raise SystemExit(
            "SpecAugment, speed, Gaussian noise, and channel bandlimit are "
            "mutually exclusive single-axis augmentations"
        )
    if args.noise_snr_min_db >= args.noise_snr_max_db:
        raise SystemExit("--noise-snr-min-db must be below --noise-snr-max-db")
    if args.dependency_lock is not None and not args.dependency_lock.is_file():
        raise SystemExit(f"Dependency lock not found: {args.dependency_lock}")
    if args.scheduler_total_steps is not None:
        if args.scheduler_total_steps < 1:
            raise SystemExit("--scheduler-total-steps must be positive")
        if args.max_steps > args.scheduler_total_steps:
            raise SystemExit(
                "--max-steps cannot exceed --scheduler-total-steps"
            )
    if args.checkpoint_steps:
        if args.evaluation_strategy != "steps":
            raise SystemExit(
                "--checkpoint-steps requires --evaluation-strategy steps"
            )
        if args.eval_steps or args.save_steps:
            raise SystemExit(
                "--checkpoint-steps replaces --eval-steps/--save-steps"
            )
        if args.max_steps > 0 and args.checkpoint_steps[-1] > args.max_steps:
            raise SystemExit(
                "--checkpoint-steps cannot extend beyond --max-steps"
            )
    elif args.evaluation_strategy == "steps":
        if not args.eval_steps or not args.save_steps:
            raise SystemExit("Step evaluation requires --eval-steps and --save-steps")
    elif args.eval_steps or args.save_steps:
        raise SystemExit("--eval-steps/--save-steps require --evaluation-strategy steps")
    if args.final_refit:
        if args.validation_manifest is not None:
            raise SystemExit("Final refit must not receive --validation-manifest")
        if args.train_probe_manifest is not None:
            raise SystemExit("Final refit must not receive --train-probe-manifest")
        if args.early_stopping_patience:
            raise SystemExit("Final refit does not allow early stopping")
    elif args.validation_manifest is None:
        raise SystemExit("--validation-manifest is required unless --final-refit is set")
    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and not args.resume_from_checkpoint
    ):
        raise SystemExit(
            f"Output directory already contains files: {args.output_dir}"
        )
    set_seed(args.seed)
    is_primary_process = int(os.environ.get("RANK", "0")) == 0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor = WhisperProcessor.from_pretrained(
        args.model, language="zh", task="transcribe", local_files_only=True
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True
    )
    model.generation_config.language = "zh"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.do_sample = False
    model.generation_config.num_beams = int(args.generation_num_beams)
    model.generation_config.no_repeat_ngram_size = int(
        args.generation_no_repeat_ngram_size
    )
    model.generation_config.repetition_penalty = float(
        args.generation_repetition_penalty
    )
    model.generation_config.length_penalty = 1.0
    model.generation_config.early_stopping = args.generation_num_beams > 1
    model.generation_config.max_length = int(args.generation_max_length)
    model.config.use_cache = False
    try:
        freeze_policy = apply_encoder_freeze(model, args.freeze_encoder_layers)
        spec_augment_policy = configure_spec_augment(model, args)
        model, lora_policy = configure_lora(model, args)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    train_dataset = ManifestDataset(
        args.train_manifest,
        processor,
        args.project_root,
        official_mix_manifest=args.official_mix_manifest,
        wenet_mix_manifest=args.wenet_mix_manifest,
        max_samples=args.max_train_samples,
        sampling_policy=args.sampling_policy,
        speed_manifest=args.official_speed_manifest,
        seed=args.data_seed if args.data_seed is not None else args.seed,
        epochs=int(np.ceil(args.epochs)),
        sampling_receipt_dir=args.output_dir / "sampling",
        online_speed_perturbation=args.online_speed_perturbation,
        online_gaussian_noise=args.online_gaussian_noise,
        online_gaussian_noise_probability=args.online_gaussian_noise_probability,
        noise_snr_min_db=args.noise_snr_min_db,
        noise_snr_max_db=args.noise_snr_max_db,
        online_channel_bandlimit=args.online_channel_bandlimit,
        channel_bandlimit_probability=args.channel_bandlimit_probability,
    )
    eval_datasets: Dataset | dict[str, Dataset] | None = None
    best_metric: str | None = None
    if not args.final_refit:
        assert args.validation_manifest is not None
        validation_dataset = ManifestDataset(
            args.validation_manifest,
            processor,
            args.project_root,
            args.max_validation_samples,
        )
        if args.train_probe_manifest:
            train_probe_dataset = ManifestDataset(
                args.train_probe_manifest,
                processor,
                args.project_root,
                args.max_train_probe_samples,
            )
            eval_datasets = {
                "validation": validation_dataset,
                "train_probe": train_probe_dataset,
            }
            best_metric = "eval_validation_sentence_accuracy_tol2"
        else:
            eval_datasets = validation_dataset
            best_metric = "eval_sentence_accuracy_tol2"

    def compute_metrics(prediction: Any) -> dict[str, float | int]:
        predicted_ids = prediction.predictions
        if isinstance(predicted_ids, tuple):
            predicted_ids = predicted_ids[0]
        predicted_ids = np.asarray(predicted_ids)
        label_ids = np.array(prediction.label_ids, copy=True)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        predictions = processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        references = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )
        metrics = compute_official_metrics(references, predictions)
        diagnostics = compute_diagnostic_metrics(references, predictions)
        metrics.update(
            {
                "substitutions": int(diagnostics["substitutions"]),
                "deletions": int(diagnostics["deletions"]),
                "insertions": int(diagnostics["insertions"]),
                "severe_error_count": int(diagnostics["severe_error_count"]),
                "severe_error_rate": float(diagnostics["severe_error_rate"]),
                **{
                    f"edit_distance_bucket_{bucket.replace('>', 'gt').replace('-', '_')}": int(
                        values["count"]
                    )
                    for bucket, values in diagnostics[
                        "edit_distance_buckets"
                    ].items()
                },
            }
        )
        eos_ids = {
            int(token)
            for token in (
                processor.tokenizer.eos_token_id,
                processor.tokenizer.pad_token_id,
            )
            if token is not None
        }
        special_ids = {int(token) for token in processor.tokenizer.all_special_ids}
        effective_max_length_count = 0
        no_eos_count = 0
        repetition_loop_count = 0
        runaway_count = 0
        for sequence in predicted_ids:
            raw_tokens = [int(token) for token in np.asarray(sequence).tolist()]
            actual_tokens = raw_tokens
            for position, token in enumerate(raw_tokens[1:], start=1):
                if token in eos_ids:
                    actual_tokens = raw_tokens[: position + 1]
                    break
            has_eos = any(token in eos_ids for token in actual_tokens)
            no_eos_count += int(not has_eos)
            effective_max_length_count += int(
                len(actual_tokens) + 4 >= args.generation_max_length
            )
            detection = detect_runaway(
                actual_tokens,
                special_token_ids=special_ids,
                sequence_width=len(actual_tokens) + 4,
                max_length=args.generation_max_length,
            )
            repetition_loop_count += int(detection.triggered)
            runaway_count += int(not has_eos or detection.triggered)
        metrics["effective_max_length_count"] = effective_max_length_count
        metrics["max_length_count"] = effective_max_length_count
        metrics["no_eos_count"] = no_eos_count
        metrics["repetition_loop_count"] = repetition_loop_count
        metrics["replacement_character_count"] = sum(
            text.count("\ufffd") for text in predictions
        )
        metrics["runaway_count"] = runaway_count
        return metrics

    exact_interval_placeholder = (
        args.checkpoint_steps[0] if args.checkpoint_steps else None
    )
    fsdp_config = None
    fsdp = None
    if args.fsdp_full_shard:
        fsdp = "full_shard auto_wrap"
        fsdp_config = {
            "transformer_layer_cls_to_wrap": [
                "WhisperEncoderLayer",
                "WhisperDecoderLayer",
            ],
            "use_orig_params": True,
            "sync_module_states": True,
            "cpu_ram_efficient_loading": False,
            "forward_prefetch": False,
            "backward_prefetch": "BACKWARD_PRE",
        }
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=False,
        run_name=args.trial_name,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.scheduler,
        optim="adamw_torch",
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=args.fp16,
        bf16=args.bf16,
        tf32=args.tf32,
        eval_strategy=(
            "no" if args.final_refit else args.evaluation_strategy
        ),
        save_strategy=(
            "steps" if args.evaluation_strategy == "steps" else "epoch"
        ),
        eval_steps=exact_interval_placeholder or args.eval_steps,
        save_steps=exact_interval_placeholder or args.save_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        predict_with_generate=not args.final_refit,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
        load_best_model_at_end=not args.final_refit,
        metric_for_best_model=best_metric,
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        max_grad_norm=1.0,
        eval_accumulation_steps=1,
        skip_memory_metrics=False,
        seed=args.seed,
        data_seed=args.data_seed if args.data_seed is not None else args.seed,
        fsdp=fsdp,
        fsdp_config=fsdp_config,
    )

    metrics_callback = JsonlMetricsCallback(
        args.output_dir / "metrics.jsonl", args.trial_name
    )
    callbacks: list[TrainerCallback] = [metrics_callback]
    if args.early_stopping_patience:
        callbacks.append(
            ValidationEarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )
    if args.checkpoint_steps:
        callbacks.append(
            ExactCheckpointStepsCallback(
                args.checkpoint_steps,
                evaluate=not args.final_refit,
            )
        )

    if is_primary_process:
        processor.save_pretrained(args.output_dir)
        write_run_config(
            args,
            model,
            args.output_dir,
            freeze_policy,
            spec_augment_policy,
            lora_policy,
        )
    trainer = ControlledSamplingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        data_collator=SpeechSeq2SeqCollator(processor),
        compute_metrics=None if args.final_refit else compute_metrics,
        processing_class=processor,
        callbacks=callbacks,
        sampling_policy=args.sampling_policy,
        sampling_seed=(
            args.sampling_seed if args.sampling_seed is not None else args.seed
        ),
        sampling_batch_size=args.batch_size,
        sampling_receipt_dir=args.output_dir / "sampling",
        sampling_wenet_ratio=args.sampling_wenet_ratio,
        sampling_stream_length=args.sampling_stream_length,
        sampling_window_length=args.sampling_window_length,
        sampling_wenet_per_window=args.sampling_wenet_per_window,
        sampling_wenet_fixed_manifest_order=(
            args.sampling_wenet_fixed_manifest_order
        ),
        scheduler_total_steps=args.scheduler_total_steps,
    )
    optimizer_state_receipt = {
        "requested": str(args.optimizer_state_from)
        if args.optimizer_state_from
        else None,
        "restored": False,
        "scheduler_state_restored": False,
        "confounder": None,
    }
    if args.optimizer_state_from:
        optimizer_path = args.optimizer_state_from
        if optimizer_path.is_dir():
            optimizer_path = optimizer_path / "optimizer.pt"
        if not optimizer_path.is_file():
            raise SystemExit(f"Optimizer state not found: {optimizer_path}")
        trainer.create_optimizer()
        assert trainer.optimizer is not None
        try:
            state = torch.load(optimizer_path, map_location="cpu", weights_only=True)
            trainer.optimizer.load_state_dict(state)
            for group in trainer.optimizer.param_groups:
                group["lr"] = args.learning_rate
                group["initial_lr"] = args.learning_rate
            optimizer_state_receipt.update(
                {
                    "resolved_path": str(optimizer_path.resolve()),
                    "sha256": sha256_file(optimizer_path),
                    "restored": True,
                    "parameter_groups": len(trainer.optimizer.param_groups),
                }
            )
        except Exception as error:
            optimizer_state_receipt["confounder"] = (
                f"optimizer_state_restore_failed:{type(error).__name__}:{error}"
            )
            raise SystemExit(optimizer_state_receipt["confounder"]) from error
    (args.output_dir / "optimizer_state_receipt.json").write_text(
        json.dumps(optimizer_state_receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    resume = resolve_resume(args.resume_from_checkpoint, args.output_dir)
    train_result = trainer.train(resume_from_checkpoint=resume)
    trainer.write_augmentation_receipt()
    trainer.write_source_receipt()
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    if args.checkpoint_steps:
        actual_checkpoint_steps = sorted(
            int(path.name.rsplit("-", 1)[-1])
            for path in args.output_dir.glob("checkpoint-*")
            if path.is_dir() and path.name.rsplit("-", 1)[-1].isdigit()
        )
        completed_schedule = [
            step
            for step in args.checkpoint_steps
            if step <= int(trainer.state.global_step)
        ]
        missing_steps = sorted(
            set(completed_schedule) - set(actual_checkpoint_steps)
        )
        unexpected_steps = sorted(
            set(actual_checkpoint_steps) - set(args.checkpoint_steps)
        )
        checkpoint_schedule_receipt = {
            "passed": not missing_steps and not unexpected_steps,
            "requested_checkpoint_steps": args.checkpoint_steps,
            "completed_schedule": completed_schedule,
            "actual_checkpoint_steps": actual_checkpoint_steps,
            "missing_steps": missing_steps,
            "unexpected_steps": unexpected_steps,
            "completed_global_step": int(trainer.state.global_step),
            "scheduler_total_steps": (
                args.scheduler_total_steps
                if args.scheduler_total_steps is not None
                else trainer.scheduler_steps_requested_by_trainer
            ),
            "training_steps_requested_by_trainer": (
                trainer.scheduler_steps_requested_by_trainer
            ),
        }
        (args.output_dir / "checkpoint_schedule_receipt.json").write_text(
            json.dumps(
                checkpoint_schedule_receipt,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if missing_steps or unexpected_steps:
            raise SystemExit(
                "Exact checkpoint schedule mismatch: "
                f"missing={missing_steps} unexpected={unexpected_steps}"
            )

    model.config.use_cache = True
    if args.final_refit:
        final_model_dir = args.output_dir / "final_model"
        trainer.save_model(str(final_model_dir))
        processor.save_pretrained(final_model_dir)
        model.generation_config.save_pretrained(final_model_dir)
        receipt = {
            "mode": "final_refit",
            "selection_policy": "fixed_last_epoch_no_validation_selection",
            "validation_used": False,
            "train_probe_used": False,
            "early_stopping_used": False,
            "final_model": str(final_model_dir),
            "completed_epoch": float(trainer.state.epoch or 0.0),
            "global_step": int(trainer.state.global_step),
            "train_manifest": str(args.train_manifest),
            "train_manifest_sha256": sha256_file(args.train_manifest),
            "seed": args.seed,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        (args.output_dir / "final_refit_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        best_model_dir = args.output_dir / "best_model"
        trainer.save_model(str(best_model_dir))
        processor.save_pretrained(best_model_dir)
        model.generation_config.save_pretrained(best_model_dir)
        final_metrics = trainer.evaluate(metric_key_prefix="final")
        trainer.save_metrics("final", final_metrics)


if __name__ == "__main__":
    main()
