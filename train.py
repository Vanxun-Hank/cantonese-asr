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
from cantonese_asr.metrics import compute_official_metrics


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        processor: WhisperProcessor,
        project_root: Path,
        max_samples: int | None = None,
    ) -> None:
        self.processor = processor
        self.project_root = project_root
        self.rows = read_jsonl(manifest)
        if max_samples is not None:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise ValueError(f"Empty manifest: {manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        audio_path = Path(str(row["audio_path"]))
        if not audio_path.is_absolute():
            audio_path = self.project_root / audio_path
        audio, _ = librosa.load(audio_path, sr=16_000, mono=True)
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
        }


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--train-probe-manifest", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--trial-name", default="baseline")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--scheduler", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--generation-max-length", type=int, default=225)
    parser.add_argument("--generation-num-beams", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--max-train-probe-samples", type=int)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    return parser.parse_args()


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


def write_run_config(
    args: argparse.Namespace,
    model: WhisperForConditionalGeneration,
    output_dir: Path,
) -> None:
    code_paths = [Path(__file__), *sorted((Path(__file__).parent / "cantonese_asr").glob("*.py"))]
    manifest_paths = [
        args.train_manifest,
        args.validation_manifest,
        args.train_probe_manifest,
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
            "frozen_parameters": frozen_parameters,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "manifest_sha256": {
            path.name: sha256_file(path)
            for path in manifest_paths
            if path is not None and path.is_file()
        },
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
    if args.early_stopping_patience < 0:
        raise SystemExit("Early stopping patience cannot be negative")
    set_seed(args.seed)
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
    model.config.use_cache = False

    train_dataset = ManifestDataset(
        args.train_manifest,
        processor,
        args.project_root,
        args.max_train_samples,
    )
    validation_dataset = ManifestDataset(
        args.validation_manifest,
        processor,
        args.project_root,
        args.max_validation_samples,
    )
    eval_datasets: Dataset | dict[str, Dataset]
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
        label_ids = np.array(prediction.label_ids, copy=True)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        predictions = processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        references = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )
        return compute_official_metrics(references, predictions)

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
        lr_scheduler_type=args.scheduler,
        optim="adamw_torch",
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=args.fp16,
        bf16=args.bf16,
        tf32=args.tf32,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=True,
        save_total_limit=3,
        save_safetensors=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        max_grad_norm=1.0,
        eval_accumulation_steps=1,
        skip_memory_metrics=False,
        seed=args.seed,
        data_seed=args.seed,
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

    processor.save_pretrained(args.output_dir)
    write_run_config(args, model, args.output_dir)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        data_collator=SpeechSeq2SeqCollator(processor),
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=callbacks,
    )
    resume = resolve_resume(args.resume_from_checkpoint, args.output_dir)
    train_result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    model.config.use_cache = True
    best_model_dir = args.output_dir / "best_model"
    trainer.save_model(str(best_model_dir))
    processor.save_pretrained(best_model_dir)
    model.generation_config.save_pretrained(best_model_dir)
    final_metrics = trainer.evaluate(metric_key_prefix="final")
    trainer.save_metrics("final", final_metrics)


if __name__ == "__main__":
    main()
