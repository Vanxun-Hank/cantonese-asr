"""Offline loading helpers for full Whisper models and PEFT adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import PeftConfig, PeftModelForSeq2SeqLM
from transformers import WhisperForConditionalGeneration


ADAPTER_CONFIG_NAME = "adapter_config.json"


class WhisperPeftModelForSeq2SeqLM(PeftModelForSeq2SeqLM):
    """PEFT seq2seq wrapper whose forward contract uses Whisper input_features.

    PEFT's generic seq2seq wrapper always forwards ``input_ids`` and
    ``inputs_embeds``. Whisper instead accepts ``input_features`` and rejects
    those text-encoder arguments. LoRA does not use prompt learning, so the
    correct behavior is PEFT's hook-enabled generic delegation.
    """

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        with self._enable_peft_forward_hooks(*args, **kwargs):
            filtered = {
                key: value
                for key, value in kwargs.items()
                if key not in self.special_peft_forward_args
            }
            return self.get_base_model()(*args, **filtered)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        with self._enable_peft_forward_hooks(*args, **kwargs):
            filtered = {
                key: value
                for key, value in kwargs.items()
                if key not in self.special_peft_forward_args
            }
            return self.get_base_model().generate(*args, **filtered)


def is_peft_adapter(path: Path) -> bool:
    return (path / ADAPTER_CONFIG_NAME).is_file()


def resolve_adapter_base_model(adapter_dir: Path) -> Path:
    config = PeftConfig.from_pretrained(
        str(adapter_dir),
        local_files_only=True,
    )
    configured = Path(str(config.base_model_name_or_path))
    candidates = [configured]
    if not configured.is_absolute():
        candidates = [
            (Path.cwd() / configured).resolve(),
            (adapter_dir / configured).resolve(),
        ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "PEFT adapter base model is unavailable offline: "
        f"{config.base_model_name_or_path!r}; checked={candidates}"
    )


def load_whisper_model(
    model_dir: Path,
    *,
    dtype: torch.dtype | None = None,
    adapter_trainable: bool = False,
) -> Any:
    """Load a standalone Whisper model or a base model plus PEFT adapter."""
    common: dict[str, Any] = {
        "local_files_only": True,
        "use_safetensors": True,
    }
    if dtype is not None:
        common["dtype"] = dtype

    if not is_peft_adapter(model_dir):
        return WhisperForConditionalGeneration.from_pretrained(
            model_dir,
            **common,
        )

    base_dir = resolve_adapter_base_model(model_dir)
    base = WhisperForConditionalGeneration.from_pretrained(
        base_dir,
        **common,
    )
    adapter_config = PeftConfig.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )
    adapter_config.inference_mode = not adapter_trainable
    wrapped = WhisperPeftModelForSeq2SeqLM(
        base,
        adapter_config,
    )
    wrapped.load_adapter(
        model_dir,
        adapter_name="default",
        is_trainable=adapter_trainable,
        local_files_only=True,
    )
    return wrapped
