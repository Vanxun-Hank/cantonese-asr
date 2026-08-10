from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from cantonese_asr.w500_decoder_interpolation import (
    audit_interpolation,
    interpolate_decoder_state,
    tensor_state_digest,
)
from scripts.interpolate_w500_decoder import interpolate_checkpoint


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state(encoder: float, decoder: float, projection: float) -> dict[str, torch.Tensor]:
    return {
        "model.encoder.weight": torch.tensor([encoder, encoder + 1]),
        "model.decoder.weight": torch.tensor([decoder, decoder + 2]),
        "proj_out.weight": torch.tensor([projection, projection + 3]),
        "unclassified_buffer": torch.tensor([17.0]),
    }


def test_interpolation_keeps_encoder_and_mixes_only_decoder_and_proj_out() -> None:
    acoustic = state(1.0, 2.0, 3.0)
    stable = state(9.0, 6.0, 7.0)
    mixed = interpolate_decoder_state(acoustic, stable, stable_weight=0.25)

    assert torch.equal(mixed["model.encoder.weight"], acoustic["model.encoder.weight"])
    assert torch.equal(mixed["unclassified_buffer"], acoustic["unclassified_buffer"])
    assert torch.equal(mixed["model.decoder.weight"], torch.tensor([3.0, 5.0]))
    assert torch.equal(mixed["proj_out.weight"], torch.tensor([4.0, 7.0]))
    assert mixed["model.encoder.weight"].data_ptr() != acoustic["model.encoder.weight"].data_ptr()

    receipt = audit_interpolation(acoustic, stable, mixed, stable_weight=0.25)
    assert receipt["passed"] is True
    assert receipt["checks"]["encoder_exact"] is True
    assert receipt["checks"]["alpha_0_reproduces_acoustic_endpoint"] is True
    assert receipt["checks"]["alpha_1_reproduces_stable_decoder_with_acoustic_encoder"] is True
    assert receipt["digests"]["acoustic_encoder"] == receipt["digests"]["mixed_encoder"]


def test_interpolation_endpoints_and_incompatibilities() -> None:
    acoustic = state(1.0, 2.0, 3.0)
    stable = state(9.0, 6.0, 7.0)
    zero = interpolate_decoder_state(acoustic, stable, stable_weight=0.0)
    one = interpolate_decoder_state(acoustic, stable, stable_weight=1.0)
    assert tensor_state_digest(zero) == tensor_state_digest(acoustic)
    assert torch.equal(one["model.encoder.weight"], acoustic["model.encoder.weight"])
    assert torch.equal(one["model.decoder.weight"], stable["model.decoder.weight"])
    assert torch.equal(one["proj_out.weight"], stable["proj_out.weight"])

    missing = dict(stable)
    missing.pop("proj_out.weight")
    with pytest.raises(ValueError, match="key mismatch"):
        interpolate_decoder_state(acoustic, missing, stable_weight=0.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        interpolate_decoder_state(acoustic, stable, stable_weight=1.1)


def _write_endpoint(path: Path, values: dict[str, torch.Tensor], config: dict[str, object]) -> None:
    path.mkdir()
    save_file(values, path / "model.safetensors", metadata={"format": "pt"})
    (path / "config.json").write_text(json.dumps(config, sort_keys=True) + "\n")


def test_interpolation_cli_core_writes_immutable_audited_checkpoint(tmp_path: Path) -> None:
    project = tmp_path / "project"
    acoustic = tmp_path / "acoustic"
    stable = tmp_path / "stable"
    output = tmp_path / "MIX15"
    project.mkdir()
    (project / "predict.py").write_text("def predict():\n    return 'ok'\n")
    config = {"model_type": "whisper", "d_model": 2, "decoder_layers": 1}
    _write_endpoint(acoustic, state(1.0, 2.0, 3.0), config)
    _write_endpoint(stable, state(9.0, 6.0, 7.0), config)
    (acoustic / "generation_config.json").write_text('{"num_beams": 2}\n')
    (acoustic / "preprocessor_config.json").write_text('{"sampling_rate": 16000}\n')
    (acoustic / "tokenizer.json").write_text("{}\n")
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "start": {
                    "config_sha256": sha256(acoustic / "config.json"),
                    "generation_config_sha256": sha256(
                        acoustic / "generation_config.json"
                    ),
                    "predict_py_sha256": sha256(project / "predict.py"),
                }
            }
        )
        + "\n"
    )

    receipt = interpolate_checkpoint(
        acoustic_checkpoint=acoustic,
        stable_decoder_checkpoint=stable,
        name="MIX15",
        stable_weight=0.15,
        output_dir=output,
        registry_path=registry,
        project_root=project,
    )

    mixed = load_file(output / "model.safetensors")
    acoustic_state = load_file(acoustic / "model.safetensors")
    assert receipt["passed"] is True
    assert receipt["output_model_sha256"] == sha256(output / "model.safetensors")
    assert torch.equal(mixed["model.encoder.weight"], acoustic_state["model.encoder.weight"])
    assert (output / "config.json").read_bytes() == (acoustic / "config.json").read_bytes()
    assert (output / "generation_config.json").read_bytes() == (
        acoustic / "generation_config.json"
    ).read_bytes()
    assert (output / "predict.py").read_bytes() == (project / "predict.py").read_bytes()
    assert all(row["passed"] for row in receipt["tensor_audit"]["sampled_formula_checks"])

    with pytest.raises(FileExistsError, match="overwrite"):
        interpolate_checkpoint(
            acoustic_checkpoint=acoustic,
            stable_decoder_checkpoint=stable,
            name="MIX15",
            stable_weight=0.15,
            output_dir=output,
            registry_path=registry,
            project_root=project,
        )


def test_candidate_name_enforces_registered_alpha(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires stable weight"):
        interpolate_checkpoint(
            acoustic_checkpoint=tmp_path / "missing-a",
            stable_decoder_checkpoint=tmp_path / "missing-b",
            name="MIX30",
            stable_weight=0.15,
            output_dir=tmp_path / "output",
            registry_path=tmp_path / "registry",
            project_root=tmp_path,
        )
