"""Pure tensor helpers for audited W500 Decoder interpolation.

The acoustic checkpoint is always the source of truth for the Encoder and for
any state key outside ``model.decoder.*`` and ``proj_out.*``.  Keeping this
policy in a small module makes it possible to audit interpolation without
loading Transformers or mutating a model object.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import torch


DECODER_PREFIX = "model.decoder."
PROJECTION_PREFIX = "proj_out."
MIXED_PREFIXES = (DECODER_PREFIX, PROJECTION_PREFIX)


def is_mixed_parameter(name: str) -> bool:
    """Return whether *name* belongs to the approved interpolation surface."""

    return name.startswith(MIXED_PREFIXES)


def validate_compatible_states(
    acoustic: Mapping[str, torch.Tensor],
    stable: Mapping[str, torch.Tensor],
) -> None:
    """Require identical keys, tensor shapes, and dtypes at both endpoints."""

    acoustic_keys = set(acoustic)
    stable_keys = set(stable)
    if acoustic_keys != stable_keys:
        missing = sorted(acoustic_keys - stable_keys)
        unexpected = sorted(stable_keys - acoustic_keys)
        raise ValueError(
            "State-dict key mismatch: "
            f"missing_from_stable={missing[:20]} "
            f"unexpected_in_stable={unexpected[:20]}"
        )
    for name in sorted(acoustic_keys):
        tensor = acoustic[name]
        other = stable[name]
        if not isinstance(tensor, torch.Tensor) or not isinstance(other, torch.Tensor):
            raise TypeError(f"State value is not a tensor: {name}")
        if tensor.shape != other.shape or tensor.dtype != other.dtype:
            raise ValueError(
                f"Tensor incompatibility: {name}; "
                f"acoustic={tuple(tensor.shape)}/{tensor.dtype}, "
                f"stable={tuple(other.shape)}/{other.dtype}"
            )


def interpolate_decoder_state(
    acoustic: Mapping[str, torch.Tensor],
    stable: Mapping[str, torch.Tensor],
    *,
    stable_weight: float,
) -> dict[str, torch.Tensor]:
    """Mix only Decoder/``proj_out`` tensors and clone all other tensors.

    Endpoint weights are special-cased so that the relevant endpoint is copied
    exactly, without relying on floating-point arithmetic to reproduce it.
    """

    alpha = float(stable_weight)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"stable_weight must be in [0, 1], got {stable_weight!r}")
    validate_compatible_states(acoustic, stable)
    output: dict[str, torch.Tensor] = {}
    for name in acoustic:
        tensor = acoustic[name]
        other = stable[name]
        if not is_mixed_parameter(name) or alpha == 0.0:
            output[name] = tensor.detach().clone()
        elif alpha == 1.0:
            output[name] = other.detach().clone()
        elif tensor.is_floating_point() or tensor.is_complex():
            output[name] = torch.lerp(tensor, other, alpha).detach().clone()
        else:
            raise ValueError(f"Cannot interpolate non-floating tensor: {name}/{tensor.dtype}")
    return output


def tensor_state_digest(
    state: Mapping[str, torch.Tensor],
    *,
    prefixes: tuple[str, ...] | None = None,
) -> str:
    """Return a deterministic digest of tensor names, metadata, and raw bytes."""

    digest = hashlib.sha256()
    selected = [
        name
        for name in sorted(state)
        if prefixes is None or name.startswith(prefixes)
    ]
    if not selected:
        raise ValueError(f"No tensors selected for digest; prefixes={prefixes!r}")
    for name in selected:
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def audit_interpolation(
    acoustic: Mapping[str, torch.Tensor],
    stable: Mapping[str, torch.Tensor],
    mixed: Mapping[str, torch.Tensor],
    *,
    stable_weight: float,
    maximum_sampled_tensors: int = 12,
) -> dict[str, Any]:
    """Audit endpoint parity, Encoder identity, scope, and sampled formulas."""

    validate_compatible_states(acoustic, stable)
    validate_compatible_states(acoustic, mixed)
    alpha = float(stable_weight)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(alpha)

    encoder_names = sorted(name for name in acoustic if name.startswith("model.encoder."))
    mixed_names = sorted(name for name in acoustic if is_mixed_parameter(name))
    if not encoder_names:
        raise ValueError("No model.encoder.* tensors found")
    if not mixed_names:
        raise ValueError("No model.decoder.* or proj_out.* tensors found")

    non_mixed_exact = all(
        torch.equal(acoustic[name], mixed[name])
        for name in acoustic
        if not is_mixed_parameter(name)
    )
    encoder_exact = all(torch.equal(acoustic[name], mixed[name]) for name in encoder_names)

    alpha_zero = interpolate_decoder_state(acoustic, stable, stable_weight=0.0)
    alpha_one = interpolate_decoder_state(acoustic, stable, stable_weight=1.0)
    endpoint_zero_exact = all(
        torch.equal(alpha_zero[name], acoustic[name]) for name in acoustic
    )
    endpoint_one_exact = all(
        torch.equal(
            alpha_one[name],
            stable[name] if is_mixed_parameter(name) else acoustic[name],
        )
        for name in acoustic
    )

    stride = max(1, len(mixed_names) // max(1, int(maximum_sampled_tensors)))
    sampled_names = mixed_names[::stride][:maximum_sampled_tensors]
    formula_checks: list[dict[str, Any]] = []
    for name in sampled_names:
        source_flat = acoustic[name].detach().cpu().reshape(-1)
        stable_flat = stable[name].detach().cpu().reshape(-1)
        mixed_flat = mixed[name].detach().cpu().reshape(-1)
        if source_flat.numel() == 0:
            continue
        indices = sorted({0, source_flat.numel() // 2, source_flat.numel() - 1})
        for index in indices:
            source_value = source_flat[index]
            stable_value = stable_flat[index]
            if alpha == 0.0:
                expected = source_value
            elif alpha == 1.0:
                expected = stable_value
            else:
                expected = torch.lerp(source_value, stable_value, alpha)
            actual = mixed_flat[index]
            passed = bool(torch.equal(actual, expected))
            formula_checks.append(
                {
                    "tensor": name,
                    "flat_index": int(index),
                    "acoustic": float(source_value),
                    "stable": float(stable_value),
                    "expected": float(expected),
                    "actual": float(actual),
                    "passed": passed,
                }
            )

    checks = {
        "state_keys_shapes_dtypes_identical": True,
        "encoder_exact": encoder_exact,
        "all_non_mixed_tensors_exact": non_mixed_exact,
        "alpha_0_reproduces_acoustic_endpoint": endpoint_zero_exact,
        "alpha_1_reproduces_stable_decoder_with_acoustic_encoder": endpoint_one_exact,
        "sampled_formula_checks_passed": all(row["passed"] for row in formula_checks),
    }
    if not all(checks.values()):
        raise ValueError(f"Interpolation audit failed: {checks}")
    return {
        "passed": True,
        "stable_weight": alpha,
        "checks": checks,
        "tensor_counts": {
            "all": len(acoustic),
            "encoder": len(encoder_names),
            "mixed": len(mixed_names),
            "decoder": sum(name.startswith(DECODER_PREFIX) for name in mixed_names),
            "proj_out": sum(name.startswith(PROJECTION_PREFIX) for name in mixed_names),
        },
        "digests": {
            "acoustic_full": tensor_state_digest(acoustic),
            "mixed_full": tensor_state_digest(mixed),
            "acoustic_encoder": tensor_state_digest(
                acoustic, prefixes=("model.encoder.",)
            ),
            "mixed_encoder": tensor_state_digest(mixed, prefixes=("model.encoder.",)),
            "acoustic_decoder_proj_out": tensor_state_digest(
                acoustic, prefixes=MIXED_PREFIXES
            ),
            "stable_decoder_proj_out": tensor_state_digest(
                stable, prefixes=MIXED_PREFIXES
            ),
            "mixed_decoder_proj_out": tensor_state_digest(
                mixed, prefixes=MIXED_PREFIXES
            ),
        },
        "sampled_formula_checks": formula_checks,
    }
