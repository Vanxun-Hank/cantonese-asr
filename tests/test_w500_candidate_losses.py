from __future__ import annotations

import pytest
import torch

from scripts.evaluate_w500_candidate_losses import per_sample_losses


def test_per_sample_losses_normalize_by_tokens_and_characters() -> None:
    logits = torch.tensor(
        [
            [[4.0, 0.0], [0.0, 4.0], [9.0, -9.0]],
            [[0.0, 4.0], [4.0, 0.0], [0.0, 4.0]],
        ]
    )
    labels = torch.tensor([[0, 1, -100], [1, 0, 1]])
    rows = per_sample_losses(logits, labels, [1, 6])
    assert rows[0]["target_tokens"] == 2
    assert rows[1]["target_tokens"] == 3
    assert rows[0]["normalized_loss"] == pytest.approx(
        rows[0]["token_mean_loss"] * 2
    )
    assert rows[1]["normalized_loss"] == pytest.approx(
        rows[1]["token_mean_loss"] * 0.5
    )


def test_per_sample_losses_reject_zero_character_count() -> None:
    logits = torch.zeros((1, 1, 2))
    labels = torch.zeros((1, 1), dtype=torch.long)
    with pytest.raises(ValueError, match="normalization counts"):
        per_sample_losses(logits, labels, [0])
