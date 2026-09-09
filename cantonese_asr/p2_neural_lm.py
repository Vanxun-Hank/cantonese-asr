"""Offline character LM. Vocabulary and training input never depend on ASR refs."""
from __future__ import annotations
import torch
from torch import nn

SPECIALS = ("<PAD>", "<BOS>", "<EOS>", "<UNK>")


def vocabulary_from_train(texts: list[str]) -> list[str]:
    return list(SPECIALS) + sorted(set("".join(texts)))


def encode(text: str, vocabulary: list[str]) -> list[int]:
    lookup = {token: i for i, token in enumerate(vocabulary)}
    return [lookup["<BOS>"]] + [lookup.get(char, lookup["<UNK>"]) for char in text] + [lookup["<EOS>"]]


def collate_texts(texts: list[str], vocabulary: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [encode(text, vocabulary) for text in texts]
    width = max(len(row) - 1 for row in encoded)
    inputs = torch.zeros((len(texts), width), dtype=torch.long)
    targets = torch.full((len(texts), width), -100, dtype=torch.long)
    for i, row in enumerate(encoded):
        inputs[i, :len(row) - 1] = torch.tensor(row[:-1])
        targets[i, :len(row) - 1] = torch.tensor(row[1:])
    return inputs, targets


class CharacterLSTM(nn.Module):
    def __init__(self, vocabulary_size: int, embedding: int = 256, hidden: int = 512,
                 layers: int = 2, dropout: float = .2):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding, padding_idx=0)
        self.recurrent = nn.LSTM(embedding, hidden, num_layers=layers,
                                 dropout=dropout if layers > 1 else 0, batch_first=True)
        self.output = nn.Linear(hidden, vocabulary_size)

    def forward(self, tokens):
        sequence, _ = self.recurrent(self.embedding(tokens))
        return self.output(sequence)


@torch.inference_mode()
def score_texts(model: CharacterLSTM, texts: list[str], vocabulary: list[str], device,
                batch_size: int = 64) -> list[float]:
    """Sentence log P(chars,EOS | BOS), divided by max(1, character count)."""
    model.eval()
    scores = []
    for start in range(0, len(texts), batch_size):
        current = texts[start:start + batch_size]
        inputs, targets = collate_texts(current, vocabulary)
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        losses = nn.functional.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
        scores.extend(-float(losses[i].sum()) / max(1, len(text)) for i, text in enumerate(current))
    return scores
