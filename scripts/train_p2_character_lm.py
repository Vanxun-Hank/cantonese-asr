#!/usr/bin/env python3
"""Train one registered offline LSTM seed on the frozen train-only LM split."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from cantonese_asr.io import sha256_file
from cantonese_asr.p2_full import atomic_json
from cantonese_asr.p2_neural_lm import CharacterLSTM, collate_texts, score_texts, vocabulary_from_train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=(42, 43), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    receipt = json.loads((a.preparation / "preflight.json").read_text())
    split_path = a.preparation / "lm_split.json"
    if receipt["status"] != "PASS" or receipt["config_sha256"] != sha256_file(a.config):
        raise ValueError("unverified preparation/config")
    if sha256_file(split_path) != receipt["lm_split_sha256"]:
        raise ValueError("LM split changed")
    split = json.loads(split_path.read_text())
    if set(split["train"]) & set(split["dev"]):
        raise ValueError("LM train/dev text leakage")
    if not torch.cuda.is_available():
        raise RuntimeError("formal neural LM uses an authorized free GPU")
    a.output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    neural = cfg["lm"]["neural"]
    vocabulary = vocabulary_from_train(split["train"])
    atomic_json(a.output_dir / "vocabulary.json", vocabulary)
    architecture = {key: neural[key] for key in ("embedding", "hidden", "layers", "dropout")}
    model = CharacterLSTM(len(vocabulary), **architecture).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=neural["lr"], weight_decay=neural["weight_decay"])
    best_nll = math.inf
    best_epoch = None
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, neural["epochs"] + 1):
        indices = list(range(len(split["train"])))
        random.Random(f"p2-lm:{a.seed}:{epoch}").shuffle(indices)
        model.train()
        train_loss = 0.0
        train_events = 0
        for start in range(0, len(indices), neural["batch"]):
            texts = [split["train"][i] for i in indices[start:start + neural["batch"]]]
            inputs, targets = collate_texts(texts, vocabulary)
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite neural LM loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), neural["clip"])
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite neural LM gradient")
            optimizer.step()
            events = int(targets.ne(-100).sum())
            train_events += events
            train_loss += float(loss.detach()) * events
        scores = score_texts(model, split["dev"], vocabulary, torch.device("cuda"))
        dev_chars = sum(max(1, len(text)) for text in split["dev"])
        dev_nll = -sum(score * max(1, len(text)) for score, text in zip(scores, split["dev"])) / dev_chars
        record = {"epoch": epoch, "train_nll_per_event": train_loss / train_events,
                  "dev_nll_per_character": dev_nll, "dev_perplexity_per_character": math.exp(dev_nll),
                  "elapsed_seconds": time.perf_counter() - started}
        with (a.output_dir / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        checkpoint = a.output_dir / f"epoch-{epoch}.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "seed": a.seed, "architecture": architecture,
                    "vocabulary": vocabulary, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(), "split_sha256": sha256_file(split_path)}, checkpoint)
        if dev_nll < best_nll:
            best_nll, best_epoch = dev_nll, epoch
        print(json.dumps(record), flush=True)
    selected = a.output_dir / f"epoch-{best_epoch}.pt"
    vocabulary_set = set(vocabulary)
    dev_total = sum(len(text) for text in split["dev"])
    atomic_json(a.output_dir / "COMPLETE.json", {"selected_epoch": best_epoch,
        "selected_checkpoint": str(selected.resolve()), "sha256": sha256_file(selected),
        "selection_surface": "LM-dev", "dev_nll_per_character": best_nll,
        "dev_oov_rate": sum(char not in vocabulary_set for text in split["dev"] for char in text) / max(1, dev_total),
        "seed": a.seed, "parameters": sum(p.numel() for p in model.parameters()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "runtime_seconds": time.perf_counter() - started, "config_sha256": sha256_file(a.config),
        "split_sha256": sha256_file(split_path)})


if __name__ == "__main__":
    main()
