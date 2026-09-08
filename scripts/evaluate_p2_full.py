#!/usr/bin/env python3
"""Offline surface evaluation, separate from distributed training ranks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from transformers import WhisperProcessor
from cantonese_asr.io import read_jsonl, sha256_file
from cantonese_asr.model_loading import load_whisper_model
from cantonese_asr.p2_full import atomic_json
from scripts.evaluate_raw_winner_decode import evaluate_surface
from scripts.evaluate_raw_winner_p2_capacity import hash_weights


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--asset-root", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--processor-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--surface", choices=("validation", "public", "ood"), required=True)
    p.add_argument("--decoder", choices=("D0_CURRENT", "P2_NATIVE5"), default="D0_CURRENT")
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-samples", type=int)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    cfg = json.loads(a.config.read_text())
    spec = cfg["surfaces"][a.surface]
    manifest = a.asset_root / spec["path"]
    if sha256_file(manifest) != spec["sha256"]:
        raise ValueError("evaluation manifest mismatch")
    rows = read_jsonl(manifest)
    if len(rows) != spec["rows"] or any(spec["reference_field"] not in row for row in rows):
        raise ValueError("invalid reference rows/field")
    initial_weights = hash_weights(a.model_dir)
    processor = WhisperProcessor.from_pretrained(a.processor_dir, language="zh", task="transcribe", local_files_only=True)
    model = load_whisper_model(a.model_dir, dtype=torch.float16).cuda().eval()
    model.generation_config.max_new_tokens = None
    model.generation_config.language = "zh"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = True
    model.generation_config.use_cache = True
    prompt_count = 1 + len(processor.get_decoder_prompt_ids(language="zh", task="transcribe", no_timestamps=True))
    summary = evaluate_surface(surface=a.surface, manifest=manifest, reference_field=spec["reference_field"],
        model=model, processor=processor, arm=a.decoder, project_root=a.asset_root, output_dir=a.output_dir,
        batch_size=a.batch_size, generation_max_length=225, prompt_token_count=prompt_count,
        device=torch.device("cuda"), dtype=torch.float16, max_samples=a.max_samples,
        mbr_anchor_dir=None, diagnostic_version=2)
    expected = min(a.max_samples, spec["rows"]) if a.max_samples else spec["rows"]
    if any(len(read_jsonl(a.output_dir / name)) != expected for name in ("predictions.jsonl", "generation_tokens.jsonl")):
        raise ValueError("output row mismatch")
    if hash_weights(a.model_dir) != initial_weights:
        raise ValueError("model weight surface changed during evaluation")
    val_loss = None
    if a.surface == "validation":
        from train import ManifestDataset, SpeechSeq2SeqCollator
        data = ManifestDataset(manifest, processor, a.asset_root, max_samples=a.max_samples)
        collate = SpeechSeq2SeqCollator(processor)
        loss_sum, tokens = 0.0, 0
        with torch.inference_mode():
            for i in range(len(data)):
                item = collate([data[i]])
                tensors = {k: v.cuda() for k, v in item.items() if not k.startswith("_")}
                tensors["input_features"] = tensors["input_features"].half()
                result = model(**tensors)
                count = int(tensors["labels"].ne(-100).sum())
                loss_sum += float(result.loss) * count
                tokens += count
        val_loss = loss_sum / tokens
    generation = summary["generation"]
    evidence = {"surface": a.surface, "complete": a.max_samples is None,
        "integrity_pass": a.max_samples is None, "step": a.step,
        "tol2": summary["metrics"]["sentence_accuracy_tol2"], "cer": summary["metrics"]["cer"],
        "severe": summary["severe_error_count"], "repeated_runaway": generation["repeated_runaway_count"],
        "replacement": generation["replacement_character_count"], "max_length": generation["effective_max_length_count"],
        "validation_loss": val_loss, "model_dir": str(a.model_dir.resolve()),
        "weights": initial_weights, "config_sha256": sha256_file(a.config),
        "decoder": a.decoder, "manifest_sha256": spec["sha256"],
        "model_generation_config": model.generation_config.to_dict(),
        "effective_generate_kwargs": summary["effective_generate_kwargs"],
        "files": [{"path": path.name, "sha256": sha256_file(path)} for path in sorted(a.output_dir.iterdir()) if path.is_file()]}
    atomic_json(a.output_dir / "evaluation_receipt.json", evidence)
    print(json.dumps({"surface": a.surface, "step": a.step, "cer": evidence["cer"], "tol2": evidence["tol2"]}))


if __name__ == "__main__":
    main()
