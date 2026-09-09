#!/usr/bin/env python3
"""CPU preparation and deterministic selectors for full P2; never submits jobs."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cantonese_asr.io import read_jsonl, sha256_file
from cantonese_asr.p2_full import atomic_json, epoch_batches, extension_decision, split_lm_texts


def prepare(a):
    import hashlib
    import librosa
    import numpy as np
    from cantonese_asr.metrics import normalize_prediction
    from cantonese_asr.training_sampling import source_group
    from scripts.evaluate_raw_winner_decode import resolve_surface_audio

    cfg = json.loads(a.config.read_text())
    if a.output_dir.exists():
        raise FileExistsError("preparation must use a new immutable directory")
    a.output_dir.mkdir(parents=True)
    sources = {"train": {**cfg["external73"], "reference_field": "text"}, **cfg["surfaces"]}
    manifests = {}
    fingerprints = {}
    audio_receipts = []
    cache = {}
    for split, spec in sources.items():
        manifest = a.asset_root / spec["path"]
        if sha256_file(manifest) != spec["sha256"]:
            raise ValueError(f"{split}: manifest hash mismatch")
        rows = read_jsonl(manifest)
        if len(rows) != spec["rows"]:
            raise ValueError(f"{split}: row count mismatch")
        manifests[split] = rows
        fingerprints[split] = set()
        for index, row in enumerate(rows):
            if spec["reference_field"] not in row or not str(row[spec["reference_field"]]).strip():
                raise ValueError(f"{split}:{index}: missing reference")
            path = (a.asset_root / row["audio_path"]).resolve()
            if not path.is_file() and split == "public":
                path = resolve_surface_audio(a.asset_root, row["audio_path"]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            if path not in cache:
                item = {"resolved_path": str(path), "file_sha256": sha256_file(path)}
                if a.decode_audio:
                    audio, rate = librosa.load(path, sr=16000, mono=True)
                    if rate != 16000 or not len(audio) or not np.isfinite(audio).all():
                        raise ValueError(f"invalid audio: {path}")
                    item.update(duration_seconds=len(audio) / rate,
                        pcm_sha256=hashlib.sha256(np.asarray(audio, dtype="<f4").tobytes()).hexdigest())
                cache[path] = item
            item = {"split": split, "row_index": index, "listed_path": row["audio_path"], **cache[path]}
            fingerprints[split].add(item.get("pcm_sha256", item["file_sha256"]))
            audio_receipts.append(item)
        print(json.dumps({"checked_surface": split, "rows": len(rows)}), flush=True)
    overlaps = {split: len(fingerprints["train"] & fingerprints[split]) for split in cfg["surfaces"]}
    if any(overlaps.values()):
        atomic_json(a.output_dir / "FAILED_OVERLAP.json", overlaps)
        raise ValueError("training/evaluation audio overlap; do not modify frozen manifests")
    rows = manifests["train"]
    counts = dict(Counter(source_group(row) for row in rows))
    if counts != cfg["external73"]["source_counts"]:
        raise ValueError(f"source counts differ: {counts}")
    ids = [str(row.get("id") or row["audio_path"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate training IDs; cannot silently drop rows")
    streams = {}
    for seed in cfg["seeds"]:
        path = a.output_dir / f"stream_s{seed}.json"
        batches = epoch_batches(len(rows), seed)
        atomic_json(path, batches)
        streams[str(seed)] = {"path": path.name, "sha256": sha256_file(path)}
        # Six unique global batches: random coverage plus the longest transcripts
        # and waveforms. This is smoke-only, never the formal exposure stream.
        chosen = [i for b in batches[:4] for i in b["indices"]]
        duration = {r["row_index"]: r.get("duration_seconds", 0) for r in audio_receipts if r["split"] == "train"}
        long_text = sorted(range(len(rows)), key=lambda i: (-len(str(rows[i]["text"])), i))
        long_audio = sorted(range(len(rows)), key=lambda i: (-duration[i], i))
        priorities = long_text[:16] + long_audio[:16] + long_text[16:64] + long_audio[16:64]
        for i in priorities:
            if i not in chosen and len(chosen) < 96:
                chosen.append(i)
        if len(chosen) != 96:
            raise ValueError("unable to construct 96 unique smoke samples")
        smoke = [{"step": offset//16+1, "epoch": 0, "offset": offset, "indices": chosen[offset:offset+16]}
                 for offset in range(0, 96, 16)]
        smoke_path = a.output_dir / f"smoke_s{seed}.json"
        atomic_json(smoke_path, smoke)
        streams[str(seed)]["smoke_sha256"] = sha256_file(smoke_path)
    texts = [normalize_prediction(str(row["text"])) for row in rows]
    atomic_json(a.output_dir / "lm_split.json", split_lm_texts(texts))
    models = {}
    from transformers import WhisperTokenizer
    for name, relative in cfg["models"].items():
        model = a.asset_root / relative
        if not model.is_dir() or not (model / "config.json").is_file():
            raise FileNotFoundError(model)
        weights = list(model.glob("*.safetensors"))
        if not weights:
            raise ValueError(f"missing safetensors: {name}")
        tokenizer = WhisperTokenizer.from_pretrained(model, language="zh", task="transcribe", local_files_only=True)
        model_config = json.loads((model / "config.json").read_text())
        lengths = [len(tokenizer(str(row["text"])).input_ids) - 1 for row in rows]
        oversized = [i for i, length in enumerate(lengths) if length > model_config["max_target_positions"]]
        if oversized:
            atomic_json(a.output_dir / f"FAILED_LABEL_LENGTH_{name}.json", {"indices": oversized, "lengths": [lengths[i] for i in oversized]})
            raise ValueError("training labels exceed decoder capacity; no silent truncation")
        models[name] = {"path": str(model.resolve()), "files": [
            {"path": str(path.relative_to(model)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(model.rglob("*")) if path.is_file()]}
    atomic_json(a.output_dir / "audio_receipt.json", audio_receipts)
    atomic_json(a.output_dir / "model_hashes.json", models)
    atomic_json(a.output_dir / "preflight.json", {"status": "PASS" if a.decode_audio else "INCOMPLETE_AUDIO_DECODE",
        "config_sha256": sha256_file(a.config), "source_counts": counts, "sample_ids": ids,
        "audio_overlap_counts": overlaps, "audio_decode_checked": a.decode_audio,
        "audio_receipt_sha256": sha256_file(a.output_dir / "audio_receipt.json"),
        "lm_split_sha256": sha256_file(a.output_dir / "lm_split.json"),
        "model_hashes_sha256": sha256_file(a.output_dir / "model_hashes.json"), "streams": streams})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--asset-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--decode-audio", action="store_true", help="mandatory for PASS; expensive CPU audit")
    p = subs.add_parser("select-extension")
    p.add_argument("--arm", required=True)
    p.add_argument("--validation-receipts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = subs.add_parser("status")
    p.add_argument("--run-root", type=Path, required=True)
    a = parser.parse_args()
    if a.command == "prepare":
        prepare(a)
    elif a.command == "select-extension":
        if a.output.exists():
            raise FileExistsError(a.output)
        raw = json.loads(a.validation_receipts.read_text())
        result = extension_decision({int(seed): rows for seed, rows in raw.items()})
        atomic_json(a.output, {"arm": a.arm, "input_sha256": sha256_file(a.validation_receipts), **result})
    else:
        print(json.dumps({"complete_checkpoints": [str(p.parent) for p in a.run_root.rglob("COMPLETE.json")],
                          "incomplete_staging": [str(p) for p in a.run_root.rglob("*.staging")]}, indent=2))


if __name__ == "__main__":
    main()
