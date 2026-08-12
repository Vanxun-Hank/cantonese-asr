#!/usr/bin/env python3
"""Audit Whisper tokenizer behavior on the four registered Cantonese surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import WhisperTokenizer

from cantonese_asr.io import read_jsonl, sha256_file

KEY_CHARS = "唔冇喺咗嘅啲佢嚟咁"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--official-manifest", type=Path, required=True)
    p.add_argument("--validation-manifest", type=Path, required=True)
    p.add_argument("--public-manifest", type=Path, required=True)
    p.add_argument("--ood-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def row_text(row: dict) -> str:
    for key in ("text", "ref_text", "text:label", "reference"):
        if row.get(key) is not None:
            return str(row[key])
    raise ValueError("manifest row has no reference text field")


def tokenizer_hash(path: Path) -> str:
    files = [path / name for name in ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json") if (path / name).is_file()]
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode()); digest.update(sha256_file(file).encode())
    return digest.hexdigest()


def audit(tokenizer: WhisperTokenizer, texts: list[str]) -> dict:
    chars = sum(len(text) for text in texts)
    token_rows = [tokenizer(text, add_special_tokens=False).input_ids for text in texts]
    tokens = sum(len(ids) for ids in token_rows)
    all_characters = [character for text in texts for character in text]
    multi = sum(len(tokenizer(character, add_special_tokens=False).input_ids) > 1 for character in all_characters)
    unk = tokenizer.unk_token_id
    return {
        "samples": len(texts), "characters": chars, "tokens": tokens,
        "characters_per_token": chars / max(tokens, 1),
        "single_character_multi_token_rate": multi / max(len(all_characters), 1),
        "unknown_token_count": sum(ids.count(unk) for ids in token_rows) if unk is not None else 0,
        "replacement_character_count": sum(text.count("�") for text in texts),
        "control_or_surrogate_count": sum(unicodedata.category(ch) in {"Cc", "Cs"} and ch not in "\n\r\t" for ch in all_characters),
        "nfc_changed": sum(unicodedata.normalize("NFC", text) != text for text in texts),
        "nfkc_changed": sum(unicodedata.normalize("NFKC", text) != text for text in texts),
    }


def main() -> None:
    args = parse_args(); config = json.loads(args.config.read_text())
    manifests = {"official_train": args.official_manifest, "validation": args.validation_manifest, "public": args.public_manifest, "ood": args.ood_manifest}
    texts = {name: [row_text(row) for row in read_jsonl(path)] for name, path in manifests.items()}
    result = {"manifest_sha256": {name: sha256_file(path) for name, path in manifests.items()}, "models": {}}
    for model_name, relative in config["models"].items():
        path = args.project_root / relative
        tokenizer = WhisperTokenizer.from_pretrained(path, language="zh", task="transcribe", local_files_only=True)
        result["models"][model_name] = {
            "path": str(path.resolve()), "tokenizer_hash": tokenizer_hash(path),
            "key_character_encodings": {ch: tokenizer(ch, add_special_tokens=False).input_ids for ch in KEY_CHARS},
            "surfaces": {name: audit(tokenizer, values) for name, values in texts.items()},
        }
    signatures = {(item["tokenizer_hash"], json.dumps(item["key_character_encodings"], sort_keys=True)) for item in result["models"].values()}
    result["tokenizers_identical"] = len(signatures) == 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tokenizers_identical": result["tokenizers_identical"], "output": str(args.output)}, indent=2))


if __name__ == "__main__": main()
