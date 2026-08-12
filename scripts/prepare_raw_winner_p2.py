#!/usr/bin/env python3
"""Freeze the P2 exposure stream, topology receipts, and model file hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from cantonese_asr.io import read_jsonl, sha256_file
from cantonese_asr.training_sampling import fixed_exposure_topology_receipt


def sample_id(row: dict, index: int) -> str:
    return str(row.get("id") or row.get("audio_path") or f"row-{index}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def model_hashes(path: Path) -> dict:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    return {
        "path": str(path.resolve()),
        "files": [
            {"path": str(item.relative_to(path)), "bytes": item.stat().st_size, "sha256": sha256_file(item)}
            for item in files
        ],
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = args.project_root / config["external73"]["path"]
    expected_sha = config["external73"]["sha256"]
    if sha256_file(source) != expected_sha:
        raise SystemExit("external73 SHA-256 mismatch")
    rows = read_jsonl(source)
    count = int(config["external73"]["fixed_exposure_rows"])
    rng = random.Random(int(config["seed"]))
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    selected_indices = indices[:count]
    selected = []
    ids = []
    seen = set()
    for index in selected_indices:
        row = dict(rows[index])
        identifier = sample_id(row, index)
        if identifier in seen:
            raise SystemExit(f"duplicate exposure ID: {identifier}")
        seen.add(identifier)
        row["_p2_source_row_index"] = index
        row["_p2_sample_id"] = identifier
        selected.append(row)
        ids.append(identifier)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "external73_fixed_exposure_2400.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    topologies = {
        name: fixed_exposure_topology_receipt(ids, world_size=world, per_device_batch=batch, gradient_accumulation_steps=accum)
        for name, world, batch, accum in (
            ("1gpu", 1, 8, 2),
            ("2gpu", 2, 2, 4),
            ("4gpu", 4, 1, 4),
        )
    }
    digests = {value["global_step_ids_sha256"] for value in topologies.values()}
    if len(digests) != 1:
        raise SystemExit("topology global exposure mismatch")
    receipt = {
        "source": str(source.resolve()),
        "source_sha256": expected_sha,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "seed": config["seed"],
        "rows": len(selected),
        "unique_ids": len(seen),
        "topology_global_ids_sha256": next(iter(digests)),
        "topologies": topologies,
    }
    (args.output_dir / "fixed_exposure_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hashes = {}
    for name, relative in config["models"].items():
        path = args.project_root / relative
        if path.is_dir():
            hashes[name] = model_hashes(path)
    (args.output_dir / "model_file_hashes.json").write_text(json.dumps(hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest), "sha256": receipt["manifest_sha256"], "topology_parity": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
