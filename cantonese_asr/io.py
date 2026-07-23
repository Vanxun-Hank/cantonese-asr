"""Small, dependency-free I/O helpers shared by training and evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_no}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_test_rows(path: Path) -> list[dict[str, Any]]:
    """Read the official CSV list or the public JSONL list without reordering."""
    if path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
        for line_no, row in enumerate(rows, start=1):
            if not str(row.get("audio_path", "")).strip():
                raise ValueError(f"Missing audio_path at {path}:{line_no}")
        return rows

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Empty test list: {path}")
        audio_key = next(
            (
                key
                for key in reader.fieldnames
                if key.strip().lower() in {"audio_path", "audio:file"}
            ),
            None,
        )
        if audio_key is None:
            raise ValueError(f"No audio_path column in {path}: {reader.fieldnames}")
        rows = []
        for line_no, row in enumerate(reader, start=2):
            audio_path = str(row.get(audio_key, "")).strip()
            if not audio_path:
                raise ValueError(f"Missing audio_path at {path}:{line_no}")
            normalized = dict(row)
            normalized["audio_path"] = audio_path
            rows.append(normalized)
    return rows


def resolve_audio_path(audio_dir: Path, listed_path: str) -> Path:
    candidate = Path(listed_path)
    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    candidates.extend((audio_dir / candidate, audio_dir / candidate.name))
    for path in candidates:
        if path.is_file():
            return path
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Audio not found for {listed_path!r}; tried: {rendered}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(item) for item in paths), key=lambda item: str(item)):
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()

