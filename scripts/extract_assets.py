#!/usr/bin/env python3
"""Safely and idempotently extract official train and public-test ZIP files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cantonese_asr.io import sha256_file


LEADING_ID = re.compile(r"^(\d+)")
SAFE_COMPONENT_BYTES = 240


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("artifacts/datasets/official")
    )
    parser.add_argument(
        "--train-dir", type=Path, default=Path("artifacts/data/train_raw")
    )
    parser.add_argument(
        "--test-dir", type=Path, default=Path("artifacts/data/test_audio")
    )
    return parser.parse_args()


def validate_member(info: zipfile.ZipInfo) -> None:
    path = PurePosixPath(info.filename)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe ZIP member: {info.filename}")
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ValueError(f"ZIP symlink is not allowed: {info.filename}")


def repair_utf8_name_without_flag(info: zipfile.ZipInfo) -> bool:
    """Repair UTF-8 filename bytes that ZIP metadata mislabeled as CP437."""
    if info.flag_bits & 0x800:
        return False
    try:
        repaired = info.filename.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False
    if repaired == info.filename:
        return False
    info.filename = repaired
    return True


def shorten_overlong_components(info: zipfile.ZipInfo) -> bool:
    """Keep numeric IDs while making path components portable to Linux."""
    original_parts = PurePosixPath(info.filename).parts
    changed = False
    safe_parts = []
    for part in original_parts:
        if len(part.encode("utf-8")) <= SAFE_COMPONENT_BYTES:
            safe_parts.append(part)
            continue
        changed = True
        suffix = Path(part).suffix
        match = LEADING_ID.match(part)
        prefix = match.group(1) if match else "member"
        digest = hashlib.sha256(part.encode("utf-8")).hexdigest()[:16]
        safe_parts.append(f"{prefix}__longname_{digest}{suffix}")
    if changed:
        repaired = PurePosixPath(*safe_parts).as_posix()
        info.filename = repaired + ("/" if info.is_dir() else "")
    return changed


def is_platform_metadata(info: zipfile.ZipInfo) -> bool:
    parts = PurePosixPath(info.filename).parts
    return bool(
        "__MACOSX" in parts
        or any(part.startswith("._") for part in parts)
        or any(part == ".DS_Store" for part in parts)
    )


def extract(zip_path: Path, destination: Path) -> dict[str, object]:
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    archive_sha = sha256_file(zip_path)
    marker = destination / ".extraction.json"
    if marker.is_file():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous.get("archive_sha256") == archive_sha:
            return {**previous, "status": "already_extracted"}
        raise RuntimeError(
            f"{destination} contains a different archive extraction; move it aside before retrying"
        )
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(
            f"Destination is non-empty without a matching marker: {destination}"
        )
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Corrupt member in {zip_path}: {bad}")
        infos = archive.infolist()
        repaired_name_count = 0
        shortened_name_count = 0
        for info in infos:
            repaired_name_count += int(repair_utf8_name_without_flag(info))
            shortened_name_count += int(shorten_overlong_components(info))
            validate_member(info)
        selected = [info for info in infos if not is_platform_metadata(info)]
        destination.mkdir(parents=True, exist_ok=True)
        archive.extractall(destination, members=selected)
    report = {
        "archive": str(zip_path.resolve()),
        "archive_sha256": archive_sha,
        "destination": str(destination.resolve()),
        "archive_member_count": len(infos),
        "extracted_member_count": len(selected),
        "skipped_platform_metadata_count": len(infos) - len(selected),
        "repaired_utf8_filename_count": repaired_name_count,
        "shortened_overlong_filename_count": shortened_name_count,
        "status": "extracted",
    }
    marker.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    reports = {
        "train": extract(args.dataset_dir / "data.zip", args.train_dir),
        "public_test": extract(args.dataset_dir / "test_audio.zip", args.test_dir),
    }
    report_dir = Path("artifacts/reports/deployment")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "extraction_report.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
