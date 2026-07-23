from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from scripts.extract_assets import (
    SAFE_COMPONENT_BYTES,
    extract,
    repair_utf8_name_without_flag,
    shorten_overlong_components,
)


def test_repairs_utf8_bytes_mislabeled_as_cp437() -> None:
    expected = "10235听多啲咯.wav"
    info = zipfile.ZipInfo(expected.encode("utf-8").decode("cp437"))
    info.flag_bits = 0

    assert repair_utf8_name_without_flag(info)
    assert info.filename == expected


def test_shortens_overlong_name_but_preserves_id_and_extension() -> None:
    info = zipfile.ZipInfo("folder/10235" + "粤" * 100 + ".wav")
    original = info.filename

    assert shorten_overlong_components(info)
    component = Path(info.filename).name
    assert component.startswith("10235__longname_")
    assert component.endswith(".wav")
    assert len(component.encode("utf-8")) <= SAFE_COMPONENT_BYTES

    same = zipfile.ZipInfo(original)
    assert shorten_overlong_components(same)
    assert same.filename == info.filename


def test_extract_skips_macos_metadata_and_is_idempotent(tmp_path: Path) -> None:
    archive = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("audio/00001.wav", b"real-audio-placeholder")
        handle.writestr("__MACOSX/audio/._00001.wav", b"resource-fork")
        handle.writestr("audio/.DS_Store", b"finder")
    destination = tmp_path / "output"

    first = extract(archive, destination)
    assert (destination / "audio/00001.wav").is_file()
    assert not (destination / "__MACOSX").exists()
    assert not (destination / "audio/.DS_Store").exists()
    assert first["archive_member_count"] == 3
    assert first["extracted_member_count"] == 1
    assert first["skipped_platform_metadata_count"] == 2

    second = extract(archive, destination)
    assert second["status"] == "already_extracted"


def test_extract_rejects_path_traversal_before_creating_destination(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", b"no")
    destination = tmp_path / "output"

    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        extract(archive, destination)
    assert not destination.exists()
