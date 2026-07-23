from __future__ import annotations

from pathlib import Path

from scripts import download_assets


class FakeResponse:
    def __init__(self, body: bytes, status_code: int) -> None:
        self.body = body
        self.status_code = status_code

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        del chunk_size
        yield self.body


def test_model_selection_excludes_other_framework_weights() -> None:
    rows = [
        {"path": "model.safetensors"},
        {"path": "pytorch_model.bin"},
        {"path": "flax_model.msgpack"},
        {"path": "tokenizer.json"},
    ]
    selected = download_assets.selected_rows(
        rows, download_assets.MODEL_ALLOW_PATTERNS
    )
    assert [row["path"] for row in selected] == [
        "model.safetensors",
        "tokenizer.json",
    ]


def test_download_file_resumes_when_server_returns_206(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "asset.bin"
    partial = tmp_path / "asset.bin.direct-part"
    partial.write_bytes(b"abcd")

    def fake_get(*args, **kwargs):
        assert kwargs["headers"]["Range"] == "bytes=4-"
        return FakeResponse(b"efghij", 206)

    monkeypatch.setattr(download_assets.requests, "get", fake_get)
    result = download_assets.download_file(
        "https://mirror.invalid",
        "org/repo",
        "model",
        "deadbeef",
        {"path": "asset.bin", "size": 10},
        tmp_path,
        1,
    )
    assert result == "downloaded"
    assert target.read_bytes() == b"abcdefghij"
    assert not partial.exists()


def test_download_file_restarts_when_range_is_ignored(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "asset.bin"
    partial = tmp_path / "asset.bin.direct-part"
    partial.write_bytes(b"stale")

    def fake_get(*args, **kwargs):
        assert kwargs["headers"]["Range"] == "bytes=5-"
        return FakeResponse(b"fresh-data", 200)

    monkeypatch.setattr(download_assets.requests, "get", fake_get)
    result = download_assets.download_file(
        "https://mirror.invalid",
        "org/repo",
        "dataset",
        "deadbeef",
        {"path": "asset.bin", "size": 10},
        tmp_path,
        1,
    )
    assert result == "downloaded"
    assert target.read_bytes() == b"fresh-data"
    assert not partial.exists()
