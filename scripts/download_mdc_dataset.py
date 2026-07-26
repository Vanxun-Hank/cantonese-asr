#!/usr/bin/env python3
"""Download a Mozilla Data Collective archive with resume and retries.

The API key is read from ``MDC_API_KEY`` and is never written to logs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests


DEFAULT_API_URL = "https://mozilladatacollective.com/api"
USER_AGENT = "cantonese-asr/1.0 mdc-downloader"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--api-url",
        default=os.environ.get("MDC_API_URL", DEFAULT_API_URL),
    )
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--progress-mib", type=int, default=128)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel HTTP range workers. Use 1 for the original serial downloader.",
    )
    parser.add_argument("--segment-mib", type=int, default=64)
    return parser.parse_args()


def content_total(response: requests.Response, offset: int) -> int | None:
    content_range = response.headers.get("Content-Range", "")
    match = re.search(r"/(\d+)$", content_range)
    if match:
        return int(match.group(1))
    content_length = response.headers.get("Content-Length")
    if content_length and content_length.isdigit():
        length = int(content_length)
        return offset + length if response.status_code == 206 else length
    return None


def create_download_session(
    session: requests.Session,
    api_url: str,
    dataset_id: str,
) -> tuple[str, str]:
    endpoint = urljoin(f"{api_url.rstrip('/')}/", f"datasets/{dataset_id}/download")
    response = session.post(endpoint, timeout=(15, 60))
    if response.status_code == 403:
        raise PermissionError(
            "MDC download denied; log in and accept the dataset terms first"
        )
    response.raise_for_status()
    payload = response.json()
    download_url = payload.get("downloadUrl")
    filename = payload.get("filename")
    if not download_url or not filename:
        raise RuntimeError("MDC returned an incomplete download session")
    safe_filename = Path(str(filename)).name
    if safe_filename != filename:
        raise RuntimeError(f"Unsafe archive filename returned by MDC: {filename!r}")
    return str(download_url), safe_filename


def byte_ranges(total: int, segment_bytes: int) -> list[tuple[int, int]]:
    if total < 1 or segment_bytes < 1:
        raise ValueError("total and segment size must be positive")
    return [
        (start, min(total - 1, start + segment_bytes - 1))
        for start in range(0, total, segment_bytes)
    ]


def range_total(download_url: str) -> int | None:
    with requests.get(
        download_url,
        headers={"Range": "bytes=0-0"},
        stream=True,
        timeout=(30, 300),
    ) as response:
        response.raise_for_status()
        if response.status_code != 206:
            return None
        return content_total(response, 0)


def download_range(
    download_url: str,
    target: Path,
    start: int,
    end: int,
    max_attempts: int,
    chunk_bytes: int,
) -> Path:
    expected = end - start + 1
    if target.is_file() and target.stat().st_size == expected:
        return target
    partial = target.with_name(f"{target.name}.part")
    for attempt in range(1, max_attempts + 1):
        try:
            offset = partial.stat().st_size if partial.is_file() else 0
            if offset > expected:
                partial.unlink()
                offset = 0
            if offset == expected:
                partial.replace(target)
                return target
            request_start = start + offset
            with requests.get(
                download_url,
                headers={"Range": f"bytes={request_start}-{end}"},
                stream=True,
                timeout=(30, 300),
            ) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise IOError(
                        f"range server returned HTTP {response.status_code}"
                    )
                written = offset
                with partial.open("ab" if offset else "wb") as handle:
                    for chunk in response.iter_content(chunk_bytes):
                        if chunk:
                            handle.write(chunk)
                            written += len(chunk)
            if written != expected:
                raise IOError(
                    f"incomplete range {start}-{end}: {written} != {expected}"
                )
            partial.replace(target)
            return target
        except (OSError, requests.RequestException):
            if attempt == max_attempts:
                raise
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def download_parallel(
    args: argparse.Namespace,
    download_url: str,
    target: Path,
) -> Path | None:
    total = range_total(download_url)
    if total is None:
        return None
    segment_bytes = args.segment_mib * 1024 * 1024
    ranges = byte_ranges(total, segment_bytes)
    segment_dir = target.with_name(f".{target.name}.segments")
    segment_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Parallel download: total={total}, workers={args.workers}, "
        f"segments={len(ranges)}, segment_mib={args.segment_mib}",
        flush=True,
    )

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for index, (start, end) in enumerate(ranges):
            segment = segment_dir / f"{index:06d}-{start}-{end}.bin"
            future = pool.submit(
                download_range,
                download_url,
                segment,
                start,
                end,
                args.max_attempts,
                args.chunk_mib * 1024 * 1024,
            )
            futures[future] = (index, start, end)
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            if completed == len(ranges) or completed % max(1, args.workers) == 0:
                print(
                    f"Completed {completed}/{len(ranges)} ranges "
                    f"({min(total, completed * segment_bytes) / total:.1%})",
                    flush=True,
                )

    assembled = target.with_name(f".{target.name}.assembling")
    with assembled.open("wb") as output:
        for index, (start, end) in enumerate(ranges):
            segment = segment_dir / f"{index:06d}-{start}-{end}.bin"
            with segment.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    output.write(chunk)
    if assembled.stat().st_size != total:
        raise IOError(f"assembled archive size mismatch: {assembled.stat().st_size} != {total}")
    assembled.replace(target)
    print(f"Saved dataset to {target} ({total} bytes)", flush=True)
    return target


def download(args: argparse.Namespace) -> Path:
    api_key = os.environ.get("MDC_API_KEY")
    if not api_key:
        raise RuntimeError("MDC_API_KEY is not set")
    if (
        args.max_attempts < 1
        or args.chunk_mib < 1
        or args.progress_mib < 1
        or args.workers < 1
        or args.segment_mib < 1
    ):
        raise ValueError("attempt and MiB arguments must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": USER_AGENT,
        }
    )

    last_error: Exception | None = None
    target: Path | None = None
    for attempt in range(1, args.max_attempts + 1):
        try:
            download_url, filename = create_download_session(
                session, args.api_url, args.dataset_id
            )
            target = args.output_dir / filename
            partial = target.with_name(f"{target.name}.part")
            if target.is_file():
                print(f"Archive already exists: {target}", flush=True)
                return target
            if args.workers > 1:
                result = download_parallel(args, download_url, target)
                if result is not None:
                    return result
                print(
                    "Server does not support HTTP ranges; falling back to serial",
                    flush=True,
                )

            offset = partial.stat().st_size if partial.is_file() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.get(
                download_url,
                headers=headers,
                stream=True,
                timeout=(30, 300),
            ) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                if offset and not append:
                    print("Server ignored Range; restarting archive", flush=True)
                    offset = 0
                total = content_total(response, offset)
                mode = "ab" if append else "wb"
                downloaded = offset
                report_bytes = args.progress_mib * 1024 * 1024
                next_report = ((downloaded // report_bytes) + 1) * report_bytes
                print(
                    f"Attempt {attempt}: {filename}, resume={offset} bytes, "
                    f"total={total if total is not None else 'unknown'}",
                    flush=True,
                )
                with partial.open(mode) as handle:
                    for chunk in response.iter_content(args.chunk_mib * 1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if downloaded >= next_report:
                            pct = (
                                f" ({downloaded / total:.1%})" if total else ""
                            )
                            print(f"Downloaded {downloaded} bytes{pct}", flush=True)
                            next_report += report_bytes

            if total is not None and downloaded != total:
                raise IOError(f"incomplete archive: {downloaded} != {total}")
            partial.replace(target)
            print(f"Saved dataset to {target} ({downloaded} bytes)", flush=True)
            return target
        except PermissionError:
            # Dataset terms are an external authorization gate. Retrying cannot
            # change that state and only wastes the audit allocation.
            raise
        except (OSError, ValueError, requests.RequestException) as exc:
            last_error = exc
            if attempt == args.max_attempts:
                break
            delay = min(2**attempt, 60)
            print(
                f"Attempt {attempt} failed: {type(exc).__name__}: {exc}; "
                f"retrying in {delay}s",
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError(f"MDC download failed after retries: {last_error!r}")


def main() -> None:
    args = parse_args()
    print(download(args))


if __name__ == "__main__":
    main()
