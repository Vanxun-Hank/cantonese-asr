#!/usr/bin/env python3
"""Download a Mozilla Data Collective archive with resume and retries.

The API key is read from ``MDC_API_KEY`` and is never written to logs.
"""

from __future__ import annotations

import argparse
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


def download(args: argparse.Namespace) -> Path:
    api_key = os.environ.get("MDC_API_KEY")
    if not api_key:
        raise RuntimeError("MDC_API_KEY is not set")
    if args.max_attempts < 1 or args.chunk_mib < 1 or args.progress_mib < 1:
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
