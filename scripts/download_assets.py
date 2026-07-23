#!/usr/bin/env python3
"""Download pinned official Hub assets sequentially with integrity checks."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cantonese_asr.io import sha256_file


MODEL_ID = "openai/whisper-small"
DATASET_ID = "leeduckgo/cantonese-life-scenarios-corpus"
DEFAULT_MIRROR = "https://hf-mirror.com"
USER_AGENT = "cantonese-asr-goal1/1.0 huggingface-hub-compatible"
CRITICAL_DATA_FILES = {
    "README.md",
    "data.zip",
    "evaluator.py",
    "evaluator_pre.py",
    "index.csv",
    "submission_example.zip",
    "template_pre.jsonl",
    "test_audio.zip",
    "赛事提交说明.md",
}
CRITICAL_MODEL_FILES = {
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
}
MODEL_ALLOW_PATTERNS = sorted(
    CRITICAL_MODEL_FILES
    | {
        ".gitattributes",
        "README.md",
        "added_tokens.json",
        "normalizer.json",
        "special_tokens_map.json",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--model-only", action="store_true")
    parser.add_argument("--dataset-only", action="store_true")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument(
        "--endpoint",
        action="append",
        help="May be repeated. Defaults to HF_ENDPOINT, hf-mirror, then Hugging Face.",
    )
    return parser.parse_args()


def endpoints(cli_values: list[str] | None) -> list[str]:
    values = cli_values or [
        os.environ.get("HF_ENDPOINT", ""),
        DEFAULT_MIRROR,
        "https://huggingface.co",
    ]
    return list(dict.fromkeys(value.rstrip("/") for value in values if value))


def repo_url(endpoint: str, repo_id: str, repo_type: str) -> str:
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"{endpoint}/{prefix}{repo_id}"


def resolved_revision(
    endpoint: str, repo_id: str, repo_type: str, revision: str
) -> str | None:
    reference = "HEAD" if revision == "main" else revision
    try:
        result = subprocess.run(
            ["git", "ls-remote", repo_url(endpoint, repo_id, repo_type), reference],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        line = result.stdout.strip().splitlines()[0]
        commit = line.split()[0]
        return commit if len(commit) == 40 else None
    except (IndexError, OSError, subprocess.SubprocessError):
        return None


def api_kind(repo_type: str) -> str:
    return "datasets" if repo_type == "dataset" else "models"


def tree_url(endpoint: str, repo_id: str, repo_type: str, revision: str) -> str:
    return (
        f"{endpoint}/api/{api_kind(repo_type)}/{repo_id}/tree/"
        f"{quote(revision, safe='')}?recursive=true&expand=false"
    )


def resolve_url(
    endpoint: str, repo_id: str, repo_type: str, revision: str, path: str
) -> str:
    prefix = "datasets/" if repo_type == "dataset" else ""
    encoded_path = quote(path, safe="/")
    encoded_revision = quote(revision, safe="")
    return (
        f"{endpoint}/{prefix}{repo_id}/resolve/{encoded_revision}/"
        f"{encoded_path}?download=true"
    )


def fetch_tree(
    endpoint: str,
    repo_id: str,
    repo_type: str,
    revision: str,
    max_attempts: int,
) -> list[dict[str, Any]]:
    url = tree_url(endpoint, repo_id, repo_type, revision)
    error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=(30, 120),
            )
            response.raise_for_status()
            rows = response.json()
            if not isinstance(rows, list):
                raise ValueError("Hub tree response is not a list")
            return [row for row in rows if row.get("type") == "file"]
        except (requests.RequestException, ValueError) as exc:
            error = exc
            if attempt < max_attempts:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Unable to list {repo_id} at {endpoint}: {error!r}")


def selected_rows(
    rows: list[dict[str, Any]], allow_patterns: list[str] | None
) -> list[dict[str, Any]]:
    if allow_patterns is None:
        return rows
    return [
        row
        for row in rows
        if any(fnmatch.fnmatch(str(row["path"]), pattern) for pattern in allow_patterns)
    ]


def download_file(
    endpoint: str,
    repo_id: str,
    repo_type: str,
    revision: str,
    row: dict[str, Any],
    target_root: Path,
    max_attempts: int,
) -> str:
    relative = Path(str(row["path"]))
    expected_size = int(row["size"])
    target = target_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == expected_size:
        return "skipped"

    partial = target.with_name(f"{target.name}.direct-part")
    if partial.is_file() and partial.stat().st_size > expected_size:
        partial.unlink()
    url = resolve_url(endpoint, repo_id, repo_type, revision, relative.as_posix())
    error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            offset = partial.stat().st_size if partial.is_file() else 0
            headers = {"User-Agent": USER_AGENT}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with requests.get(
                url,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=(30, 300),
            ) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                mode = "ab" if append else "wb"
                with partial.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            actual_size = partial.stat().st_size
            if actual_size != expected_size:
                raise IOError(
                    f"size mismatch for {relative}: {actual_size} != {expected_size}"
                )
            partial.replace(target)
            return "downloaded"
        except (OSError, requests.RequestException) as exc:
            error = exc
            if partial.is_file() and partial.stat().st_size > expected_size:
                partial.unlink()
            if attempt < max_attempts:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Unable to download {relative}: {error!r}")


def direct_snapshot(
    endpoint: str,
    repo_id: str,
    repo_type: str,
    target: Path,
    revision: str,
    allow_patterns: list[str] | None,
    max_attempts: int,
) -> dict[str, Any]:
    commit = resolved_revision(endpoint, repo_id, repo_type, revision)
    pinned_revision = commit or revision
    rows = selected_rows(
        fetch_tree(endpoint, repo_id, repo_type, pinned_revision, max_attempts),
        allow_patterns,
    )
    if not rows:
        raise RuntimeError(f"No selected files returned for {repo_id}")
    counts = {"downloaded": 0, "skipped": 0}
    for index, row in enumerate(rows, start=1):
        result = download_file(
            endpoint,
            repo_id,
            repo_type,
            pinned_revision,
            row,
            target,
            max_attempts,
        )
        counts[result] += 1
        print(
            json.dumps(
                {
                    "repo": repo_id,
                    "file": row["path"],
                    "index": index,
                    "total": len(rows),
                    "status": result,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "requested_revision": revision,
        "resolved_revision": commit,
        "download_revision": pinned_revision,
        "endpoint": endpoint,
        "target": str(target.resolve()),
        "selected_file_count": len(rows),
        **counts,
        "tree": rows,
    }


def download_with_fallback(
    repo_id: str,
    repo_type: str,
    target: Path,
    revision: str,
    endpoint_values: list[str],
    allow_patterns: list[str] | None,
    max_attempts: int,
) -> dict[str, Any]:
    failures = []
    for endpoint in endpoint_values:
        try:
            return direct_snapshot(
                endpoint,
                repo_id,
                repo_type,
                target,
                revision,
                allow_patterns,
                max_attempts,
            )
        except Exception as exc:
            failures.append({"endpoint": endpoint, "error": repr(exc)})
    raise RuntimeError(
        f"All download endpoints failed for {repo_id}: "
        + json.dumps(failures, ensure_ascii=False)
    )


def inventory(root: Path, critical_names: set[str]) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if ".cache" in path.parts or path.name.endswith(".direct-part"):
            continue
        relative = path.relative_to(root).as_posix()
        entry: dict[str, Any] = {"size_bytes": path.stat().st_size}
        if path.name in critical_names:
            entry["sha256"] = sha256_file(path)
        result[relative] = entry
    return result


def missing_critical_files(
    file_inventory: dict[str, dict[str, Any]], critical_names: set[str]
) -> list[str]:
    present_names = {Path(relative).name for relative in file_inventory}
    return sorted(critical_names - present_names)


def validate_lfs(
    snapshot: dict[str, Any], file_inventory: dict[str, dict[str, Any]]
) -> list[str]:
    mismatches = []
    for row in snapshot["tree"]:
        lfs = row.get("lfs") or {}
        expected = lfs.get("oid")
        actual = file_inventory.get(str(row["path"]), {}).get("sha256")
        if expected and actual and expected != actual:
            mismatches.append(str(row["path"]))
    return mismatches


def main() -> None:
    args = parse_args()
    if args.model_only and args.dataset_only:
        raise SystemExit("Choose at most one of --model-only and --dataset-only")
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be positive")
    endpoint_values = endpoints(args.endpoint)
    report: dict[str, Any] = {"endpoints": endpoint_values}
    if not args.dataset_only:
        model_dir = args.output_root / "models" / "whisper-small"
        model_dir.mkdir(parents=True, exist_ok=True)
        model = download_with_fallback(
            MODEL_ID,
            "model",
            model_dir,
            args.model_revision,
            endpoint_values,
            MODEL_ALLOW_PATTERNS,
            args.max_attempts,
        )
        model_files = inventory(model_dir, CRITICAL_MODEL_FILES)
        model["files"] = model_files
        model["missing_critical_files"] = missing_critical_files(
            model_files, CRITICAL_MODEL_FILES
        )
        model["lfs_sha256_mismatches"] = validate_lfs(model, model_files)
        model.pop("tree")
        report["model"] = model
    if not args.model_only:
        dataset_dir = args.output_root / "datasets" / "official"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset = download_with_fallback(
            DATASET_ID,
            "dataset",
            dataset_dir,
            args.dataset_revision,
            endpoint_values,
            None,
            args.max_attempts,
        )
        dataset_files = inventory(dataset_dir, CRITICAL_DATA_FILES)
        dataset["files"] = dataset_files
        dataset["missing_critical_files"] = missing_critical_files(
            dataset_files, CRITICAL_DATA_FILES
        )
        dataset["lfs_sha256_mismatches"] = validate_lfs(dataset, dataset_files)
        dataset.pop("tree")
        report["dataset"] = dataset

    report_dir = args.output_root / "reports" / "deployment"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "download_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    invalid = {
        key: {
            "missing_critical_files": value.get("missing_critical_files", []),
            "lfs_sha256_mismatches": value.get("lfs_sha256_mismatches", []),
        }
        for key, value in report.items()
        if isinstance(value, dict)
        and (
            value.get("missing_critical_files")
            or value.get("lfs_sha256_mismatches")
        )
    }
    if invalid:
        raise SystemExit(
            "Downloaded snapshot failed integrity checks: "
            + json.dumps(invalid, ensure_ascii=False)
        )


if __name__ == "__main__":
    main()
