#!/usr/bin/env python3
"""Audit Common Voice 26.0 Cantonese (`yue`) before training admission."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import os
import sys
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import librosa
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cantonese_asr.common_voice_audit import (  # noqa: E402
    canonical_text,
    classify_candidate,
    compare_rows,
    decide_admission,
    enrich_row,
    profile_split,
    profile_text,
    select_review_sample,
    sha256_path,
)
from cantonese_asr.io import read_jsonl, write_jsonl  # noqa: E402


EXPECTED_DATASET_ID = "cmqinjd7x00vynq07pwzo3lmp"
SPLITS = ("train", "dev", "test", "validated", "other", "invalidated")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extract-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--reference-manifest", action="append", default=[])
    parser.add_argument("--protected-manifest", type=Path, action="append", default=[])
    parser.add_argument("--dataset-id", default=EXPECTED_DATASET_ID)
    parser.add_argument("--whisper-model", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--max-label-tokens", type=int, default=225)
    return parser.parse_args()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.tmp")
    partial.write_text(content, encoding="utf-8")
    os.replace(partial, path)


def write_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def safe_member(member: tarfile.TarInfo) -> PurePosixPath:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"Unsafe archive member: {member.name}")
    return path


def locale_root_and_members(
    archive: tarfile.TarFile,
) -> tuple[PurePosixPath, dict[str, tarfile.TarInfo]]:
    members: dict[str, tarfile.TarInfo] = {}
    roots: set[PurePosixPath] = set()
    for member in archive.getmembers():
        path = safe_member(member)
        if not member.isfile():
            continue
        if "yue" not in path.parts:
            continue
        index = path.parts.index("yue")
        root = PurePosixPath(*path.parts[: index + 1])
        roots.add(root)
        relative = PurePosixPath(*path.parts[index + 1 :]).as_posix()
        members[relative] = member
    if len(roots) != 1:
        raise RuntimeError(f"Expected one /yue archive root, found {sorted(map(str, roots))}")
    return roots.pop(), members


def member_bytes(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    handle = archive.extractfile(member)
    if handle is None:
        raise RuntimeError(f"Cannot read archive member: {member.name}")
    return handle.read()


def tsv_rows(data: bytes) -> list[dict[str, str]]:
    text = data.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def duration_map(rows: Iterable[dict[str, str]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in rows:
        clip = str(row.get("clip") or "")
        value = str(row.get("duration[ms]") or "")
        if clip and value:
            result[clip] = float(value) / 1000.0
    return result


def pcm_sha256(path: Path) -> tuple[str, float, int, int]:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if samples.size == 0 or sample_rate <= 0:
        raise RuntimeError("empty decoded audio")
    channels = int(samples.shape[1])
    mono = np.mean(samples, axis=1)
    mono_16k = librosa.resample(mono, orig_sr=int(sample_rate), target_sr=16_000)
    pcm = np.clip(mono_16k * 32767.0, -32768, 32767).astype("<i2")
    return (
        hashlib.sha256(pcm.tobytes()).hexdigest(),
        len(mono) / float(sample_rate),
        int(sample_rate),
        channels,
    )


def token_counter(model_path: Path | None):
    if model_path is None:
        return lambda text: 0
    from transformers import WhisperTokenizerFast

    tokenizer = WhisperTokenizerFast.from_pretrained(
        model_path, language="zh", task="transcribe", local_files_only=True
    )
    return lambda text: len(tokenizer(str(text), add_special_tokens=True).input_ids)


def parse_named_manifest(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise ValueError(f"Expected NAME=PATH, got {value!r}")
    return name.strip(), Path(raw_path)


def resolve_audio(path_value: object, project_root: Path) -> Path:
    path = Path(str(path_value or ""))
    return path if path.is_absolute() else project_root / path


def load_reference(name: str, path: Path, project_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in read_jsonl(path):
        text = str(raw.get("text", raw.get("ref_text", "")))
        row = {
            **raw,
            "id": str(raw.get("id") or raw.get("audio_path") or ""),
            "source": str(raw.get("source") or name),
            "text": profile_text(text)["clean"],
            "canonical": canonical_text(text),
            "audio_sha256": str(
                raw.get("audio_sha256") or raw.get("audio_sha256_observed") or ""
            ),
            "pcm_sha256": str(raw.get("pcm_sha256") or ""),
        }
        audio_path = resolve_audio(raw.get("audio_path"), project_root)
        if not row["audio_sha256"] and audio_path.is_file():
            row["audio_sha256"] = sha256_path(audio_path)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.tmp")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)


def build_style_profile(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split, rows in rows_by_split.items():
        result[split] = {
            "rows": len(rows),
            "rows_with_cantonese_markers": sum(
                bool(row["text_profile"]["cantonese_marker_count"]) for row in rows
            ),
            "rows_with_written_risk_markers": sum(
                bool(row["text_profile"]["written_risk_count"]) for row in rows
            ),
            "rows_with_latin": sum(bool(row["text_profile"]["latin_count"]) for row in rows),
            "rows_with_digits": sum(bool(row["text_profile"]["digit_count"]) for row in rows),
            "cantonese_markers": dict(
                Counter(
                    marker
                    for row in rows
                    for marker, count in row["text_profile"]["cantonese_markers"].items()
                    for _ in range(int(count))
                )
            ),
        }
    return result


def write_html(path: Path, report: dict[str, Any]) -> None:
    split_rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(profile.get(field, '')))}</td>"
            for field in ("split", "rows", "speakers", "hours", "top10_speaker_share")
        )
        + "</tr>"
        for profile in report["split_profiles"]
    )
    overlap_rows = "".join(
        f"<tr><td>{html.escape(name)}</td>"
        f"<td>{value['canonical_sentence_overlap']['count']}</td>"
        f"<td>{value['audio_sha256_overlap']['count']}</td>"
        f"<td>{value['speaker_row_overlap']['count']}</td></tr>"
        for name, value in report["overlaps"].items()
    )
    gate_rows = "".join(
        f"<tr><td>{html.escape(gate['name'])}</td><td>{html.escape(str(gate['observed']))}</td>"
        f"<td>{html.escape(gate['operator'])} {html.escape(str(gate['threshold']))}</td>"
        f"<td>{'PASS' if gate['passed'] else 'FAIL/PENDING'}</td></tr>"
        for gate in report["admission"]["gates"]
    )
    document = f"""<!doctype html>
<html lang="zh"><meta charset="utf-8"><title>Common Voice yue admission audit</title>
<style>body{{font-family:system-ui;max-width:1100px;margin:2rem auto;line-height:1.5}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #ccc;padding:.45rem}}
code,pre{{background:#f5f5f5;padding:.2rem}}</style>
<h1>Common Voice 26.0 Cantonese (`yue`) 数据准入审计</h1>
<p><b>Decision:</b> {html.escape(report['admission']['decision'])}</p>
<p>本报告只审计数据；训练作业 0，平台提交 0。</p>
<h2>Provenance</h2><pre>{html.escape(json.dumps(report['provenance'], ensure_ascii=False, indent=2))}</pre>
<h2>Split profile</h2><table><tr><th>split</th><th>rows</th><th>speakers</th><th>hours</th><th>top10 speaker share</th></tr>{split_rows}</table>
<h2>Overlap</h2><table><tr><th>reference</th><th>canonical sentence</th><th>audio bytes</th><th>speaker rows</th></tr>{overlap_rows}</table>
<h2>Admission gates</h2><table><tr><th>gate</th><th>observed</th><th>threshold</th><th>result</th></tr>{gate_rows}</table>
<h2>Next step</h2><p>完成固定 acoustic review 后才可作最终准入判断；在此之前不训练。</p>
</html>
"""
    atomic_text(path, document)


def main() -> None:
    args = parse_args()
    if args.dataset_id != EXPECTED_DATASET_ID:
        raise RuntimeError(f"Wrong dataset ID: {args.dataset_id}")
    if not args.archive.is_file() or args.archive.name.endswith(".part"):
        raise RuntimeError(f"Completed archive not found: {args.archive}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_root = args.extract_root / "metadata"
    clips_root = args.extract_root / "extracted_train" / "clips"
    metadata_root.mkdir(parents=True, exist_ok=True)
    clips_root.mkdir(parents=True, exist_ok=True)

    archive_hash = sha256_path(args.archive)
    with tarfile.open(args.archive, "r:*") as archive:
        root, members = locale_root_and_members(archive)
        metadata_names = [
            name
            for name in members
            if name.endswith(".tsv") or Path(name).name.lower().startswith("readme")
        ]
        for name in metadata_names:
            target = metadata_root / Path(name).name
            if not target.is_file():
                target.write_bytes(member_bytes(archive, members[name]))
        durations = duration_map(tsv_rows(member_bytes(archive, members["clip_durations.tsv"])))
        raw_splits: dict[str, list[dict[str, str]]] = {}
        for split in SPLITS:
            member = members.get(f"{split}.tsv")
            if member:
                raw_splits[split] = tsv_rows(member_bytes(archive, member))
        if "train" not in raw_splits:
            raise RuntimeError("Archive has no train.tsv")
        train_paths = {str(row.get("path") or "") for row in raw_splits["train"]}
        for index, filename in enumerate(sorted(train_paths), start=1):
            member = members.get(f"clips/{filename}")
            if member is None:
                continue
            target = clips_root / Path(filename).name
            if not target.is_file() or target.stat().st_size != member.size:
                target.write_bytes(member_bytes(archive, member))
            if index % 1000 == 0:
                print(f"Extracted {index}/{len(train_paths)} train clips", flush=True)

    count_tokens = token_counter(args.whisper_model)
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, raw_rows in raw_splits.items():
        rows: list[dict[str, Any]] = []
        for raw in raw_rows:
            filename = str(raw.get("path") or "")
            base = {
                **raw,
                "path": filename,
                "duration_s": durations.get(filename),
                "client_id": str(raw.get("client_id") or ""),
            }
            rows.append(enrich_row(base, "common_voice_26_yue", split))
        rows_by_split[split] = rows

    train_rows = rows_by_split["train"]
    seen_audio: set[str] = set()
    seen_pcm: set[str] = set()
    for index, row in enumerate(train_rows, start=1):
        path = clips_root / Path(str(row.get("path") or "")).name
        row["audio_path"] = str(path)
        row["label_tokens"] = count_tokens(row["text"])
        row["audio_readable"] = False
        if path.is_file():
            row["audio_sha256"] = sha256_path(path)
            try:
                pcm_hash, decoded_duration, sample_rate, channels = pcm_sha256(path)
                row.update(
                    {
                        "pcm_sha256": pcm_hash,
                        "duration_s": decoded_duration,
                        "sample_rate": sample_rate,
                        "channels": channels,
                        "audio_readable": True,
                    }
                )
            except Exception as exc:
                row["audio_error"] = str(exc)
        row["internal_duplicate_audio"] = bool(
            row.get("audio_sha256") in seen_audio or row.get("pcm_sha256") in seen_pcm
        )
        if row.get("audio_sha256"):
            seen_audio.add(str(row["audio_sha256"]))
        if row.get("pcm_sha256"):
            seen_pcm.add(str(row["pcm_sha256"]))
        if index % 1000 == 0:
            print(f"Hashed {index}/{len(train_rows)} train clips", flush=True)

    references = {
        name: load_reference(name, path, args.project_root)
        for name, path in map(parse_named_manifest, args.reference_manifest)
    }
    protected_rows = [
        row
        for path in args.protected_manifest
        for row in load_reference(f"protected:{path.name}", path, args.project_root)
    ]
    protected_text = {str(row.get("canonical") or "") for row in protected_rows}
    protected_audio = {
        value
        for row in protected_rows
        for value in (str(row.get("audio_sha256") or ""), str(row.get("pcm_sha256") or ""))
        if value
    }
    overlaps = {
        name: compare_rows(train_rows, rows, name) for name, rows in sorted(references.items())
    }
    all_reference_audio = {
        str(row.get("audio_sha256") or "")
        for rows in references.values()
        for row in rows
        if row.get("audio_sha256")
    }
    all_reference_pcm = {
        str(row.get("pcm_sha256") or "")
        for rows in references.values()
        for row in rows
        if row.get("pcm_sha256")
    }
    all_reference_canonical = {
        str(row.get("canonical") or "")
        for rows in references.values()
        for row in rows
        if row.get("canonical")
    }
    existing_speakers = {
        str(row.get("client_id") or "")
        for rows in references.values()
        for row in rows
        if row.get("client_id")
    }
    accepted: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    accepted_audio: set[str] = set()
    accepted_pcm: set[str] = set()
    for row in train_rows:
        row["same_sentence_new_audio"] = (
            row.get("canonical") in all_reference_canonical
            and row.get("audio_sha256") not in all_reference_audio
            and row.get("pcm_sha256") not in all_reference_pcm
        )
        decision = classify_candidate(
            row, protected_text, protected_audio, args.max_duration, args.max_label_tokens
        )
        duplicate = (
            bool(row.get("audio_sha256") and row["audio_sha256"] in accepted_audio)
            or bool(row.get("pcm_sha256") and row["pcm_sha256"] in accepted_pcm)
        )
        if duplicate:
            decision = {"status": "hard_quarantine", "reasons": ["duplicate_audio"]}
        enriched = {**row, "qc_status": decision["status"], "qc_reasons": decision["reasons"]}
        if decision["status"] == "accepted":
            accepted.append(enriched)
            if row.get("audio_sha256"):
                accepted_audio.add(str(row["audio_sha256"]))
            if row.get("pcm_sha256"):
                accepted_pcm.add(str(row["pcm_sha256"]))
        else:
            quarantine.append(enriched)

    train_profile = profile_split(train_rows, "train")
    new_speakers = {
        str(row.get("client_id") or "")
        for row in accepted
        if row.get("client_id") and str(row.get("client_id")) not in existing_speakers
    }
    new_rows = [
        row
        for row in accepted
        if row.get("audio_sha256") not in all_reference_audio
        and row.get("pcm_sha256") not in all_reference_pcm
    ]
    protected_text_overlap = sum(row["canonical"] in protected_text for row in train_rows)
    protected_audio_overlap = sum(
        row.get("audio_sha256") in protected_audio or row.get("pcm_sha256") in protected_audio
        for row in train_rows
    )
    gate_metrics = {
        "retention_rate": len(accepted) / len(train_rows) if train_rows else 0.0,
        "new_utterances": len(new_rows),
        "new_speakers": len(new_speakers),
        "new_hours": sum(float(row.get("duration_s") or 0) for row in new_rows) / 3600,
        "sample_yue_rate": None,
        "protected_text_overlap": protected_text_overlap,
        "protected_audio_overlap": protected_audio_overlap,
        "top10_speaker_share": train_profile["top10_speaker_share"],
        "incremental_coverage": bool(new_rows and new_speakers),
    }
    split_profiles = [profile_split(rows, split) for split, rows in rows_by_split.items()]
    provenance = {
        "dataset_id": args.dataset_id,
        "locale": "yue",
        "archive": str(args.archive.resolve()),
        "archive_bytes": args.archive.stat().st_size,
        "archive_sha256": archive_hash,
        "archive_root": str(root),
        "seed": args.seed,
        "training_runs_started": 0,
        "platform_submissions": 0,
    }
    report = {
        "provenance": provenance,
        "split_profiles": split_profiles,
        "overlaps": overlaps,
        "quality": {
            "input_train": len(train_rows),
            "accepted": len(accepted),
            "quarantine": len(quarantine),
            "quarantine_reasons": dict(
                Counter(reason for row in quarantine for reason in row["qc_reasons"])
            ),
        },
        "incremental": gate_metrics,
        "admission": decide_admission(gate_metrics),
    }

    review = select_review_sample(accepted, args.seed)
    write_json(args.output_dir / "provenance.json", provenance)
    write_json(args.output_dir / "overlap_summary.json", overlaps)
    write_json(args.output_dir / "style_profile.json", build_style_profile(rows_by_split))
    write_json(args.output_dir / "data_report.json", report)
    write_jsonl(args.output_dir / "accepted_train.jsonl", accepted)
    write_jsonl(args.output_dir / "quarantine.jsonl", quarantine)
    write_jsonl(
        args.output_dir / "overlap_examples.jsonl",
        [
            row
            for row in train_rows
            if row.get("same_sentence_new_audio")
            or row.get("canonical") in protected_text
            or row.get("audio_sha256") in all_reference_audio
        ][:500],
    )
    write_csv(
        args.output_dir / "split_profile.csv",
        split_profiles,
        ["split", "rows", "unique_audio_paths", "unique_transcripts",
         "unique_canonical_sentences", "speakers", "hours",
         "top1_speaker_share", "top10_speaker_share"],
    )
    speaker_counts = Counter(str(row.get("client_id") or "") for row in train_rows)
    speaker_hours = Counter()
    for row in train_rows:
        speaker_hours[str(row.get("client_id") or "")] += float(row.get("duration_s") or 0) / 3600
    write_csv(
        args.output_dir / "speaker_coverage.csv",
        [
            {
                "client_id": speaker,
                "utterances": count,
                "hours": speaker_hours[speaker],
                "seen_in_existing": speaker in existing_speakers,
            }
            for speaker, count in speaker_counts.most_common()
            if speaker
        ],
        ["client_id", "utterances", "hours", "seen_in_existing"],
    )
    write_csv(
        args.output_dir / "duration_distribution.csv",
        [
            {"id": row["id"], "split": row["publisher_split"], "duration_s": row.get("duration_s")}
            for rows in rows_by_split.values()
            for row in rows
        ],
        ["id", "split", "duration_s"],
    )
    review_fields = [
        "id", "review_bucket", "client_id", "publisher_split", "duration_s",
        "text", "audio_path", "audio_sha256", "pcm_sha256",
    ]
    write_csv(args.output_dir / "style_review_sample.csv", review, review_fields)
    write_html(args.output_dir / "report.html", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
