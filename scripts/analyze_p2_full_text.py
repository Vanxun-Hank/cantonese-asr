#!/usr/bin/env python3
"""Versioned, offline LM/fusion analysis with validation-only decisions."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cantonese_asr.io import read_jsonl, sha256_file, write_jsonl
from cantonese_asr.metrics import (build_error_analysis, compute_official_metrics,
    compute_diagnostic_metrics, levenshtein_ops, normalize_prediction, normalize_reference)
from cantonese_asr.p2_analysis import CharacterNgramLM, rerank_candidates, rover_anchor_vote
from cantonese_asr.p2_full import atomic_json


def raw_mbr(texts, priorities):
    risks = [sum(levenshtein_ops(list(normalize_prediction(text)), list(normalize_prediction(other)))[0]
                 for other in texts) / len(texts) for text in texts]
    return min(range(len(texts)), key=lambda i: (risks[i], priorities[i], i))


def metrics(refs, texts):
    return {**compute_official_metrics(refs, texts), "diagnostics": compute_diagnostic_metrics(refs, texts)}


def metric_key(value):
    return (-value["sentence_accuracy_tol2"], value["cer"], value["diagnostics"]["severe_error_count"])


def aligned(path, reference_rows, identities, surface):
    rows = read_jsonl(path)
    if len(rows) != len(reference_rows):
        raise ValueError(f"wrong row count: {path}")
    for index, (row, reference) in enumerate(zip(rows, reference_rows)):
        listed = str(row["audio_path"])
        expected = identities[(surface, index)]
        # Exact frozen manifest path is valid; alternate paths require the
        # preparation's content identity, never a basename-only fallback.
        if listed != str(reference["audio_path"]) and identities.get(listed) != expected:
            raise ValueError(f"unverified audio/order identity: {path}:{index}")
    return rows


def emit(path, reference_rows, field, texts, sidecars, kind):
    from scripts.evaluate_raw_winner_decode import long_text_cycle, percentile
    path.mkdir(parents=True, exist_ok=False)
    refs = [str(row[field]) for row in reference_rows]
    value = metrics(refs, texts)
    write_jsonl(path / "predictions.jsonl", [{"audio_path": row["audio_path"], "pred_text": text}
                for row, text in zip(reference_rows, texts)])
    tokens = []
    for row, text, sidecar in zip(reference_rows, texts, sidecars):
        tokens.append({**(sidecar or {}), "audio_path": row["audio_path"], "pred_text": text,
                       "origin": kind, "generated_by_asr": sidecar is not None,
                       "stop_reason": "unverified" if sidecar is not None else "not_applicable_postprocessing"})
    write_jsonl(path / "generation_tokens.jsonl", tokens)
    analysis = build_error_analysis([{"audio_path": row["audio_path"], "reference": ref,
                     "prediction": text, "scene": "unknown"} for row, ref, text in zip(reference_rows, refs, texts)])
    atomic_json(path / "metrics.json", value)
    atomic_json(path / "error_examples.json", analysis["error_examples"])
    with (path / "top_confusions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["type", "reference", "prediction", "count"])
        for row in analysis["substitutions"]:
            writer.writerow(["substitution", row["reference"], row["prediction"], row["count"]])
    drift = {"唔→不": 0, "冇→无": 0}
    denominators = {"唔": 0, "冇": 0}
    for ref, text in zip(refs, texts):
        ref, text = normalize_reference(ref), normalize_prediction(text)
        for char in denominators:
            denominators[char] += ref.count(char)
        for left, right in levenshtein_ops(list(ref), list(text))[1]:
            key = left + "→" + right
            if key in drift:
                drift[key] += 1
    lengths = [len(normalize_prediction(text)) for text in texts]
    atomic_json(path / "generation.json", {"origin": kind,
        "text_repeat_count": sum(long_text_cycle(text)["triggered"] for text in texts),
        "replacement_character_count": sum(text.count("�") for text in texts),
        "no_eos_count": None, "max_length_count": None,
        "token_diagnostic_note": "Do not infer generation EOS/length from retokenized fusion text; inspect source candidate sidecars.",
        "character_length": {f"p{q}": percentile(lengths, q/100) for q in (50,90,95,99)},
        "character_length_max": max(lengths, default=0), "writing_drift": drift,
        "reference_marker_counts": denominators})
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("lm", "fusion"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--neural-lm", type=Path, nargs="*", default=[])
    parser.add_argument("--model-roots", type=Path, nargs=3)
    a = parser.parse_args()
    cfg = json.loads(a.config.read_text())
    prep = json.loads((a.preparation / "preflight.json").read_text())
    if prep["status"] != "PASS" or prep["config_sha256"] != sha256_file(a.config):
        raise ValueError("preflight/config mismatch")
    audio_path = a.preparation / "audio_receipt.json"
    if sha256_file(audio_path) != prep["audio_receipt_sha256"]:
        raise ValueError("audio identity receipt changed")
    identities = {}
    for row in json.loads(audio_path.read_text()):
        fingerprint = row["pcm_sha256"]
        identities[(row["split"], row["row_index"])] = fingerprint
        identities[row["resolved_path"]] = fingerprint
    references = {}
    for surface, spec in cfg["surfaces"].items():
        path = a.asset_root / spec["path"]
        if sha256_file(path) != spec["sha256"]:
            raise ValueError("reference manifest changed")
        references[surface] = read_jsonl(path)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    matrix = {"version": 2, "selection_surface": "validation", "surfaces": {}, "inputs": {}}
    selection = {}
    if a.mode == "lm":
        if a.candidate_root is None:
            raise ValueError("LM requires native candidate root")
        split_path = a.preparation / "lm_split.json"
        if sha256_file(split_path) != prep["lm_split_sha256"]:
            raise ValueError("LM split changed")
        split = json.loads(split_path.read_text())
        lm = CharacterNgramLM(order=5, alpha=.1)
        lm.fit(split["train"])
        providers = {"ngram": lambda texts: [lm.score(text) for text in texts]}
        for directory in a.neural_lm:
            import torch
            from cantonese_asr.p2_neural_lm import CharacterLSTM, score_texts
            receipt = json.loads((directory / "COMPLETE.json").read_text())
            checkpoint = Path(receipt["selected_checkpoint"])
            if receipt["split_sha256"] != sha256_file(split_path) or sha256_file(checkpoint) != receipt["sha256"]:
                raise ValueError("neural LM provenance mismatch")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = CharacterLSTM(len(saved["vocabulary"]), **saved["architecture"])
            model.load_state_dict(saved["model"])
            key = f"lstm_s{saved['seed']}"
            if key in providers:
                raise ValueError("duplicate neural seed")
            providers[key] = lambda texts, m=model, v=saved["vocabulary"]: score_texts(m, [normalize_prediction(t) for t in texts], v, torch.device("cpu"))
        matrix["neural_seeds_complete"] = {"lstm_s42", "lstm_s43"} <= set(providers)
        matrix["inputs"]["lm_split_sha256"] = sha256_file(split_path)
    else:
        if a.model_roots is None:
            raise ValueError("fusion requires three model roots")
        val_refs = references["validation"]
        field = cfg["surfaces"]["validation"]["reference_field"]
        val_metrics = [metrics([str(row[field]) for row in val_refs], [str(row["pred_text"]) for row in
            aligned(root/"validation"/"predictions.jsonl", val_refs, identities, "validation")]) for root in a.model_roots]
        ordering = sorted(range(3), key=lambda i: (metric_key(val_metrics[i]), str(a.model_roots[i])))
        priorities = [ordering.index(i) for i in range(3)]
        matrix["validation_model_ranks"] = priorities
    for surface in ("validation", "public", "ood"):
        reference_rows = references[surface]
        field = cfg["surfaces"][surface]["reference_field"]
        refs = [str(row[field]) for row in reference_rows]
        outputs = {}
        source_tokens = {}
        started = time.perf_counter()
        if a.mode == "lm":
            root = a.candidate_root / surface
            receipt = json.loads((root / "evaluation_receipt.json").read_text())
            if receipt.get("decoder") != "P2_NATIVE5" or not receipt.get("complete"):
                raise ValueError("legacy/incomplete candidates are forbidden")
            sidecars = aligned(root/"generation_tokens.jsonl", reference_rows, identities, surface)
            matrix["inputs"][surface] = {"tokens_sha256": sha256_file(root/"generation_tokens.jsonl")}
            candidates = [row["candidates"] for row in sidecars]
            if any(len(row) != 5 or [c["rank"] for c in row] != list(range(5)) for row in candidates):
                raise ValueError("invalid native five-best")
            if any(not math.isfinite(c["sequence_score"]) or "token_ids" not in c for row in candidates for c in row):
                raise ValueError("missing native scores/tokens")
            flat = [str(c["text"]) for row in candidates for c in row]
            lm_scores = {name: provider(flat) for name, provider in providers.items()}
            outputs.update(top1=[], mbr=[], oracle=[])
            chosen_indices = {name: [] for name in outputs}
            for name in providers:
                weights = cfg["lm"]["lambdas"] if surface == "validation" else [selection[name]]
                for weight in weights:
                    key = f"{name}_lambda{weight}"
                    outputs[key] = []
                    chosen_indices[key] = []
            for i, row in enumerate(candidates):
                texts = [str(c["text"]) for c in row]
                scores = [float(c["sequence_score"]) for c in row]
                choices = {"top1": 0, "mbr": raw_mbr(texts, [-v for v in scores]),
                    "oracle": min(range(5), key=lambda j: (levenshtein_ops(list(normalize_reference(refs[i])), list(normalize_prediction(texts[j])))[0], j))}
                for name in providers:
                    weights = cfg["lm"]["lambdas"] if surface == "validation" else [selection[name]]
                    for weight in weights:
                        choices[f"{name}_lambda{weight}"] = rerank_candidates(texts, scores, lm_scores[name][i*5:i*5+5], weight)
                for name, selected in choices.items():
                    outputs[name].append(texts[selected])
                    chosen_indices[name].append(selected)
            source_tokens = {name: [candidates[i][j] for i, j in enumerate(indices)] for name, indices in chosen_indices.items()}
            if surface == "validation":
                for name in providers:
                    weight = min(cfg["lm"]["lambdas"], key=lambda w: (*metric_key(metrics(refs, outputs[f"{name}_lambda{w}"])), w))
                    selection[name] = weight
                atomic_json(a.output_dir / "selection.json", {"surface": "validation", "lambdas": selection})
        else:
            systems = [aligned(root/surface/"predictions.jsonl", reference_rows, identities, surface) for root in a.model_roots]
            outputs.update({f"model_{i}": [str(row["pred_text"]) for row in system] for i, system in enumerate(systems)})
            outputs.update(mbr=[], rover=[], oracle=[])
            for i, reference in enumerate(refs):
                hypotheses = [str(system[i]["pred_text"]) for system in systems]
                anchor = raw_mbr(hypotheses, priorities)
                outputs["mbr"].append(hypotheses[anchor])
                outputs["rover"].append(rover_anchor_vote(hypotheses, anchor))
                outputs["oracle"].append(min(hypotheses, key=lambda text: levenshtein_ops(list(normalize_reference(reference)), list(normalize_prediction(text)))[0]))
            if surface == "validation":
                selection["fusion_method"] = min(("mbr", "rover"), key=lambda key: (metric_key(metrics(refs, outputs[key])), key))
                atomic_json(a.output_dir / "selection.json", {"surface": "validation", **selection})
        matrix["surfaces"][surface] = {name: emit(a.output_dir/surface/name, reference_rows, field, texts,
            source_tokens.get(name, [None]*len(texts)), name) for name, texts in outputs.items()}
        matrix["surfaces"][surface]["analysis_seconds"] = time.perf_counter() - started
    matrix["selection"] = selection
    matrix["config_sha256"] = sha256_file(a.config)
    atomic_json(a.output_dir / f"{a.mode}_matrix.json", matrix)


if __name__ == "__main__":
    main()
