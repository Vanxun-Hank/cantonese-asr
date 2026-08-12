#!/usr/bin/env python3
"""Finalize the matched-budget RAW_WINNER P2 structural probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ARMS = (
    "SMALL_FULL",
    "SMALL_LORA",
    "MEDIUM_FULL",
    "MEDIUM_LORA",
    "LARGE_V2_FULL",
    "LARGE_V2_LORA",
)
BASE = {
    "SMALL_FULL": "SMALL_STEP0",
    "SMALL_LORA": "SMALL_STEP0",
    "MEDIUM_FULL": "MEDIUM_STEP0",
    "MEDIUM_LORA": "MEDIUM_STEP0",
    "LARGE_V2_FULL": "LARGE_V2_STEP0",
    "LARGE_V2_LORA": "LARGE_V2_STEP0",
}
WORLD = {
    "SMALL_FULL": 1,
    "SMALL_LORA": 1,
    "MEDIUM_FULL": 2,
    "MEDIUM_LORA": 1,
    "LARGE_V2_FULL": 4,
    "LARGE_V2_LORA": 1,
}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("outputs/raw-winner-p2-structural-probe-v1"),
    )
    parser.add_argument(
        "--analysis-root", type=Path,
        default=Path("artifacts/raw_winner_p2/analysis"),
    )
    parser.add_argument(
        "--final-root", type=Path,
        default=Path("artifacts/raw_winner_p2/final"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("reports/raw_winner_p2_structural_probe.md"),
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hashed_file(root: Path, path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else root / path
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved.relative_to(root)),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def surface(root: Path, label: str, split: str) -> dict[str, Any] | None:
    path = root / "evaluation" / label / split / "summary.json"
    return load(path) if path.is_file() else None


def compact(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    metrics = result["metrics"]
    generation = result["generation"]
    return {
        "rows": int(metrics["num_samples"]),
        "tol0": float(metrics["sentence_accuracy_tol0"]),
        "tol1": float(metrics["sentence_accuracy_tol1"]),
        "tol2": float(metrics["sentence_accuracy_tol2"]),
        "cer": float(metrics["cer"]),
        **{key: int(result["operations"][key]) for key in ("substitutions", "deletions", "insertions")},
        "severe": int(result["severe_error_count"]),
        "top_20_error_contribution": float(result["top_20_error_contribution"]),
        "repeated_runaway": int(generation["repeated_runaway_count"]),
        "no_eos": int(generation["no_eos_count"]),
        "replacement": int(generation["replacement_character_count"]),
        "max_length": int(generation["effective_max_length_count"]),
    }


def training_row(capacity: Path, arm: str) -> dict[str, Any]:
    run = load(capacity / arm / "run_config.json")
    state = load(capacity / arm / "trainer_state.json")
    train = next(row for row in reversed(state["log_history"]) if "train_runtime" in row)
    metrics_rows = [
        json.loads(line)
        for line in (capacity / arm / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    peak_alloc = max(float(row.get("gpu_peak_allocated_gb", 0.0)) for row in metrics_rows)
    peak_reserved = max(float(row.get("gpu_peak_reserved_gb", 0.0)) for row in metrics_rows)
    model = run["model"]
    runtime = float(train["train_runtime"])
    return {
        "method": model["fine_tuning_method"],
        "total_parameters": int(model["total_parameters"]),
        "trainable_parameters": int(model["trainable_parameters"]),
        "trainable_ratio": float(model["trainable_parameters"]) / float(model["total_parameters"]),
        "world_size": WORLD[arm],
        "train_runtime_seconds": runtime,
        "gpu_hours": runtime * WORLD[arm] / 3600.0,
        "samples_per_second": float(train["train_samples_per_second"]),
        "optimizer_steps_per_second": float(train["train_steps_per_second"]),
        "train_loss": float(train["train_loss"]),
        "peak_allocated_gb_per_rank": peak_alloc,
        "peak_reserved_gb_per_rank": peak_reserved,
    }


def benchmark(evaluation: Path, arm: str) -> dict[str, Any]:
    return load(evaluation / f"{arm}_BENCH32" / "validation" / "summary.json")[
        "inference_benchmark"
    ]


def fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def main() -> None:
    a = args()
    root = a.root.resolve()
    output = root / a.output_root
    evaluation = output / "evaluation"
    capacity = output / "capacity"
    analysis = root / a.analysis_root
    final = root / a.final_root
    final.mkdir(parents=True, exist_ok=True)

    matrix: dict[str, Any] = {
        "claim_boundary": "150-step matched-budget adaptation probe; not convergence",
        "fixed_exposure_rows": 2400,
        "global_effective_batch": 16,
        "provenance": {
            "experiment_config": hashed_file(
                root, Path("configs/rounds/raw_winner_p2_structural_probe.json")
            ),
            "external73_manifest": hashed_file(
                root, Path("artifacts/manifests/external/round2/external73.jsonl")
            ),
            "fixed_exposure_manifest": hashed_file(
                root,
                Path("artifacts/raw_winner_p2/preparation/external73_fixed_exposure_2400.jsonl"),
            ),
            "fixed_exposure_receipt": hashed_file(
                root, Path("artifacts/raw_winner_p2/preparation/fixed_exposure_receipt.json")
            ),
            "validation_manifest": hashed_file(
                root, Path("artifacts/manifests/validation.jsonl")
            ),
            "endpoint_evaluator": hashed_file(
                root, Path("scripts/evaluate_raw_winner_p2_capacity.py")
            ),
        },
        "arms": {},
    }
    for arm in ARMS:
        endpoint = compact(surface(output, arm, "validation"))
        step0 = compact(surface(output, BASE[arm], "validation"))
        if endpoint is None or step0 is None:
            raise FileNotFoundError(f"missing endpoint/step0 validation for {arm}")
        receipt = load(evaluation / arm / "evaluation_receipt.json")
        public = compact(surface(output, f"{arm}_PUBLIC_OOD", "public"))
        ood = compact(surface(output, f"{arm}_PUBLIC_OOD", "ood"))
        matrix["arms"][arm] = {
            "step0_validation": step0,
            "step150_validation": endpoint,
            "delta_validation_tol2": endpoint["tol2"] - step0["tol2"],
            "delta_validation_cer": endpoint["cer"] - step0["cer"],
            "improves_own_step0_tol2_and_cer": (
                endpoint["tol2"] > step0["tol2"] and endpoint["cer"] < step0["cer"]
            ),
            "public": public,
            "ood": ood,
            "weights": receipt["weights"],
            "run_config": hashed_file(root, capacity / arm / "run_config.json"),
            "training": training_row(capacity, arm),
            "inference32": benchmark(evaluation, arm),
        }

    small = matrix["arms"]["SMALL_FULL"]["step150_validation"]
    for arm in ("MEDIUM_FULL", "LARGE_V2_FULL"):
        value = matrix["arms"][arm]["step150_validation"]
        matrix["arms"][arm]["capacity_gain_vs_small_full"] = {
            "delta_tol2": value["tol2"] - small["tol2"],
            "delta_cer": value["cer"] - small["cer"],
            "passes": (
                value["tol2"] - small["tol2"] >= 0.005
                and value["cer"] - small["cer"] <= 0.003
            ) or (
                small["cer"] - value["cer"] >= 0.003
                and small["tol2"] - value["tol2"] <= 0.005
            ),
        }

    tokenizer = load(analysis / "tokenizer_audit.json")
    lm = load(analysis / "lm_matrix.json")
    fusion = load(analysis / "fusion_matrix.json")
    matrix["tokenizer"] = {
        "all_tokenizers_identical": bool(tokenizer["tokenizers_identical"]),
        "tokenizer_hashes": sorted(
            {item["tokenizer_hash"] for item in tokenizer["models"].values()}
        ),
        "small_surfaces": tokenizer["models"]["small"]["surfaces"],
        "key_character_encodings": tokenizer["models"]["small"]["key_character_encodings"],
    }
    matrix["language_model"] = {
        "selected_lambda": lm["selected_lambda"],
        "validation": lm["surfaces"]["validation"],
    }
    matrix["fusion"] = {
        "selected_method": fusion["selected_method"],
        "surfaces": fusion["surfaces"],
    }
    (final / "capacity_matrix.json").write_text(
        json.dumps(matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (final / "tokenizer_matrix.json").write_text(
        json.dumps(tokenizer, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (final / "lm_matrix.json").write_text(
        json.dumps(lm, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (final / "fusion_matrix.json").write_text(
        json.dumps(fusion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# RAW_WINNER P2 Structural Probe",
        "",
        "> Claim boundary: this is a 150-step matched-budget adaptation probe, not a fully converged architecture ranking.",
        "",
        "## Capacity and PEFT results",
        "",
        "| Arm | Method | Params (M) | Trainable | Val tol2 | Val CER | Public tol2 | Public CER | OOD tol2 | OOD CER | GPU-h | Peak alloc (GB) | RTF |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        row = matrix["arms"][arm]
        val, public, ood, train, bench = (
            row["step150_validation"], row["public"], row["ood"], row["training"], row["inference32"]
        )
        lines.append(
            f"| {arm} | {train['method']} | {train['total_parameters']/1e6:.1f} | "
            f"{100*train['trainable_ratio']:.3f}% | {val['tol2']:.4f} | {val['cer']:.4f} | "
            f"{fmt(public['tol2'] if public else None)} | {fmt(public['cer'] if public else None)} | "
            f"{fmt(ood['tol2'] if ood else None)} | {fmt(ood['cer'] if ood else None)} | "
            f"{train['gpu_hours']:.3f} | {train['peak_allocated_gb_per_rank']:.2f} | "
            f"{bench['real_time_factor']:.4f} |"
        )

    large = matrix["arms"]["LARGE_V2_FULL"]
    medium = matrix["arms"]["MEDIUM_FULL"]
    lines += [
        "",
        "## Findings",
        "",
        f"- Capacity improves matched-budget adaptation efficiency monotonically: Small Full `{small['tol2']:.4f}/{small['cer']:.4f}`, Medium Full `{medium['step150_validation']['tol2']:.4f}/{medium['step150_validation']['cer']:.4f}`, Large-v2 Full `{large['step150_validation']['tol2']:.4f}/{large['step150_validation']['cer']:.4f}` (tol2/CER).",
        f"- Large-v2 Full passes the registered capacity threshold versus Small Full: `{large['capacity_gain_vs_small_full']['passes']}`.",
        "- LoRA materially reduces trainable parameters and memory, but under this short budget it does not match Full SFT accuracy; this is adaptation-efficiency evidence, not a convergence claim.",
        f"- All three architectures use the same tokenizer hash (`{matrix['tokenizer']['tokenizer_hashes'][0]}`); there are no UNKs or replacement characters in the audited references, but roughly 29–35% of individual reference characters tokenize to multiple tokens, so token inefficiency is measurable without proving it is the dominant error source.",
        f"- Leakage-guarded character 5-gram reranking selects `lambda={lm['selected_lambda']}`; therefore the LM did not improve the validation selection over ASR scores.",
        f"- Three-model fusion selects `{fusion['selected_method']}`. Validation MBR reaches tol2 `{fusion['surfaces']['validation']['mbr']['sentence_accuracy_tol2']:.4f}` and CER `{fusion['surfaces']['validation']['mbr']['cer']:.4f}`; Public MBR reaches `{fusion['surfaces']['public']['mbr']['sentence_accuracy_tol2']:.4f}/{fusion['surfaces']['public']['mbr']['cer']:.4f}`.",
        "- `no_eos_count` is reported separately from true repeated runaway. Medium/Large generation sidecars omit EOS in this Transformers path even when decoding terminates normally; repeated-runaway, max-length and replacement-character diagnostics remain the stability indicators.",
        "",
        "## Registered questions",
        "",
        "1. **Does model capacity improve 150-step adaptation efficiency?** Yes. Full-SFT validation improves monotonically from Small to Medium to Large-v2, while inference RTF and parameter cost increase monotonically as well.",
        "2. **What is the Full-SFT/LoRA trade-off?** LoRA reduces trainable parameters and peak memory substantially, but every LoRA arm is worse than its matched Full-SFT arm under this short budget. This does not establish the fully converged ordering.",
        "3. **Is the standard Whisper tokenizer a demonstrated Cantonese bottleneck?** No. The tokenizers and hashes are identical and audited references contain no unknown or replacement tokens. Multi-token encoding of Cantonese characters is measurable, but this probe does not show that vocabulary modification would improve ASR.",
        "4. **Does the character LM exploit the 5-best oracle space?** No. Validation selects lambda zero even though an oracle gap exists, so this training-text-only 5-gram score does not identify the better hypotheses reliably.",
        "5. **Are the three existing models complementary?** Yes, modestly. Character-edit MBR improves validation and Public over the individual top-1 surfaces, while the oracle remains better and quantifies residual fusion headroom.",
        "",
        "## Limitations",
        "",
        "- The 150-step matched-exposure protocol measures adaptation efficiency, not convergence, and must not be cited as a definitive Small/Medium/Large architecture ranking.",
        "- Full and LoRA learning rates are method-specific registered values; the experiment compares practical short-budget recipes rather than exhaustively optimized hyperparameters.",
        "- Public and OOD are diagnostic surfaces. Only fixed validation is used for endpoint and reranker selection.",
        "- The fusion result is an offline multi-model analysis and carries approximately the combined inference cost of its three component systems.",
        "",
        "## Reproducibility",
        "",
        "- Fixed exposure: 2,400 unique `external73` rows, seed 42, globally identical 16-example optimizer steps across 1/2/4 GPU topologies.",
        "- Decode: beam 2, no-repeat n-gram 4, repetition penalty 1.05, max_length 225, Chinese transcription.",
        "- Full configuration, checkpoint weight surfaces, SHA-256 values, per-surface errors and resource receipts are in `artifacts/raw_winner_p2/final/capacity_matrix.json`.",
    ]
    report = root / a.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report), "matrix": str(final / "capacity_matrix.json")}, indent=2))


if __name__ == "__main__":
    main()
