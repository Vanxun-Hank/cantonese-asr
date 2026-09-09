#!/usr/bin/env python3
"""Zero-shot Public-surface evaluation under the round's registered D0_CURRENT decode.

predict.py cannot reproduce D0_CURRENT: it only sets language/task/num_beams/
max_length and omits no_repeat_ngram_size and repetition_penalty. The round's own
evaluate_raw_winner_decode.py does carry D0_CURRENT but requires --validation-manifest,
which is a derived artifact we cannot rebuild without Common Voice and MDCC.

So this reuses the package's own cantonese_asr.metrics with the registered decode
kwargs, on the Public manifest only. It is not a substitute for the round's
finalizer receipts.
"""
import argparse, json, sys, time
from pathlib import Path

import torch, librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cantonese_asr.metrics import compute_official_metrics

D0_CURRENT = dict(num_beams=2, no_repeat_ngram_size=4, repetition_penalty=1.05,
                  max_length=225, early_stopping=True)

def build_index(audio_dir: Path) -> dict[str, Path]:
    """Recursive basename index.

    template_pre.jsonl carries absolute paths from the corpus author's machine and
    the files land under data.zip -> train_raw/<range>/<name>.wav, so neither
    audio_dir/<listed> nor audio_dir/<basename> resolves. Basenames are unique
    across the extraction (verified: 1900/1900 hit, 0 collisions).
    """
    idx: dict[str, Path] = {}
    for p in audio_dir.rglob("*.wav"):
        idx.setdefault(p.name, p)
    return idx

def resolve(idx: dict[str, Path], listed: str) -> Path:
    c = Path(listed)
    hit = idx.get(c.name)
    if hit is None:
        raise FileNotFoundError(listed)
    return hit

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()

    rows = [json.loads(l) for l in a.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    idx = build_index(a.audio_dir)
    print(f"[data] {len(rows)} rows from {a.manifest.name}; {len(idx)} wav indexed under {a.audio_dir}", flush=True)
    missing = [r["audio_path"] for r in rows if Path(r["audio_path"]).name not in idx]
    if missing:
        raise SystemExit(f"{len(missing)} manifest rows have no audio, e.g. {missing[:3]}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = WhisperProcessor.from_pretrained(str(a.model_dir))
    model = WhisperForConditionalGeneration.from_pretrained(
        str(a.model_dir), torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    print(f"[model] {a.model_dir.name} on {dev}", flush=True)

    preds, t0 = [], time.time()
    for i in range(0, len(rows), a.batch_size):
        chunk = rows[i:i + a.batch_size]
        audio = [librosa.load(resolve(idx, r["audio_path"]), sr=16000, mono=True)[0] for r in chunk]
        feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(dev, dtype=model.dtype)
        with torch.no_grad():
            ids = model.generate(feats, language="zh", task="transcribe", **D0_CURRENT)
        preds.extend(proc.batch_decode(ids, skip_special_tokens=True))
        if (i // a.batch_size) % 20 == 0:
            done = min(i + a.batch_size, len(rows))
            print(f"[gen ] {done}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)

    refs = [r["ref_text"] for r in rows]
    m = compute_official_metrics(refs, preds)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"decode": {**D0_CURRENT, "language": "zh", "task": "transcribe"},
                                 "model": str(a.model_dir), "metrics": m}, indent=2, ensure_ascii=False))
    print(json.dumps(m, indent=2), flush=True)
    print(f"[done] {time.time()-t0:.0f}s -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
