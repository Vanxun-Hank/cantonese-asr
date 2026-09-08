#!/usr/bin/env python3
"""Build a source-conditional exposure stream (Design A) for the P2 arms.

W500_ADAPTIVE_METHOD.md requires the source to be fixed for a whole optimizer
step -- "Sources are never mixed inside one optimizer step" -- because updating
the decoder on external audio changes Cantonese word choice, insertion and EOS
behaviour. external73.jsonl is shuffled, so its 16-example entries mix official,
CV and MDCC and cannot be used as-is.

Design A (chosen over 1:1 alternation): keep STRICT single-pass exposure -- every
row is seen exactly once per epoch, same as the baseline arms -- and let the
alternation ratio fall out of the data (official 6292 : external 17012 = 1:2.70).
1:1 alternation would have required oversampling official, which breaks the
matched-exposure premise that every baseline comparison rests on.

Step accounting differs slightly from the baseline and that is unavoidable:
  official  6292 = 393*16 + 4  -> 394 steps
  external 17012 = 1063*16 + 4 -> 1064 steps
  total 1458 steps/epoch vs the baseline's 1457 (23304 = 1456*16 + 8)
Neither source count is divisible by 16, so two tails exist instead of one. Over
3 epochs that is 4374 steps vs 4371 -- 0.07% along the same 7285-step cosine
horizon, and exposure per row is identical.

Official steps are spread with Bresenham so they never clump.
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cantonese_asr.io import read_jsonl
from cantonese_asr.training_sampling import source_group


def chunk(seq: list[int], size: int) -> list[list[int]]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def interleave(official: list[list[int]], external: list[list[int]]) -> list[tuple[str, list[int]]]:
    """Bresenham merge so official steps are spread evenly through the epoch."""
    out, oi, xi = [], 0, 0
    total = len(official) + len(external)
    for k in range(total):
        # place an official step when its running share falls behind
        want_official = (oi * total) <= (k * len(official))
        if want_official and oi < len(official):
            out.append(("official", official[oi])); oi += 1
        elif xi < len(external):
            out.append(("external", external[xi])); xi += 1
        else:
            out.append(("official", official[oi])); oi += 1
    assert oi == len(official) and xi == len(external)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    rows = read_jsonl(a.manifest)
    groups = [source_group(r) for r in rows]
    n_off = sum(g == "official" for g in groups)
    print(f"[data] {len(rows)} rows  official={n_off}  external={len(rows) - n_off}")

    stream: list[dict] = []
    for epoch in range(a.epochs):
        idx = list(range(len(rows)))
        # same shuffle convention as cantonese_asr.p2_full.epoch_batches
        random.Random(f"p2-full-sc:{a.seed}:epoch:{epoch}").shuffle(idx)
        off = [i for i in idx if groups[i] == "official"]
        ext = [i for i in idx if groups[i] != "official"]
        merged = interleave(chunk(off, a.batch), chunk(ext, a.batch))
        for offset, (mode, indices) in enumerate(merged):
            stream.append({
                "step": len(stream) + 1,
                "epoch": epoch + 1,
                "offset": offset,
                "source_mode": mode,
                "indices": indices,
            })
        if epoch == 0:
            per = [e for e in stream if e["epoch"] == 1]
            o = sum(1 for e in per if e["source_mode"] == "official")
            print(f"[epoch1] {len(per)} steps  official={o}  external={len(per)-o}  ratio=1:{(len(per)-o)/o:.2f}")
            seen = sorted(i for e in per for i in e["indices"])
            assert seen == list(range(len(rows))), "epoch is not a strict single pass"
            print(f"[check] strict single pass over all {len(rows)} rows: OK")
            mixed = [e for e in per if len({groups[i] != "official" for i in e["indices"]}) > 1]
            assert not mixed, f"{len(mixed)} steps mix sources"
            print("[check] no step mixes official with external: OK")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(stream) , encoding="utf-8")
    print(f"[done] {len(stream)} steps over {a.epochs} epochs -> {a.out}")


if __name__ == "__main__":
    main()
