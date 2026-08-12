#!/usr/bin/env python3
"""Run leakage-guarded character-LM reranking and deterministic model fusion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cantonese_asr.io import read_jsonl, sha256_file
from cantonese_asr.metrics import compute_diagnostic_metrics, compute_official_metrics, levenshtein_ops, normalize_prediction, normalize_reference
from cantonese_asr.p2_analysis import CharacterNgramLM, mbr_medoid, rerank_candidates, rover_anchor_vote


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--train-manifest",type=Path,required=True)
    p.add_argument("--validation-reference",type=Path,required=True); p.add_argument("--public-reference",type=Path,required=True); p.add_argument("--ood-reference",type=Path,required=True)
    p.add_argument("--d7-validation-root",type=Path,required=True); p.add_argument("--d7-diagnostics-root",type=Path,required=True); p.add_argument("--raw-root",type=Path,required=True); p.add_argument("--noise-root",type=Path,required=True); p.add_argument("--full-root",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); return p.parse_args()


def text(row):
    for key in ("text","ref_text","text:label","reference"):
        if key in row: return str(row[key])
    raise ValueError("reference missing")


def refs(path): return [(str(r["audio_path"]),text(r)) for r in read_jsonl(path)]


def same_audio(left: str, right: str) -> bool:
    return Path(left).name == Path(right).name


def metrics(references,predictions):
    m=compute_official_metrics(references,predictions); m["diagnostics"]=compute_diagnostic_metrics(references,predictions); return m


def aligned_predictions(path, reference_rows):
    rows=read_jsonl(path)
    if len(rows)!=len(reference_rows): raise ValueError(f"row mismatch: {path}")
    values=[]
    for row,(audio,_) in zip(rows,reference_rows):
        if not same_audio(str(row["audio_path"]), audio): raise ValueError(f"audio order mismatch: {path}")
        values.append(str(row["pred_text"]))
    return values


def main():
    a=parse_args(); cfg=json.loads(a.config.read_text()); a.output_dir.mkdir(parents=True,exist_ok=True)
    lm=CharacterNgramLM(order=cfg["lm"]["order"],alpha=cfg["lm"]["alpha"]); lm.fit(text(r) for r in read_jsonl(a.train_manifest))
    lm_matrix={"training_manifest":str(a.train_manifest),"training_manifest_sha256":sha256_file(a.train_manifest),"selection_surface":"validation","lambdas":{}}
    chosen=None; chosen_key=None
    for split,reference_path in (("validation",a.validation_reference),("public",a.public_reference),("ood",a.ood_reference)):
        reference_rows=refs(reference_path)
        d7_root = a.d7_validation_root if split == "validation" else a.d7_diagnostics_root
        sidecars=read_jsonl(d7_root/split/"generation_tokens.jsonl")
        if len(sidecars)!=len(reference_rows): raise ValueError(f"D7 row mismatch {split}")
        outputs={weight:[] for weight in cfg["lm"]["lambdas"]}; oracle=[]; mbr=[]; top1=[]
        for sidecar,(audio,reference) in zip(sidecars,reference_rows):
            if not same_audio(str(sidecar["audio_path"]), audio): raise ValueError(f"D7 order mismatch {split}")
            candidates=sidecar["candidates"]; texts=[str(c["text"]) for c in candidates]; scores=[float(c["sequence_score"]) for c in candidates]; lm_scores=[lm.score(x) for x in texts]
            for weight in outputs: outputs[weight].append(texts[rerank_candidates(texts,scores,lm_scores,float(weight))])
            top1.append(texts[0]); mbr.append(str(sidecar["pred_text"])); oracle.append(min(texts,key=lambda x:levenshtein_ops(list(normalize_reference(reference)),list(normalize_prediction(x)))[0]))
        if split=="validation":
            for weight,preds in outputs.items():
                row=metrics([r for _,r in reference_rows],preds); lm_matrix["lambdas"][str(weight)]=row
                diag=row.get("diagnostics",{}); key=(-float(row["sentence_accuracy_tol2"]),float(row["cer"]),int(diag.get("severe_error_count",0)),float(weight))
                if chosen_key is None or key<chosen_key: chosen_key=key; chosen=float(weight)
        lm_matrix.setdefault("surfaces",{})[split]={"top1":metrics([r for _,r in reference_rows],top1),"mbr":metrics([r for _,r in reference_rows],mbr),"oracle":metrics([r for _,r in reference_rows],oracle),"lm_by_lambda":{str(w):metrics([r for _,r in reference_rows],p) for w,p in outputs.items()}}
    lm_matrix["selected_lambda"]=chosen
    fusion={"selection_surface":"validation","models":["RAW_WINNER","NOISE_S43","FULL_LR1E6_S43"],"surfaces":{}}
    model_roots=[a.raw_root,a.noise_root,a.full_root]
    validation_metrics=[]
    for root in model_roots:
        validation_metrics.append(json.loads((root/"validation"/"metrics.json").read_text()))
    ranks=sorted(range(3),key=lambda i:(-float(validation_metrics[i]["sentence_accuracy_tol2"]),float(validation_metrics[i]["cer"])))
    model_rank=[ranks.index(i) for i in range(3)]
    for split,reference_path in (("validation",a.validation_reference),("public",a.public_reference),("ood",a.ood_reference)):
        reference_rows=refs(reference_path); model_preds=[aligned_predictions(root/split/"predictions.jsonl",reference_rows) for root in model_roots]
        mbr=[]; rover=[]; oracle=[]
        for row_index,(_,reference) in enumerate(reference_rows):
            hypotheses=[preds[row_index] for preds in model_preds]; anchor=mbr_medoid(hypotheses,model_rank); mbr.append(hypotheses[anchor]); rover.append(rover_anchor_vote(hypotheses,anchor)); oracle.append(min(hypotheses,key=lambda x:levenshtein_ops(list(normalize_reference(reference)),list(normalize_prediction(x)))[0]))
        fusion["surfaces"][split]={"models":{name:metrics([r for _,r in reference_rows],preds) for name,preds in zip(fusion["models"],model_preds)},"mbr":metrics([r for _,r in reference_rows],mbr),"rover":metrics([r for _,r in reference_rows],rover),"oracle":metrics([r for _,r in reference_rows],oracle)}
    val=fusion["surfaces"]["validation"]; fusion["selected_method"]=min(("mbr","rover"),key=lambda name:(-float(val[name]["sentence_accuracy_tol2"]),float(val[name]["cer"]),int(val[name]["diagnostics"].get("severe_error_count",0))))
    (a.output_dir/"lm_matrix.json").write_text(json.dumps(lm_matrix,ensure_ascii=False,indent=2)+"\n"); (a.output_dir/"fusion_matrix.json").write_text(json.dumps(fusion,ensure_ascii=False,indent=2)+"\n"); print(json.dumps({"selected_lambda":chosen,"fusion":fusion["selected_method"]},indent=2))


if __name__=="__main__": main()
