#!/usr/bin/env python3
"""Evaluate one P2 base or adapted Whisper checkpoint on registered surfaces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import WhisperProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cantonese_asr.io import sha256_file
from cantonese_asr.model_loading import load_whisper_model
from scripts.evaluate_raw_winner_decode import evaluate_surface


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(); p.add_argument("--model-dir",type=Path,required=True); p.add_argument("--processor-dir",type=Path,required=True); p.add_argument("--project-root",type=Path,default=PROJECT_ROOT)
    p.add_argument("--validation-manifest",type=Path,required=True); p.add_argument("--public-manifest",type=Path); p.add_argument("--ood-manifest",type=Path); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--batch-size",type=int,default=1); p.add_argument("--max-samples",type=int); p.add_argument("--surfaces",nargs="+",choices=("validation","public","ood"),default=("validation",)); return p.parse_args()


def hash_weights(path: Path) -> dict:
    names=("model.safetensors","adapter_model.safetensors")
    found=[path/name for name in names if (path/name).is_file()]
    if len(found)!=1: raise ValueError(f"expected exactly one full/adapter weight file in {path}: {found}")
    return {"file":found[0].name,"sha256":sha256_file(found[0]),"bytes":found[0].stat().st_size}


def main() -> None:
    a=parse_args(); a.output_dir.mkdir(parents=True,exist_ok=False)
    device=torch.device("cuda"); dtype=torch.float16
    processor=WhisperProcessor.from_pretrained(a.processor_dir,language="zh",task="transcribe",local_files_only=True)
    model=load_whisper_model(a.model_dir,dtype=dtype).to(device); model.eval(); model.generation_config.language="zh"; model.generation_config.task="transcribe"; model.generation_config.forced_decoder_ids=None
    prompt_count=1+len(processor.get_decoder_prompt_ids(language="zh",task="transcribe",no_timestamps=True))
    manifests={"validation":(a.validation_manifest,"text"),"public":(a.public_manifest,"text"),"ood":(a.ood_manifest,"text")}
    summaries={}
    for split in a.surfaces:
        manifest,field=manifests[split]
        if manifest is None: raise ValueError(f"missing {split} manifest")
        summaries[split]=evaluate_surface(surface=split,manifest=manifest,reference_field=field,model=model,processor=processor,arm="D0_CURRENT",project_root=a.project_root,output_dir=a.output_dir/split,batch_size=a.batch_size,generation_max_length=225,prompt_token_count=prompt_count,device=device,dtype=dtype,max_samples=a.max_samples,mbr_anchor_dir=None)
    receipt={"model":str(a.model_dir.resolve()),"processor":str(a.processor_dir.resolve()),"weights":hash_weights(a.model_dir),"effective_generate":{"max_length":225,"max_new_tokens":None,"num_beams":2,"no_repeat_ngram_size":4,"repetition_penalty":1.05,"do_sample":False},"summaries":summaries}
    (a.output_dir/"evaluation_receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(a.output_dir),"weights":receipt["weights"],"surfaces":list(summaries)},indent=2))


if __name__=="__main__": main()
