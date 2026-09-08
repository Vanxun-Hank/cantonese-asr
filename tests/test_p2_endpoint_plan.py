import json
from pathlib import Path
import pytest
from scripts.plan_p2_full_endpoints import build_plan
from cantonese_asr.io import sha256_file


def config():
    return json.loads((Path(__file__).resolve().parents[1]/"configs/rounds/raw_winner_p2_full.json").read_text())


def test_missing_evidence_cannot_produce_selection_or_extension(tmp_path):
    result = build_plan(config(), "expected", tmp_path)
    assert result["status"] == "PARTIAL"
    assert len(result["missing"]) == 72
    assert result["selected3"] == result["extension_decisions"] == {}
    assert result["evaluation_tasks"] == []
    assert result["extension_submission_allowed"] is False


def test_public_receipt_cannot_masquerade_as_validation(tmp_path):
    p = tmp_path/"SMALL_FULL_S42/train/diagnostics/checkpoint-729/validation/evaluation_receipt.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"step":729,"surface":"public","complete":True}))
    with pytest.raises(ValueError, match="invalid selection evidence"):
        build_plan(config(), "expected", tmp_path)


def test_selected_and_endpoint_both_retained_without_public_selection(tmp_path):
    cfg = config()
    cfg["arms"] = {"SMALL_FULL": cfg["arms"]["SMALL_FULL"]}
    for seed in (42, 43):
        root = tmp_path/f"SMALL_FULL_S{seed}/train"
        for step in cfg["training"]["save_steps"][:6]:
            directory = root/f"diagnostics/checkpoint-{step}/validation"
            directory.mkdir(parents=True)
            files = []
            for name in ("predictions.jsonl", "generation_tokens.jsonl"):
                path = directory/name
                path.write_text('{}\n'*702)
                files.append({"path":name,"sha256":sha256_file(path)})
            row = {"step":step,"surface":"validation","complete":True,"integrity_pass":True,
                "decoder":"D0_CURRENT","config_sha256":"expected",
                "manifest_sha256":cfg["surfaces"]["validation"]["sha256"],
                "tol2":.85,"cer":.08 if step==3643 else .10,"severe":1,
                "repeated_runaway":0,"replacement":0,"max_length":0,"files":files}
            (directory/"evaluation_receipt.json").write_text(json.dumps(row))
            checkpoint = root/f"training/checkpoint-{step}"
            checkpoint.mkdir(parents=True)
            (checkpoint/"COMPLETE.json").write_text(json.dumps({"arm":"SMALL_FULL","seed":seed,"step":step,"phase":"train"}))
        # Public data cannot influence the selected step or extension.
        public = root/"diagnostics/checkpoint-4371/public"
        public.mkdir()
        (public/"metrics.json").write_text('{"cer":0,"sentence_accuracy_tol2":1}')
    result = build_plan(cfg, "expected", tmp_path)
    assert result["status"] == "READY"
    assert {row["step"] for row in result["selected3"].values()} == {3643}
    assert len(result["evaluation_tasks"]) == 8
    assert result["extension_decisions"]["SMALL_FULL"]["extend_both_seeds"] is True
    assert result["extension_submission_allowed"] is False
