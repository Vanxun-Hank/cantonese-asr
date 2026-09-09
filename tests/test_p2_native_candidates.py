from types import SimpleNamespace
import json
import numpy as np
import torch

from scripts import evaluate_raw_winner_decode as evaluation


def test_native_five_candidates_preserved_without_anchor(monkeypatch, tmp_path):
    paths = [{"audio_path": "audio.wav"}]
    refs = [{"audio_path": "audio.wav", "text": "a"}]
    monkeypatch.setattr(evaluation, "read_test_rows", lambda _: paths)
    monkeypatch.setattr(evaluation, "read_references", lambda *_: refs)
    monkeypatch.setattr(evaluation, "resolve_surface_audio", lambda *_: tmp_path/"audio.wav")
    monkeypatch.setattr(evaluation.librosa, "load", lambda *a, **k: (np.zeros(16000), 16000))
    monkeypatch.setattr(evaluation.librosa, "get_duration", lambda **k: 1.0)
    class Features(dict):
        input_features = torch.zeros(1, 80, 3000)
    class Processor:
        tokenizer = SimpleNamespace(eos_token_id=0, all_special_ids=[0,1,2,3,4])
        def feature_extractor(self, *a, **k):
            return Features(attention_mask=torch.ones(1,3000))
        def get_decoder_prompt_ids(self, **k):
            return [(1,2),(2,3),(3,4)]
        def batch_decode(self, ids, **k):
            return ["a", "ab", "ac", "ad", "ae"]
    sequences = torch.tensor([[1,2,3,4,8+i,0] for i in range(5)])
    class Model:
        generation_config = SimpleNamespace(eos_token_id=0)
        config = SimpleNamespace(decoder_start_token_id=1)
        def generate(self, *a, **k):
            assert k["num_return_sequences"] == 5
            assert k["max_length"] == 225
            assert k["num_beams"] == 5
            return SimpleNamespace(sequences=sequences, sequences_scores=torch.tensor([-.1,-.2,-.3,-.4,-.5]))
    output = tmp_path/"result"
    summary = evaluation.evaluate_surface(surface="validation", manifest=tmp_path/"manifest.jsonl",
        reference_field="text", model=Model(), processor=Processor(), arm="P2_NATIVE5",
        project_root=tmp_path, output_dir=output, batch_size=1, generation_max_length=225,
        prompt_token_count=4, device=torch.device("cpu"), dtype=torch.float32,
        max_samples=None, mbr_anchor_dir=None, diagnostic_version=2)
    rows = [json.loads(line) for line in (output/"generation_tokens.jsonl").read_text().splitlines()]
    candidates = rows[0]["candidates"]
    assert [candidate["text"] for candidate in candidates] == ["a","ab","ac","ad","ae"]
    assert [candidate["token_ids"] for candidate in candidates] == sequences.tolist()
    assert rows[0]["effective_total_length"] == 6
    assert rows[0]["generated_token_count"] == 2
    assert summary["diagnostic_version"] == 2
    assert summary["metrics"]["cer"] == 0
