from cantonese_asr.p2_analysis import (
    CharacterNgramLM,
    mbr_medoid,
    rerank_candidates,
    rover_anchor_vote,
)


def test_lm_rerank_is_deterministic() -> None:
    lm = CharacterNgramLM(order=3, alpha=0.1)
    lm.fit(["我唔知道", "我唔知道", "我不知道"])
    texts = ["我不知道", "我唔知道"]
    scores = [0.0, 0.0]
    first = rerank_candidates(texts, scores, [lm.score(x) for x in texts], 0.4)
    assert first == rerank_candidates(texts, scores, [lm.score(x) for x in texts], 0.4)


def test_mbr_and_rover_are_deterministic() -> None:
    hypotheses = ["今日天气好", "今日天气好", "今日天氣好"]
    assert mbr_medoid(hypotheses, [0, 1, 2]) == 0
    assert rover_anchor_vote(hypotheses, 0) == "今日天气好"
