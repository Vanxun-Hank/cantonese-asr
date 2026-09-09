import math

import pytest

from cantonese_asr.p2_full import (cosine_factor, epoch_batches, extension_decision,
                                  rank_microbatches, select_validation, split_lm_texts)
from cantonese_asr.p2_analysis import BOS, EOS, UNK, CharacterNgramLM, rover_anchor_vote


def test_epoch_tail_and_topology_parity():
    batches = epoch_batches(23304, 42)
    assert len(batches) == 7285
    assert sum(len(b["indices"]) for b in batches[:4371]) == 69912
    for epoch in range(5):
        segment = batches[epoch * 1457:(epoch + 1) * 1457]
        assert len(segment[-1]["indices"]) == 8
        assert sorted(i for b in segment for i in b["indices"]) == list(range(23304))
    for world, batch in ((1, 8), (2, 2), (4, 1), (1, 4), (1, 1)):
        for original in (batches[0]["indices"], batches[1456]["indices"]):
            assert [i for micro in rank_microbatches(original, world, batch)
                    for rank in micro for i in rank] == original
    assert batches == epoch_batches(23304, 42)
    assert batches != epoch_batches(23304, 43)


def row(step, cer):
    return dict(surface="validation", complete=True, integrity_pass=True, step=step,
                cer=cer, tol2=0.85, severe=3, repeated_runaway=0, replacement=0, max_length=0)


def test_extension_requires_both_seeds_and_validation():
    rows = [row(step, cer) for step, cer in ((2186,.1),(2914,.099),(3643,.097),(4371,.096))]
    assert extension_decision({42: rows, 43: rows})["extend_both_seeds"]
    bad = [dict(r, cer=.11) if r["step"] > 2914 else r for r in rows]
    assert not extension_decision({42: rows, 43: bad})["extend_both_seeds"]
    with pytest.raises(ValueError):
        extension_decision({42: rows})
    with pytest.raises(ValueError):
        select_validation([dict(rows[0], surface="public")])
    missing = [dict(r) for r in rows]
    del missing[2]["replacement"]
    # Make the incomplete stability row the selected late candidate.
    missing[2]["cer"] = .09
    with pytest.raises(ValueError):
        extension_decision({42: rows, 43: missing})


def test_scheduler_does_not_restart_at_review():
    assert cosine_factor(0) == 0
    assert cosine_factor(365) == 1
    assert 0 < cosine_factor(4372) < cosine_factor(4371)
    assert cosine_factor(7285) == 0


def test_lm_split_groups_duplicates():
    texts = [str(i) for i in range(100)] * 2
    split = split_lm_texts(texts)
    assert not set(split["train"]) & set(split["dev"])
    assert split["train_rows"] + split["dev_rows"] == 200
    assert split == split_lm_texts(texts)


def test_ngram_boundary_symbols_and_oov():
    lm = CharacterNgramLM(order=5)
    lm.fit(["ab"])
    assert lm.counts[(BOS,) * 4]["a"] == 1
    assert sum(lm.context_totals.values()) == 3
    assert set(token for counter in lm.counts.values() for token in counter) <= lm.vocabulary
    assert EOS in lm.vocabulary and UNK in lm.vocabulary
    assert math.isfinite(lm.score("z"))
    assert lm.score("z") == lm.score("q")


@pytest.mark.parametrize("anchor", [0, 1, 2])
def test_rover_one_vote_per_system(anchor):
    assert rover_anchor_vote(["a", "xxa", "a"], anchor) == "a"
    assert rover_anchor_vote(["a", "xyza", "xyza"], anchor) == "xyza"
    assert rover_anchor_vote(["", "", "xxxxx"], anchor) == ""


def test_rover_insertion_tie_preserves_anchor_gap():
    assert rover_anchor_vote(["a", "xa", "ya"], 0) == "a"
