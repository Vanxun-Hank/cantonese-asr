import copy
import torch

from cantonese_asr.p2_full import rank_microbatches
from cantonese_asr.p2_neural_lm import CharacterLSTM, collate_texts, vocabulary_from_train, score_texts


def test_global_token_normalization_topologies_and_tail():
    torch.manual_seed(42)
    original = torch.nn.Linear(5, 7).double()
    for count in (16, 8):
        x = torch.randn(count, 9, 5, dtype=torch.double)
        target = torch.randint(0, 7, (count, 9))
        target[::2, 3:] = -100
        total = int(target.ne(-100).sum())
        expected = copy.deepcopy(original)
        torch.nn.functional.cross_entropy(expected(x).transpose(1, 2), target).backward()
        for world, batch in ((1, 8), (2, 2), (4, 1)):
            models = [copy.deepcopy(original) for _ in range(world)]
            for micro in rank_microbatches(list(range(count)), world, batch):
                for rank, ids in enumerate(micro):
                    labels = target[ids]
                    loss = torch.nn.functional.cross_entropy(models[rank](x[ids]).transpose(1, 2), labels)
                    (loss * labels.ne(-100).sum() * world / total).backward()
            for key, parameter in expected.named_parameters():
                actual = sum(dict(model.named_parameters())[key].grad for model in models) / world
                torch.testing.assert_close(actual, parameter.grad, rtol=1e-10, atol=1e-12)


def test_neural_lm_padding_and_vocab_no_truncation():
    torch.manual_seed(42)
    vocab = vocabulary_from_train(["abc", "a"])
    inputs, targets = collate_texts(["abc", "a"], vocab)
    assert inputs.shape == targets.shape == (2, 4)
    assert int(targets.ne(-100).sum()) == 6
    model = CharacterLSTM(len(vocab), embedding=8, hidden=12, layers=2, dropout=.2)
    batched = score_texts(model, ["abc", "a"], vocab, torch.device("cpu"))
    single = [score_texts(model, [text], vocab, torch.device("cpu"))[0] for text in ("abc", "a")]
    torch.testing.assert_close(torch.tensor(batched), torch.tensor(single))
    assert "z" not in vocab


def test_neural_lm_rng_optimizer_resume():
    torch.manual_seed(43)
    vocab = vocabulary_from_train(["abc"])
    model = CharacterLSTM(len(vocab), embedding=8, hidden=12, layers=2, dropout=.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    x, y = collate_texts(["abc"], vocab)
    def step(m, o):
        m.train()
        o.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(x).transpose(1, 2), y)
        loss.backward()
        o.step()
    step(model, optimizer)
    saved = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()), torch.get_rng_state()
    step(model, optimizer)
    restored = CharacterLSTM(len(vocab), embedding=8, hidden=12, layers=2, dropout=.2)
    restored.load_state_dict(saved[0])
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.001)
    restored_optimizer.load_state_dict(saved[1])
    torch.set_rng_state(saved[2])
    step(restored, restored_optimizer)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
