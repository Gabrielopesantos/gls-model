"""gls.sft - ChatML rendering, exact prompt-mask boundary, TokenSource conformance.

The `tok`-dependent tests skip without the tokenizer artifact (mirrors
test_tokenizer.py). `SFTData` is built against a monkeypatched `load_dataset`, so
no test here touches the network.
"""

from __future__ import annotations

import pytest
import torch

from gls import sft
from gls.data import TokenSource
from gls.paths import tokenizer_artifact

DOLLY = sft.SFT_CORPORA["dolly"]

_EXAMPLES = [
    {"instruction": "Name a color.", "context": "", "response": "Blue."},
    {"instruction": "Summarize.", "context": "A long passage here.", "response": "Short."},
    {"instruction": "Explain gravity briefly.", "context": "", "response": "Mass attracts mass."},
]


@pytest.fixture(scope="module")
def tok():
    if not tokenizer_artifact().exists():
        pytest.skip("tokenizer artifact not trained yet (`gls tokenizer train`)")
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(tokenizer_artifact()))


# --- constants: no artifact, no network --------------------------------------


def test_sft_corpora_well_formed():
    for spec in sft.SFT_CORPORA.values():
        assert "/" in spec["hf"]
        assert {"instruction", "context", "response"} <= set(spec)


# --- rendering/encoding ----------------------------------------------------


def test_render_is_one_shape_with_and_without_context():
    head_a, ctx_a, resp_a = sft.render(_EXAMPLES[0], DOLLY)
    head_b, ctx_b, resp_b = sft.render(_EXAMPLES[1], DOLLY)
    assert head_a.startswith("<|im_start|>user\n") and head_b.startswith("<|im_start|>user\n")
    assert ctx_a == "" and ctx_b == "A long passage here."
    assert resp_a.endswith("<|im_end|>") and resp_b.endswith("<|im_end|>")


def test_encode_mask_boundary_is_exact(tok):
    enc = sft.encode(_EXAMPLES[0], DOLLY, tok, block_size=512)
    assert enc is not None
    ids, prompt_len, truncated = enc
    assert not truncated
    # the prompt span ends exactly at the separately-encoded assistant tag
    head, _, resp = sft.render(_EXAMPLES[0], DOLLY)
    expect_prompt = (
        tok.encode(head, add_special_tokens=False).ids
        + tok.encode(sft.ASSISTANT_OPEN, add_special_tokens=False).ids
    )
    assert ids[:prompt_len] == expect_prompt
    assert ids[prompt_len:] == tok.encode(resp, add_special_tokens=False).ids


def test_encode_truncates_context_then_drops(tok):
    big = {"instruction": "Q", "context": "word " * 5000, "response": "A."}
    enc = sft.encode(big, DOLLY, tok, block_size=128)
    assert enc is not None
    ids, _, truncated = enc
    assert truncated and len(ids) <= 128

    # instruction + response alone already overflow -> dropped
    huge = {"instruction": "x " * 200, "context": "", "response": "y " * 200}
    assert sft.encode(huge, DOLLY, tok, block_size=64) is None


# --- SFTData ----------------------------------------------------------------


@pytest.fixture
def sftdata(tok, monkeypatch):
    # sft.SFTData does `from datasets import load_dataset` at call time
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: list(_EXAMPLES) * 20)
    return sft.SFTData("dolly", block_size=128, val_fraction=0.1, seed=0)


def test_sftdata_is_a_token_source(sftdata):
    assert sftdata.n_tokens() > 0
    assert isinstance(sftdata, TokenSource)
    assert sftdata.has_val is True
    assert sftdata.checkpoint_state() == {}
    sftdata.close()


def test_sftdata_batch_masks_prompt_and_pads(sftdata):
    x, y = sftdata.batch(8, "cpu", generator=torch.Generator().manual_seed(1))
    assert x.shape == y.shape and x.shape[0] == 8

    # every row has at least one supervised (response) target and at least one
    # masked (prompt) target
    for r in range(8):
        assert (y[r] != sft.IGNORE_INDEX).any()
        assert (y[r] == sft.IGNORE_INDEX).any()

    # wherever the target is live, it equals the next input token
    live = y != sft.IGNORE_INDEX
    assert torch.equal(y[live], x.roll(-1, dims=1)[live])

    # padding: past a row's real length, inputs are <|pad|> and targets are masked
    pad_id = sftdata.pad_id
    for r in range(8):
        pad_positions = x[r] == pad_id
        if pad_positions.any():
            assert (y[r][pad_positions] == sft.IGNORE_INDEX).all()


def test_sftdata_sampling_is_generator_deterministic(sftdata):
    a = sftdata.batch(4, "cpu", generator=torch.Generator().manual_seed(3))[0]
    b = sftdata.batch(4, "cpu", generator=torch.Generator().manual_seed(3))[0]
    assert torch.equal(a, b)


def test_sftdata_holdout_is_carved_by_example(sftdata):
    # disjoint train/val index sets, neither empty
    assert set(sftdata._train_idx.tolist()).isdisjoint(sftdata._val_idx.tolist())
    assert len(sftdata._val_idx) > 0
    total = len(sftdata._train_idx) + len(sftdata._val_idx)
    assert total == len(sftdata._ex)


def test_sweep_covers_the_split_exactly_once(sftdata):
    """The measurement path must be a census, not a sample. `batch` draws with
    replacement, so a perplexity built on it depends on iters/batch_size/seed."""
    seen = 0
    for x, y in sftdata.sweep(2, "cpu", val=True):
        assert x.shape == y.shape
        seen += x.shape[0]
    assert seen == len(sftdata._val_idx)


def test_sweep_is_batch_size_invariant(sftdata):
    """Same supervised tokens regardless of batching - the property that makes
    two checkpoints comparable when they were scored at different batch sizes."""

    def supervised(bs):
        out = []
        for x, y in sftdata.sweep(bs, "cpu", val=True):
            live = y != sft.IGNORE_INDEX
            out.append(torch.stack([y[live], x.roll(-1, dims=1)[live]]))
        return torch.cat(out, dim=1)

    a, b = supervised(1), supervised(3)
    assert a.shape == b.shape
    assert torch.equal(a.sort(dim=1).values, b.sort(dim=1).values)


def test_sweep_is_deterministic_across_calls(sftdata):
    first = [x.clone() for x, _ in sftdata.sweep(2, "cpu", val=True)]
    second = [x.clone() for x, _ in sftdata.sweep(2, "cpu", val=True)]
    assert len(first) == len(second)
    assert all(torch.equal(p, q) for p, q in zip(first, second, strict=True))
