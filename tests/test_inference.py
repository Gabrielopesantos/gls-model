"""``gls.inference.Session`` - the KV cache and the decode loop, on random weights.

The gate that matters is ``test_cached_logits_match_full_forward``: an incremental
decode must produce the same logits as one pass over the whole sequence, for all
three shapes the cache path sees.
"""

from __future__ import annotations

import pytest
import torch

from gls.inference import ContextFull, Session
from gls.model import GLSModel, ModelConfig
from gls.sampling import SamplingConfig

GREEDY = SamplingConfig(temperature=0.0)


def _model() -> GLSModel:
    cfg = ModelConfig(
        vocab=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        max_seq_len=32,
    )
    torch.manual_seed(0)
    return GLSModel(cfg).eval()


def _run(session: Session, *, n: int, stop_ids: frozenset[int] = frozenset()) -> list[int]:
    return list(session.generate(GREEDY, torch.Generator(), max_new_tokens=n, stop_ids=stop_ids))


@pytest.mark.parametrize("splits", [(12,), (1, 1, 1, 1), (5, 4, 1, 1, 1), (8, 4)])
def test_cached_logits_match_full_forward(splits):
    """Feeding the sequence in `splits` chunks through the cache must match a
    single uncached forward at every step - covers prefill-from-empty (first
    chunk), single-token decode (len-1 chunks), and a multi-token chunk onto a
    warm cache (the explicit causal mask)."""
    model = _model()
    ids = torch.randint(0, model.cfg.vocab, (1, sum(splits)))
    with torch.no_grad():
        ref = model(ids)[0][0]  # (T, vocab)

    session = Session(model)
    pos = 0
    for n in splits:
        session.push(ids[0, pos : pos + n].tolist())
        logits = session._forward(session.history[session._cached :])
        pos += n
        assert torch.allclose(logits, ref[pos - 1], atol=1e-4)


def _greedy_first(model: GLSModel, prompt: list[int]) -> int:
    """The token greedy decoding emits first for `prompt` - deterministic."""
    s = Session(model)
    s.push(list(prompt))
    return _run(s, n=1)[0]


def test_generate_stops_on_stop_id():
    model = _model()
    stop = _greedy_first(model, [1, 2, 3])
    session = Session(model)
    session.push([1, 2, 3])
    out = _run(session, n=20, stop_ids=frozenset({stop}))
    assert out == [stop]
    assert session.history[-1] == stop  # present until discard_last


def test_discard_last_removes_uncached_token():
    model = _model()
    session = Session(model)
    session.push([1, 2, 3])
    (tok,) = _run(session, n=1, stop_ids=frozenset({_greedy_first(model, [1, 2, 3])}))
    assert session.history == [1, 2, 3, tok]
    assert session.discard_last() == tok
    assert session.history == [1, 2, 3]
    # the cache now covers the whole conversation, so the caller pushes the next
    # turn before generating again.
    session.push([4])
    assert _run(session, n=1)


def test_context_full_raises_at_max_seq_len():
    model = _model()
    session = Session(model)
    session.push(list(range(1, 6)))
    with pytest.raises(ContextFull):
        _run(session, n=1000)
    assert len(session) >= model.cfg.max_seq_len


def test_reset_clears_history_and_cache():
    model = _model()
    session = Session(model)
    session.push([1, 2, 3])
    session._forward(session.history)
    session.reset()
    assert len(session) == 0 and session._cached == 0 and session._past is None
