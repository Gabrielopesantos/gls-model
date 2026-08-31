"""``gls.sampling.sample`` - the logits-to-token policy, on hand-built logits."""

from __future__ import annotations

import torch

from gls.sampling import SamplingConfig, sample

G = lambda seed=0: torch.Generator().manual_seed(seed)  # noqa: E731


def test_temperature_zero_is_argmax():
    logits = torch.tensor([0.1, 5.0, -2.0, 3.0])
    cfg = SamplingConfig(temperature=0.0)
    assert sample(logits, cfg, generator=G(), seen=()) == 1


def test_top_k_one_collapses_to_argmax():
    logits = torch.randn(50, generator=G(7))
    cfg = SamplingConfig(temperature=1.0, top_k=1, top_p=1.0, repetition_penalty=1.0)
    picks = {sample(logits, cfg, generator=G(s), seen=()) for s in range(20)}
    assert picks == {int(logits.argmax())}


def test_top_p_keeps_only_the_nucleus():
    # one token holds ~0.99 of the mass; top_p below that must never sample the rest
    logits = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0])
    cfg = SamplingConfig(temperature=1.0, top_k=0, top_p=0.5, repetition_penalty=1.0)
    assert {sample(logits, cfg, generator=G(s), seen=()) for s in range(30)} == {0}


def test_repetition_penalty_lowers_seen_tokens():
    logits = torch.tensor([2.0, 2.0, 2.0, 2.0])
    # tokens 0 and 2 seen -> their logits halved -> greedy picks an unseen one
    cfg = SamplingConfig(temperature=0.0, repetition_penalty=2.0)
    assert sample(logits, cfg, generator=G(), seen=[0, 2]) in (1, 3)


def test_same_seed_same_draw():
    logits = torch.randn(1000, generator=G(1))
    cfg = SamplingConfig()
    a = [sample(logits, cfg, generator=G(123), seen=()) for _ in range(5)]
    b = [sample(logits, cfg, generator=G(123), seen=()) for _ in range(5)]
    assert a == b
