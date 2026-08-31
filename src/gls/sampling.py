"""Logits in, one token id out.

A pure function over a single logit vector plus a small config. It knows nothing
about the model, the KV cache, or text - ``gls.inference`` owns those. Kept
separate so the sampling policy can be unit-tested on hand-built logits and swapped
without touching the decode loop.

The pipeline is repetition penalty, then temperature, then top-k, then top-p
nucleus, then a multinomial draw. ``temperature == 0`` short-circuits to greedy
argmax.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SamplingConfig:
    """Decoding knobs. The defaults are conventional sampling values; ``gls chat``
    exposes each one as a flag."""

    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.95
    repetition_penalty: float = 1.1


def sample(
    logits: Tensor,
    cfg: SamplingConfig,
    *,
    generator: torch.Generator,
    seen: Sequence[int] = (),
) -> int:
    """Draw the next token id from a 1-D ``logits`` vector.

    ``seen`` is every token already in the conversation - prompt and generated
    alike - so the repetition penalty covers context the model can no longer
    attend to as well. ``generator`` carries all the randomness; global RNG is
    never touched, so a fixed seed gives a reproducible sequence.
    """
    if logits.ndim != 1:
        raise ValueError(f"expected a 1-D logit vector, got shape {tuple(logits.shape)}")

    # Sample on the CPU regardless of where the model ran: the vector is one row
    # of vocab floats, the copy is nothing next to a forward pass, and it keeps a
    # seeded run reproducible without a device-matched Generator.
    logits = logits.detach().float().cpu()

    if cfg.repetition_penalty != 1.0 and len(seen):
        idx = torch.tensor(sorted(set(seen)), dtype=torch.long, device=logits.device)
        idx = idx[idx < logits.numel()]
        vals = logits[idx]
        # CTRL's asymmetric form: a positive logit is divided, a negative one
        # multiplied, so the penalty always moves the value toward zero.
        penalised = torch.where(
            vals > 0, vals / cfg.repetition_penalty, vals * cfg.repetition_penalty
        )
        logits = logits.index_copy(0, idx, penalised)

    if cfg.temperature <= 0.0:
        return int(torch.argmax(logits))

    logits = logits / cfg.temperature

    if cfg.top_k > 0:
        k = min(cfg.top_k, logits.numel())
        kth = torch.topk(logits, k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    probs = torch.softmax(logits, dim=-1)

    if 0.0 < cfg.top_p < 1.0:
        ordered, order = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(ordered, dim=-1)
        # Keep the smallest prefix whose mass reaches top_p (the token that tips
        # it over is included, so at least one always survives).
        keep = cumulative < cfg.top_p
        keep[0] = True
        probs = torch.zeros_like(probs).index_copy(0, order[keep], ordered[keep])
        probs = probs / probs.sum()

    return int(torch.multinomial(probs, 1, generator=generator))
