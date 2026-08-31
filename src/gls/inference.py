"""Single-request generation over a straightforward KV cache.

``Session`` holds three things together: an immutable model, the per-layer keys
and values it has already computed, and every token of the conversation so far.
The model could be shared by any number of sessions; the cache is mutable and
belongs to exactly one, because it is the conversation, encoded.

Token ids in, token ids out. ``gls.chat`` wraps this with ChatML framing
and a streaming detokenizer. Imports only ``gls.model`` and ``gls.sampling``:
the inference path stays free of the optimizer and the training loop
(enforced by ``tests/test_package.py``).

Context policy: when the conversation fills ``max_seq_len`` the session stops and
raises ``ContextFull`` rather than evicting. Sliding-window eviction belongs with
a paged KV allocator, which is where the eviction policy gets designed for real.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, nullcontext

import torch
from torch import Tensor

from gls.model import GLSModel
from gls.sampling import SamplingConfig, sample

Kv = tuple[Tensor, Tensor]


class ContextFull(Exception):
    """The conversation has reached ``max_seq_len`` and cannot grow further.
    ``length`` is the token count at the point generation stopped."""

    def __init__(self, length: int) -> None:
        super().__init__(f"context full ({length} tokens); reset to continue")
        self.length = length


class Session:
    """One conversation's generation state."""

    def __init__(
        self,
        model: GLSModel,
        device: str | torch.device = "cpu",
        *,
        autocast: AbstractContextManager | None = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.autocast: AbstractContextManager = autocast or nullcontext()
        self.history: list[int] = []
        self._past: list[Kv] | None = None
        self._cached = 0  # tokens from the start of `history` the cache covers

    @property
    def max_seq_len(self) -> int:
        return self.model.cfg.max_seq_len

    def __len__(self) -> int:
        return len(self.history)

    def reset(self) -> None:
        """Forget the conversation and the cache with it."""
        self.history.clear()
        self._past = None
        self._cached = 0

    def push(self, ids: list[int]) -> None:
        """Add tokens to the conversation without running the model; the next
        ``generate`` prefills them."""
        self.history.extend(ids)

    def discard_last(self) -> int:
        """Drop the most recently generated token from the conversation.

        ``generate`` appends what it samples, because the next step has to run the
        model over it. A caller that stops on a token - a turn terminator, say -
        does not want it in the history: it would be context for the next turn and
        would count against the repetition penalty, making the model less willing
        to stop again.

        Valid only for a token the cache has not yet absorbed, which is exactly the
        one ``generate`` just yielded.
        """
        if len(self.history) <= self._cached:
            raise RuntimeError("the last token is already in the cache and cannot be discarded")
        return self.history.pop()

    def generate(
        self,
        cfg: SamplingConfig,
        generator: torch.Generator,
        *,
        max_new_tokens: int,
        stop_ids: frozenset[int] = frozenset(),
    ) -> Iterator[int]:
        """Prefill everything pushed since the last step, then yield one token at
        a time. Stops on a member of ``stop_ids`` or after ``max_new_tokens``;
        raises ``ContextFull`` if the conversation reaches ``max_seq_len`` first.
        """
        if self._cached >= len(self.history):
            raise RuntimeError("nothing to generate from; push a prompt first")

        pending = self.history[self._cached :]
        if self._cached + len(pending) > self.max_seq_len:
            raise ContextFull(self._cached + len(pending))
        logits = self._forward(pending)

        for _ in range(max_new_tokens):
            token = sample(logits, cfg, generator=generator, seen=self.history)
            self.history.append(token)
            yield token
            if token in stop_ids:
                return
            if self._cached >= self.max_seq_len:
                raise ContextFull(self._cached)
            logits = self._forward([token])

    def _forward(self, ids: list[int]) -> Tensor:
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad(), self.autocast:
            logits, self._past = self.model.forward_cached(x, self._past)
        self._cached += len(ids)
        return logits[0]
