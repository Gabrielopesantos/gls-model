"""Instruction fine-tuning data.

One corpus for now - ``databricks/databricks-dolly-15k`` - rendered to ChatML
with the tokenizer's reserved ``<|im_start|>``/``<|im_end|>`` pair and
supervised on the response span only: prompt and padding positions carry the
``-100`` ignore label, so the reported loss is "how well does it answer", not
"how well does it echo the question".

``SFTData`` conforms to ``gls.data.TokenSource``, so ``gls.train`` samples it
exactly like a packed pretraining corpus - ``_make_data`` routes ``--corpus
dolly`` here. The examples fit in memory (15k x ~300 tokens), so there is no
prepare step and no shard layout.

Note the ``<|im_start|>``/``<|im_end|>`` embedding rows are zeroed at model
init (``data.prepare`` never emits them, so they would otherwise get no
gradient); SFT is where they first learn. Starting from exact zero with tied
embeddings is fine, worth knowing when reading the first few hundred steps.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import torch
from torch import Tensor

from gls import data

IGNORE_INDEX = -100
DEFAULT_VAL_FRACTION = 0.05

# HF dataset id + the field names each corpus uses. Fallback if Dolly's category
# mix proves too narrow: HuggingFaceH4/no_robots (prompt/prompt_id/messages).
SFT_CORPORA: dict[str, dict] = {
    "dolly": {
        "hf": "databricks/databricks-dolly-15k",
        "split": "train",
        "instruction": "instruction",
        "context": "context",
        "response": "response",
    },
}

# ChatML framing on the reserved <|im_start|>/<|im_end|> pair.
USER_OPEN = "<|im_start|>user\n"
ASSISTANT_OPEN = "<|im_end|>\n<|im_start|>assistant\n"
TURN_CLOSE = "<|im_end|>"


def _note(msg: str) -> None:
    print(f"[sft] {msg}", file=sys.stderr)


def render(ex: dict, spec: dict) -> tuple[str, str, str]:
    """``(head, context, response)`` text pieces.

    ``head`` is everything up to and including the assistant tag without the
    optional context; ``context`` is the retrieval passage (may be empty). Kept
    as pieces rather than one string so the length policy can trim the context
    at the token level without disturbing the ChatML framing.
    """
    instr = ex[spec["instruction"]].strip()
    ctx = (ex.get(spec["context"]) or "").strip()
    resp = ex[spec["response"]].strip()
    head = f"{USER_OPEN}{instr}"
    return head, ctx, f"{resp}{TURN_CLOSE}"


def encode(ex: dict, spec: dict, tok, block_size: int) -> tuple[list[int], int, bool] | None:
    """``(ids, prompt_len, context_was_truncated)`` or ``None`` if the example
    cannot fit ``block_size`` even after dropping the context entirely.

    Prompt and response are tokenized as separate spans so ``prompt_len`` is
    exact - the supervised region is ``ids[prompt_len:]``, never inferred by a
    string search.
    """
    head, ctx, resp = render(ex, spec)
    head_ids = tok.encode(head, add_special_tokens=False).ids
    tail_ids = tok.encode(ASSISTANT_OPEN, add_special_tokens=False).ids
    resp_ids = tok.encode(resp, add_special_tokens=False).ids

    fixed = len(head_ids) + len(tail_ids) + len(resp_ids)
    if fixed > block_size:
        return None

    truncated = False
    ctx_ids: list[int] = []
    if ctx:
        sep_ids = tok.encode("\n\n", add_special_tokens=False).ids
        budget = block_size - fixed - len(sep_ids)
        if budget > 0:
            full = tok.encode(ctx, add_special_tokens=False).ids
            ctx_ids = sep_ids + full[:budget]
            truncated = len(full) > budget
        else:
            truncated = True  # no room for the passage at all

    prompt_ids = head_ids + ctx_ids + tail_ids
    return prompt_ids + resp_ids, len(prompt_ids), truncated


class SFTData:
    """In-memory instruction corpus, sampled like ``data.PackedData``.

    Each ``batch`` right-pads its rows to the longest with ``<|pad|>`` in the
    inputs and ``-100`` in the targets, and masks the prompt span so loss lands
    on response tokens only. The holdout is a seeded permutation carved by
    example, not by token offset (which would split an example in half).
    """

    block_size: int

    def __init__(
        self,
        corpus: str,
        block_size: int,
        *,
        val_fraction: float = DEFAULT_VAL_FRACTION,
        seed: int = 1337,
    ) -> None:
        if corpus not in SFT_CORPORA:
            raise SystemExit(f"unknown SFT corpus {corpus!r}; choose from {list(SFT_CORPORA)}")
        spec = SFT_CORPORA[corpus]
        self.block_size = block_size

        tok = data.load_tokenizer()
        pad = tok.token_to_id("<|pad|>")
        if pad is None:
            raise SystemExit("tokenizer artifact has no <|pad|> token")
        self.pad_id = pad

        from datasets import load_dataset

        ds = load_dataset(spec["hf"], split=spec["split"])
        examples: list[tuple[np.ndarray, int]] = []
        n_trunc = n_drop = 0
        for ex in ds:
            enc = encode(ex, spec, tok, block_size)
            if enc is None:
                n_drop += 1
                continue
            ids, prompt_len, truncated = enc
            n_trunc += truncated
            examples.append((np.asarray(ids, dtype=np.int64), prompt_len))

        if not examples:
            raise SystemExit(f"{corpus}: every example exceeds block_size {block_size}")

        perm = np.random.default_rng(seed).permutation(len(examples))
        n_val = int(len(examples) * val_fraction)
        self._ex = examples
        self._val_idx = perm[:n_val]
        self._train_idx = perm[n_val:]
        _note(
            f"{corpus}: {len(examples)} examples "
            f"({n_trunc} context-truncated, {n_drop} dropped as too long for "
            f"block_size {block_size}); {len(self._train_idx)} train/{n_val} val"
        )

    @property
    def has_val(self) -> bool:
        return len(self._val_idx) > 0

    def _pool(self, val: bool) -> np.ndarray:
        return self._val_idx if val else self._train_idx

    def n_tokens(self, val: bool = False) -> int:
        return int(sum(len(self._ex[i][0]) for i in self._pool(val)))

    def checkpoint_state(self) -> dict[str, Any]:
        # Deterministic from (seed, val_fraction) at construction plus the
        # sampler RNG the loop already checkpoints - nothing to add.
        return {}

    def close(self) -> None:
        pass

    def batch(
        self,
        batch_size: int,
        device: str | torch.device,
        val: bool = False,
        generator: torch.Generator | None = None,
        pin_memory: bool = False,
    ) -> tuple[Tensor, Tensor]:
        pool = self._pool(val)
        if len(pool) == 0:
            raise ValueError(f"no examples in the {'val' if val else 'train'} split")
        sel = torch.randint(0, len(pool), (batch_size,), generator=generator).tolist()
        rows = [self._ex[pool[i]] for i in sel]
        width = max(len(ids) for ids, _ in rows)

        x = np.full((batch_size, width), self.pad_id, dtype=np.int64)
        y = np.full((batch_size, width), IGNORE_INDEX, dtype=np.int64)
        for r, (ids, prompt_len) in enumerate(rows):
            n = len(ids)
            x[r, :n] = ids
            # position t predicts ids[t+1]; supervise only where the target is a
            # response token, i.e. t+1 >= prompt_len.
            y[r, prompt_len - 1 : n - 1] = ids[prompt_len:n]

        return data._to_device(torch.from_numpy(x), torch.from_numpy(y), device, pin_memory)
