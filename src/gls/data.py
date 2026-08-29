"""Training data.

Two sources, one sampling interface (``.batch(batch_size, device, val=...)``):

* **Packed shards (default).** ``prepare`` tokenizes a corpus once to a set of
  flat ``uint16`` files ``data/packed/<corpus>/<split>-00000.bin`` (+01, ...),
  each capped at ``--shard-tokens`` tokens, with a ``<split>-index.json``
  listing the completed shards. ``PackedData`` ``np.memmap``s them and presents
  one logical array for random ``block_size + 1`` window sampling. A single
  ``.bin`` file (``data/packed/<corpus>-<split>.bin``) also loads, via
  ``resolve_source``'s fallback.

* **Live stream (``--stream``).** ``StreamingTokens`` pulls
  ``load_dataset(..., streaming=True)``, tokenizes in a background thread, and
  packs into a rolling buffer. For corpora too large to land on local disk.
  Resume is *approximate* here - it fast-forwards by document count, not by
  token - so packed shards remain the choice for any run whose loss curve must
  survive a restart exactly.

vocab 32000 fits ``uint16``. Documents are joined with the ``<|endoftext|>`` id.
Exactly-once sharding across ranks is a distributed-training concern; the
``(rank, world_size)`` argument here is the seam for it and defaults to ``(0, 1)``.

    gls data prepare --corpus tinystories --split train
    gls data prepare --corpus fineweb-edu --split train --shard-tokens 100_000_000
"""

from __future__ import annotations

import itertools
import json
import queue
import sys
import threading
from bisect import bisect_right
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor

from gls import paths

CORPORA: dict[str, dict] = {
    "tinystories": {
        "hf": "roneneldan/TinyStories",
        "config": None,
        "field": "text",
        "splits": {"train": "train", "val": "validation"},
    },
    "fineweb-edu": {
        "hf": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "field": "text",
        "splits": {"train": "train"},
    },
}

_ENCODE_CHUNK = 2_000  # docs per encode_batch call
DEFAULT_SHARD_TOKENS = 100_000_000  # ~200 MiB per shard at uint16


# --------------------------------------------------------------------------- #
# the contract train() samples through                                        #
# --------------------------------------------------------------------------- #


@runtime_checkable
class TokenSource(Protocol):
    """What ``gls.train`` needs from a corpus. Two implementations:
    ``PackedData`` (memmapped shards, random ``block_size + 1`` windows) and
    ``StreamingTokens`` (HF stream tokenized on the fly, sequential windows).

    ``checkpoint_state`` is whatever the source must record to resume where it
    left off - ``{}`` for the packed source (position is the sampler RNG's, not
    the data's), ``{"_stream_docs": n}`` for the stream.
    """

    block_size: int

    @property
    def has_val(self) -> bool: ...

    def n_tokens(self, val: bool = False) -> int | None: ...

    def batch(
        self,
        batch_size: int,
        device: str | torch.device,
        val: bool = False,
        generator: torch.Generator | None = None,
        pin_memory: bool = False,
    ) -> tuple[Tensor, Tensor]: ...

    def checkpoint_state(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# paths                                                                       #
# --------------------------------------------------------------------------- #


def packed_path(corpus: str, split: str) -> Path:
    """Legacy single-file location. Still read by ``PackedData``."""
    return paths.packed_dir() / f"{corpus}-{split}.bin"


def shard_dir(corpus: str) -> Path:
    return paths.packed_dir() / corpus


def _index_path(corpus: str, split: str) -> Path:
    return shard_dir(corpus) / f"{split}-index.json"


def resolve_source(corpus: str, split: str) -> Path | None:
    """Prefer the sharded layout, fall back to the legacy single ``.bin``,
    else ``None`` (nothing prepared)."""
    if _index_path(corpus, split).exists():
        return shard_dir(corpus)
    legacy = packed_path(corpus, split)
    return legacy if legacy.exists() else None


# --------------------------------------------------------------------------- #
# tokenizer/stream helpers                                                  #
# --------------------------------------------------------------------------- #


def _load_tokenizer():
    from tokenizers import Tokenizer

    path = paths.tokenizer_artifact()
    if not path.exists():
        raise SystemExit(f"missing {path} - run `gls tokenizer train` first")
    return Tokenizer.from_file(str(path))


def _eot_id(tok) -> int:
    eot = tok.token_to_id("<|endoftext|>")
    if eot is None:
        raise SystemExit("tokenizer artifact has no <|endoftext|> token")
    return eot


def _open_stream(corpus: str, split: str):
    from datasets import load_dataset

    spec = CORPORA[corpus]
    hf_split = spec["splits"].get(split)
    if hf_split is None:
        raise SystemExit(
            f"{corpus} has no upstream {split!r} split; prepare 'train' and pass "
            f"--val-fraction to the trainer to carve a holdout"
        )
    kw = {"split": hf_split, "streaming": True}
    if spec["config"]:
        kw["name"] = spec["config"]
    return load_dataset(spec["hf"], **kw), spec["field"]


# --------------------------------------------------------------------------- #
# prepare - tokenize to packed shards                                         #
# --------------------------------------------------------------------------- #


def prepare(
    corpus: str,
    split: str,
    limit: int | None = None,
    force: bool = False,
    shard_tokens: int = DEFAULT_SHARD_TOKENS,
) -> Path:
    """Encode a corpus split to sharded uint16 arrays. Idempotent.

    Each shard is written to ``<name>.tmp`` and renamed on completion; the
    index is rewritten after every finished shard, so an interrupted run leaves
    a valid index of whatever completed and no half-file the next run mistakes
    for real data.
    """
    if corpus not in CORPORA:
        raise SystemExit(f"unknown corpus {corpus!r}; choose from {list(CORPORA)}")

    out_dir = shard_dir(corpus)
    idx_path = _index_path(corpus, split)
    if idx_path.exists() and not force:
        shards = json.loads(idx_path.read_text())["shards"]
        n = sum(s["n_tokens"] for s in shards)
        print(f"{idx_path} exists ({len(shards)} shards, {n:,} tokens); --force to rebuild")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in itertools.chain(out_dir.glob(f"{split}-*.bin"), out_dir.glob(f"{split}-*.tmp")):
        stale.unlink()

    tok = _load_tokenizer()
    eot = _eot_id(tok)

    ds, field = _open_stream(corpus, split)
    rows = iter(ds)
    shards: list[dict] = []
    seen = 0
    shard_idx = 0
    shard_buf: list[int] = []

    def _flush(buf: list[int]) -> None:
        nonlocal shard_idx
        name = f"{split}-{shard_idx:05d}.bin"
        tmp = out_dir / (name + ".tmp")
        np.asarray(buf, dtype=np.uint16).tofile(tmp)
        tmp.rename(out_dir / name)
        shards.append({"name": name, "n_tokens": len(buf)})
        idx_path.write_text(
            json.dumps({"corpus": corpus, "split": split, "shards": shards}, indent=2)
        )
        shard_idx += 1

    try:
        while limit is None or seen < limit:
            n = _ENCODE_CHUNK if limit is None else min(_ENCODE_CHUNK, limit - seen)
            chunk = list(itertools.islice(rows, n))
            if not chunk:
                break
            seen += len(chunk)
            for enc in tok.encode_batch(
                [r[field] for r in chunk if r[field]], add_special_tokens=False
            ):
                shard_buf.extend(enc.ids)
                shard_buf.append(eot)
            total = sum(s["n_tokens"] for s in shards) + len(shard_buf)
            print(f"  {corpus}-{split}: {total:,} tokens ({len(shards)} shards)", end="\r")
            if len(shard_buf) >= shard_tokens:
                _flush(shard_buf)
                shard_buf = []
    finally:
        rows.close()

    if shard_buf:
        _flush(shard_buf)

    total = sum(s["n_tokens"] for s in shards)
    print(f"\nwrote {len(shards)} shard(s) to {out_dir}  {total:,} tokens")
    return out_dir


# --------------------------------------------------------------------------- #
# packed sampling                                                             #
# --------------------------------------------------------------------------- #


def _to_device(
    x: Tensor, y: Tensor, device: str | torch.device, pin_memory: bool
) -> tuple[Tensor, Tensor]:
    """Move a sampled batch to ``device``. ``pin_memory`` enables the pinned +
    non-blocking H2D copy that overlaps with compute."""
    if pin_memory:
        x, y = x.pin_memory(), y.pin_memory()
    return x.to(device, non_blocking=pin_memory), y.to(device, non_blocking=pin_memory)


class _Shards:
    """One or more memmapped uint16 arrays presented as a single logical array.
    Reads that straddle a shard boundary are stitched (rare - a boundary is just
    another arbitrary cut in an already-concatenated token stream)."""

    def __init__(self, arrays: Sequence[np.ndarray]):
        self._arrays = arrays
        self._cum = [0]
        for a in arrays:
            self._cum.append(self._cum[-1] + len(a))

    def __len__(self) -> int:
        return self._cum[-1]

    def read(self, start: int, length: int) -> np.ndarray:
        si = bisect_right(self._cum, start) - 1
        local = start - self._cum[si]
        first = self._arrays[si][local : local + length]
        if len(first) == length:
            return first
        return np.concatenate([first, self.read(start + len(first), length - len(first))])


def _open_shards(source: Path, split: str) -> _Shards:
    if source.is_file():
        return _Shards([np.memmap(source, dtype=np.uint16, mode="r")])
    idx = json.loads((source / f"{split}-index.json").read_text())
    arrays = [np.memmap(source / s["name"], dtype=np.uint16, mode="r") for s in idx["shards"]]
    if not arrays:
        raise SystemExit(f"{source}: index lists no shards for split {split!r}")
    return _Shards(arrays)


class PackedData:
    """Memmapped uint16 token stream, sampled as random ``block_size + 1``
    windows (inputs = w[:-1], targets = w[1:]).

    ``source`` is a shard directory, a single ``.bin`` file, or ``None``.
    A validation split comes from ``val_source`` when given, else ``val_fraction``
    carves the tail of the train stream.

    ``rank``/``world_size`` partition the train stream into disjoint contiguous
    slabs, one per rank - a no-op at the default ``(0, 1)``, the hook for
    distributed sharding.
    """

    def __init__(
        self,
        source: Path,
        block_size: int,
        *,
        split: str = "train",
        val_source: Path | None = None,
        val_fraction: float = 0.0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        source = Path(source)
        if not source.exists():
            raise SystemExit(f"missing {source} - run `gls data prepare` first")
        train = _open_shards(source, split)
        train_len = len(train)

        if val_source is not None and Path(val_source).exists():
            if val_fraction > 0.0:
                print(
                    "[data] --val-fraction ignored: a prepared 'val' split is present",
                    file=sys.stderr,
                )
            self._val = _open_shards(Path(val_source), "val")
        elif val_fraction > 0.0:
            # Carve the tail. Only the (small) val slice is materialised in RAM;
            # the train portion stays a lazy memmap, capped by train_len below.
            cut = int(train_len * (1.0 - val_fraction))
            self._val = _Shards([np.ascontiguousarray(train.read(cut, train_len - cut))])
            train_len = cut
        else:
            self._val = _Shards([])

        self._train = train
        self.block_size = block_size
        self._lo, self._hi = self._rank_span(train_len, rank, world_size)

    @staticmethod
    def _rank_span(n: int, rank: int, world_size: int) -> tuple[int, int]:
        per = n // world_size
        return rank * per, (rank + 1) * per if rank < world_size - 1 else n

    @property
    def has_val(self) -> bool:
        return len(self._val) > self.block_size + 1

    def _split(self, val: bool) -> _Shards:
        return self._val if val else self._train

    def n_tokens(self, val: bool = False) -> int:
        return len(self._split(val))

    def checkpoint_state(self) -> dict[str, Any]:
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
        data = self._split(val)
        if val:
            lo, hi = 0, len(data) - self.block_size - 1
        else:
            lo, hi = self._lo, min(self._hi, len(data)) - self.block_size - 1
        if hi <= lo:
            raise ValueError(f"split slab [{lo}, {hi}] too small for block_size {self.block_size}")
        ix = torch.randint(lo, hi, (batch_size,), generator=generator).tolist()
        xs = np.stack([data.read(i, self.block_size).astype(np.int64) for i in ix])
        ys = np.stack([data.read(i + 1, self.block_size).astype(np.int64) for i in ix])
        return _to_device(torch.from_numpy(xs), torch.from_numpy(ys), device, pin_memory)


# --------------------------------------------------------------------------- #
# live streaming                                                              #
# --------------------------------------------------------------------------- #


class StreamingTokens:
    """Tokenize an HF streaming split on the fly into ``block_size + 1`` windows.

    A background thread reads + tokenizes documents into a bounded queue; the
    main thread pulls token chunks and slices windows from a rolling buffer.
    No validation split (``has_val`` is always False). ``n_tokens`` is unknown
    and returns ``None``. Resume fast-forwards by ``skip_docs`` documents, which
    lands the stream near - not exactly at - the pre-kill position.
    """

    def __init__(
        self,
        corpus: str,
        block_size: int,
        *,
        skip_docs: int = 0,
        queue_chunks: int = 64,
    ) -> None:
        self.corpus = corpus
        self.block_size = block_size
        self.docs_consumed = skip_docs
        self._buf = np.empty(0, dtype=np.int64)
        self._q: queue.Queue = queue.Queue(maxsize=queue_chunks)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, args=(skip_docs,), daemon=True)
        self._worker.start()

    @property
    def has_val(self) -> bool:
        return False

    def checkpoint_state(self) -> dict[str, Any]:
        return {"_stream_docs": self.docs_consumed}

    def _run(self, skip_docs: int) -> None:
        tok = _load_tokenizer()
        eot = _eot_id(tok)
        ds, field = _open_stream(self.corpus, "train")
        rows = itertools.islice(iter(ds), skip_docs, None)
        try:
            while not self._stop.is_set():
                chunk = list(itertools.islice(rows, _ENCODE_CHUNK))
                if not chunk:
                    break
                buf: list[int] = []
                for enc in tok.encode_batch(
                    [r[field] for r in chunk if r[field]], add_special_tokens=False
                ):
                    buf.extend(enc.ids)
                    buf.append(eot)
                self._q.put((len(chunk), np.asarray(buf, dtype=np.int64)))
        finally:
            self._q.put(None)

    def n_tokens(self, val: bool = False) -> int | None:
        return None

    def _fill(self, need: int) -> None:
        while len(self._buf) < need:
            item = self._q.get()
            if item is None:
                raise RuntimeError("stream exhausted before the requested step count")
            n_docs, arr = item
            self.docs_consumed += n_docs
            self._buf = np.concatenate([self._buf, arr])

    def batch(
        self,
        batch_size: int,
        device: str | torch.device,
        val: bool = False,
        generator: torch.Generator | None = None,
        pin_memory: bool = False,
    ) -> tuple[Tensor, Tensor]:
        span = self.block_size + 1
        self._fill(batch_size * span)
        windows = self._buf[: batch_size * span].reshape(batch_size, span)
        self._buf = self._buf[batch_size * span :]
        x = torch.from_numpy(np.ascontiguousarray(windows[:, :-1]))
        y = torch.from_numpy(np.ascontiguousarray(windows[:, 1:]))
        return _to_device(x, y, device, pin_memory)

    def close(self) -> None:
        self._stop.set()
