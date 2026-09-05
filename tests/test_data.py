"""gls.data - sharded packs, legacy single-file, deterministic sampling."""

from __future__ import annotations

import json

import numpy as np
import torch

import gls.data as gd
from gls.data import PackedData, StreamingTokens, TokenSource, _Shards


def _write_shards(tmp_path, split, sizes):
    d = tmp_path / "corpusX"
    d.mkdir()
    shards = []
    tok = 0
    for i, n in enumerate(sizes):
        name = f"{split}-{i:05d}.bin"
        np.arange(tok, tok + n, dtype=np.uint16).tofile(d / name)
        shards.append({"name": name, "n_tokens": n})
        tok += n
    (d / f"{split}-index.json").write_text(json.dumps({"split": split, "shards": shards}))
    return d


def test_shards_present_one_logical_array(tmp_path):
    s = _Shards([np.arange(0, 10, dtype=np.uint16), np.arange(10, 25, dtype=np.uint16)])
    assert len(s) == 25
    got = s.read(8, 6)  # straddles the boundary at 10
    assert got.tolist() == [8, 9, 10, 11, 12, 13]


def test_packeddata_reads_across_shard_boundary(tmp_path):
    d = _write_shards(tmp_path, "train", [64, 64, 32])
    data = PackedData(d, block_size=8)
    assert data.n_tokens() == 160
    x, y = data.batch(4, "cpu", generator=torch.Generator().manual_seed(0))
    assert x.shape == (4, 8) and y.shape == (4, 8)
    # targets are inputs shifted by one everywhere (contiguous token ids)
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_legacy_single_bin_still_loads(tmp_path):
    p = tmp_path / "tinystories-train.bin"
    np.arange(0, 500, dtype=np.uint16).tofile(p)
    data = PackedData(p, block_size=16)
    assert data.n_tokens() == 500
    x, _ = data.batch(2, "cpu", generator=torch.Generator().manual_seed(1))
    assert x.shape == (2, 16)


def test_sampling_is_generator_deterministic(tmp_path):
    d = _write_shards(tmp_path, "train", [200])
    data = PackedData(d, block_size=8)
    a = data.batch(4, "cpu", generator=torch.Generator().manual_seed(7))[0]
    b = data.batch(4, "cpu", generator=torch.Generator().manual_seed(7))[0]
    assert torch.equal(a, b)


def test_val_fraction_carves_tail(tmp_path):
    d = _write_shards(tmp_path, "train", [1000])
    data = PackedData(d, block_size=8, val_fraction=0.1)
    assert data.has_val
    assert data.n_tokens(val=True) == 100
    # train slab must not reach into the val tail
    assert data._hi <= 900


# --- prepare(): the sharded writer, exercised end to end -----------------------


class _FakeEnc:
    def __init__(self, ids):
        self.ids = ids


class _FakeTok:
    """5 ids per doc; prepare() appends the eot itself."""

    def encode_batch(self, texts, add_special_tokens=False):
        return [_FakeEnc([7, 7, 7, 7, 7]) for _ in texts]


def _patch_prepare(monkeypatch, tmp_path, n_docs):
    monkeypatch.setattr(gd.paths, "packed_dir", lambda: tmp_path)
    monkeypatch.setattr(gd, "_load_tokenizer", lambda: _FakeTok())
    monkeypatch.setattr(gd, "_eot_id", lambda tok: 0)
    # small encode chunk so the shard-flush threshold is crossed several times
    monkeypatch.setattr(gd, "_ENCODE_CHUNK", 5)
    docs = [{"text": f"doc {i}"} for i in range(n_docs)]

    class _FakeDS:  # iter(ds) must yield a generator - prepare() calls .close() on it
        def __iter__(self):
            yield from docs

    monkeypatch.setattr(gd, "_open_stream", lambda corpus, split: (_FakeDS(), "text"))


def test_prepare_writes_multiple_shards(monkeypatch, tmp_path):
    _patch_prepare(monkeypatch, tmp_path, n_docs=30)
    # 30 docs x 6 tokens = 180; a 50-token shard cap forces 4 shards.
    out = gd.prepare("fineweb-edu", "train", shard_tokens=50)

    idx = json.loads((out / "train-index.json").read_text())
    names = [s["name"] for s in idx["shards"]]
    assert len(names) >= 3
    assert names == sorted(names)  # train-00000.bin, train-00001.bin, ...
    assert sum(s["n_tokens"] for s in idx["shards"]) == 30 * 6

    assert not list(out.glob("*.tmp"))
    for s in idx["shards"]:
        assert (out / s["name"]).exists()

    # the pack round-trips through the reader as one logical stream
    data = PackedData(out, block_size=8)
    assert data.n_tokens() == 180
    x, y = data.batch(4, "cpu", generator=torch.Generator().manual_seed(0))
    assert x.shape == (4, 8) and torch.equal(y[:, :-1], x[:, 1:])


def test_prepare_force_clears_stale_shards(monkeypatch, tmp_path):
    _patch_prepare(monkeypatch, tmp_path, n_docs=30)
    gd.prepare("fineweb-edu", "train", shard_tokens=50)
    out = tmp_path / "fineweb-edu"
    n_first = len(json.loads((out / "train-index.json").read_text())["shards"])
    assert n_first >= 3

    _patch_prepare(monkeypatch, tmp_path, n_docs=6)
    gd.prepare("fineweb-edu", "train", shard_tokens=50, force=True)
    idx = json.loads((out / "train-index.json").read_text())
    assert sum(s["n_tokens"] for s in idx["shards"]) == 6 * 6
    # no shard from the larger first run left behind
    bins = sorted(p.name for p in out.glob("train-*.bin"))
    assert bins == [s["name"] for s in idx["shards"]]


# --- both sources honour the TokenSource contract train() samples through -----


def test_packeddata_is_a_token_source(tmp_path):
    d = _write_shards(tmp_path, "train", [200])
    data = PackedData(d, block_size=8)
    assert isinstance(data, TokenSource)
    assert data.has_val is False
    assert data.n_tokens() == 200
    assert data.checkpoint_state() == {}  # position lives in the sampler RNG
    data.close()  # no-op, must exist


def test_streamingtokens_conforms_without_starting_the_worker():
    # __init__ spins a network thread; bypass it and check the contract members
    # the refactor changed: has_val is a property (False), n_tokens is unknown
    # (None), checkpoint_state carries the doc cursor for approximate resume.
    st = object.__new__(StreamingTokens)
    st.docs_consumed = 7
    st.block_size = 16
    assert isinstance(st, TokenSource)
    assert st.has_val is False
    assert st.n_tokens() is None
    assert st.checkpoint_state() == {"_stream_docs": 7}
