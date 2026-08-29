"""gls.data - sharded packs, legacy single-file, deterministic sampling."""

from __future__ import annotations

import json

import numpy as np
import torch

from gls.data import PackedData, _Shards


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
