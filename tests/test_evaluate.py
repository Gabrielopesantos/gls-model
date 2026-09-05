"""gls.evaluate - the GLSModel lm-eval adapter (teacher-forced scoring).

The adapter's tokenizer is faked, so these run without the artifact; the
`lm_eval` package (dev sync pulls it via the `eval` group) is imported lazily
inside `_make_adapter` and `harness`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from gls import checkpoint, evaluate, sft
from gls.model import PRESETS, GLSModel

pytest.importorskip("lm_eval")


class _FakeEnc:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _FakeTok:
    """Deterministic char -> id, vocab 64."""

    def encode(self, text: str, add_special_tokens: bool = False) -> _FakeEnc:
        return _FakeEnc([ord(c) % 64 for c in text])

    def token_to_id(self, _t: str) -> int:
        return 0


def _tiny_ckpt(tmp_path):
    cfg = replace(
        PRESETS["small"],
        vocab=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        head_dim=8,
        n_kv_heads=2,
        d_ff=64,
        max_seq_len=16,
    )
    torch.manual_seed(0)
    model = GLSModel(cfg)
    d = tmp_path / "ckpt"
    checkpoint.save_model_dir(model, d, step=0)
    return d, model, cfg


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    monkeypatch.setattr("gls.data.load_tokenizer", lambda: _FakeTok())
    d, model, cfg = _tiny_ckpt(tmp_path)
    return evaluate._make_adapter(d, "cpu", batch_size=1), model, cfg


def _req(*args):
    return SimpleNamespace(arguments=tuple(args))


def test_loglikelihood_matches_direct_gather(adapter):
    lm, model, _ = adapter
    tok = _FakeTok()
    ctx_ids = tok.encode("the cat").ids
    cont_ids = tok.encode(" sat").ids

    ((logp, is_greedy),) = lm.loglikelihood([_req("the cat", " sat")])

    full = ctx_ids + cont_ids
    with torch.no_grad():
        logits, _ = model(torch.tensor([full[:-1]]))
    ref_rows = torch.log_softmax(logits[0].float(), dim=-1)[len(ctx_ids) - 1 :]
    tgt = torch.tensor(cont_ids)
    ref_logp = ref_rows.gather(-1, tgt[:, None]).squeeze(-1).sum().item()

    assert logp == pytest.approx(ref_logp, rel=1e-5)
    assert is_greedy == bool((ref_rows.argmax(-1) == tgt).all())


def test_loglikelihood_truncates_overlong_context(adapter):
    lm, _, cfg = adapter
    long_ctx = "x" * (cfg.max_seq_len * 3)
    ((logp, _),) = lm.loglikelihood([_req(long_ctx, " y")])
    assert logp == pytest.approx(logp) and logp < 0  # finite, a real log-prob


def test_rolling_is_refused(adapter):
    lm, _, _ = adapter
    with pytest.raises(NotImplementedError, match="rolling"):
        lm.loglikelihood_rolling([_req("hello world")])


def test_generate_until_is_refused(adapter):
    lm, _, _ = adapter
    with pytest.raises(NotImplementedError, match="decode loop"):
        lm.generate_until([_req("q", {})])


def test_harness_without_lm_eval_raises_install_hint(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "lm_eval", None)
    with pytest.raises(SystemExit, match="lm-eval not installed"):
        evaluate.harness(tmp_path / "nope", ("piqa",))


# --- perplexity rebuilds training's holdout ---------------------------------


class _FakeSFTData:
    """Records the args perplexity constructs it with; yields one trivial batch."""

    seen: dict = {}

    def __init__(self, corpus, block_size, *, val_fraction, seed):
        _FakeSFTData.seen = {
            "corpus": corpus,
            "block_size": block_size,
            "val_fraction": val_fraction,
            "seed": seed,
        }

    def batch(self, batch_size, device, val=False, generator=None, pin_memory=False):
        x = torch.zeros((batch_size, 8), dtype=torch.long)
        y = torch.full((batch_size, 8), sft.IGNORE_INDEX)
        y[:, 3] = 1  # one supervised target so total_tok > 0
        return x, y


def _run_with_meta(tmp_path, meta: dict):
    _, model, _ = _tiny_ckpt(tmp_path)  # writes tmp_path/ckpt/{model.safetensors,config.json}
    step_dir = tmp_path / "run" / "checkpoints" / "step-000010"
    checkpoint.save_model_dir(model, step_dir, step=10)
    (tmp_path / "run" / "train_config.json").write_text(json.dumps(meta))
    return step_dir


def test_perplexity_takes_seed_and_block_size_from_run_meta(tmp_path, monkeypatch):
    monkeypatch.setattr("gls.data.load_tokenizer", lambda: _FakeTok())
    monkeypatch.setattr(sft, "SFTData", _FakeSFTData)
    step_dir = _run_with_meta(tmp_path, {"seed": 99, "val_fraction": 0.2, "block_size": 7})

    evaluate.perplexity(step_dir, "dolly", iters=1, batch_size=2, device="cpu")

    assert _FakeSFTData.seen == {
        "corpus": "dolly",
        "block_size": 7,
        "val_fraction": 0.2,
        "seed": 99,
    }


def test_perplexity_block_size_falls_back_to_max_seq_len(tmp_path, monkeypatch):
    monkeypatch.setattr("gls.data.load_tokenizer", lambda: _FakeTok())
    monkeypatch.setattr(sft, "SFTData", _FakeSFTData)
    # train_config.json carries block_size=null (the TrainConfig default), so the
    # checkpoint's own max_seq_len is what training resolved to.
    step_dir = _run_with_meta(tmp_path, {"seed": 1, "block_size": None})

    evaluate.perplexity(step_dir, "dolly", iters=1, batch_size=2, device="cpu")

    assert _FakeSFTData.seen["block_size"] == 16  # _tiny_ckpt's max_seq_len


def test_normalise_device_maps_gpu_to_cuda():
    assert evaluate._normalise_device("gpu") == "cuda"
    assert evaluate._normalise_device("cpu") == "cpu"
