"""Architecture checks (verification.md -> "Model architecture").

These run in seconds on CPU. The Llama round-trip skips when `transformers`
(the `eval` dependency group) is absent, mirroring test_tokenizer.py's
skip-when-the-artifact-is-missing pattern.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from gls.model import (
    N_SPECIAL,
    PRESETS,
    GLSModel,
    ModelConfig,
    build_rope_cache,
    llama_config_dict,
    to_llama_state_dict,
)

# Parameter counts from privatedocs/plan/model-sizes.md#tiers, recomputed by the
# formula in that file's #parameter-count-formula section.
DOC_PARAM_COUNTS = {
    "small": 21_730_176,
    "medium": 303_350_784,
    "large": 1_128_892_416,
}
DOC_APPROX = {"small": "21.7M", "medium": "303M", "large": "1.13B"}


def formula_params(cfg: ModelConfig) -> int:
    """The model-sizes.md formula, independent of the nn.Module implementation."""
    head_total = cfg.n_heads * cfg.head_dim
    kv_total = cfg.n_kv_heads * cfg.head_dim
    embedding = cfg.vocab * cfg.d_model  # tied, counted once
    per_layer = (
        cfg.d_model * head_total  # q_proj
        + cfg.d_model * kv_total  # k_proj
        + cfg.d_model * kv_total  # v_proj
        + head_total * cfg.d_model  # o_proj
        + 3 * cfg.d_model * cfg.d_ff  # SwiGLU gate/up/down
        + 2 * cfg.d_model  # input + post-attention norms
    )
    if cfg.attention_bias:
        per_layer += head_total + 2 * kv_total
    return embedding + cfg.n_layers * per_layer + cfg.d_model  # + final norm


# --- parameter count: the guard against a learned position table -------------


@pytest.mark.parametrize("tier", list(PRESETS))
def test_formula_matches_doc_table(tier):
    assert formula_params(PRESETS[tier]) == DOC_PARAM_COUNTS[tier]


@pytest.mark.parametrize("tier", list(PRESETS))
def test_doc_approx_label_matches(tier):
    # model-sizes.md writes these as "~21.7M/~303M/~1.13B" - approximate, so
    # allow 1% either way rather than an exact rounding.
    n = DOC_PARAM_COUNTS[tier]
    label = DOC_APPROX[tier]
    approx = float(label[:-1]) * (1e6 if label.endswith("M") else 1e9)
    assert abs(n - approx) / approx < 0.01


@pytest.mark.parametrize("tier", list(PRESETS))
def test_built_model_matches_formula(tier):
    # `large` is ~4.5 GiB in fp32 - build it on the meta device (shapes, no
    # storage) so the structural count is still checked without the memory.
    device = "meta" if tier == "large" else "cpu"
    with torch.device(device):
        model = GLSModel(PRESETS[tier])
    assert model.num_parameters() == formula_params(PRESETS[tier])


# --- RoPE contributes zero parameters ---------------------------------------


def test_rope_tables_are_buffers_not_params_or_state():
    model = GLSModel(PRESETS["small"])
    names = dict(model.named_parameters())
    assert "rope_cos" not in names and "rope_sin" not in names
    # persistent=False -> absent from the checkpoint payload too.
    assert "rope_cos" not in model.state_dict()
    assert "rope_sin" not in model.state_dict()
    assert isinstance(model.rope_cos, torch.Tensor)
    # no parameter anywhere is position-indexed
    assert not any("rope" in n or "pos_emb" in n for n in names)


def test_rope_cache_matches_rotate_half_construction():
    cos, sin = build_rope_cache(head_dim=8, max_seq_len=4, theta=10_000.0)
    assert cos.shape == (4, 8) and sin.shape == (4, 8)
    # rotate_half layout: first half of the freqs equals the second half.
    assert torch.allclose(cos[:, :4], cos[:, 4:])
    assert torch.allclose(sin[:, :4], sin[:, 4:])
    assert torch.allclose(cos[0], torch.ones(8))  # position 0: angle 0


# --- no biases, tied head --------------------------------------------------


@pytest.mark.parametrize("tier", list(PRESETS))
def test_no_biases_in_default_presets(tier):
    device = "meta" if tier == "large" else "cpu"
    with torch.device(device):
        model = GLSModel(PRESETS[tier])
    assert not any(n.endswith(".bias") for n, _ in model.named_parameters())


def test_lm_head_tied_to_embedding():
    model = GLSModel(PRESETS["small"])
    assert model.lm_head.weight is model.embed_tokens.weight


def test_from_dict_roundtrips_and_rejects_unknown_keys():
    cfg = PRESETS["small"]
    assert ModelConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError, match="unknown ModelConfig keys"):
        ModelConfig.from_dict({**cfg.to_dict(), "n_experts": 8})


def test_attention_bias_flag_adds_qkv_bias():
    cfg = replace(PRESETS["small"], attention_bias=True)
    model = GLSModel(cfg)
    biases = {n for n, _ in model.named_parameters() if n.endswith(".bias")}
    assert any("q_proj" in n for n in biases)
    assert any("k_proj" in n for n in biases)
    assert any("v_proj" in n for n in biases)
    assert not any("o_proj" in n for n in biases)  # o_proj stays biasless
    assert model.num_parameters() == formula_params(cfg)


# --- special-token rows zeroed at init ------------------------------------


def test_special_token_embedding_rows_zeroed():
    model = GLSModel(PRESETS["small"])
    assert torch.all(model.embed_tokens.weight[:N_SPECIAL] == 0.0)
    # a normal row is not zero
    assert model.embed_tokens.weight[N_SPECIAL].abs().sum() > 0


# --- forward shape + causality ------------------------------------------


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(
        vocab=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        max_seq_len=32,
    )


def test_forward_shapes_and_loss():
    cfg = _tiny_cfg()
    model = GLSModel(cfg)
    ids = torch.randint(0, cfg.vocab, (3, 16))
    logits, loss = model(ids, ids)
    assert logits.shape == (3, 16, cfg.vocab)
    assert loss.ndim == 0 and torch.isfinite(loss)
    logits_only, none_loss = model(ids)
    assert none_loss is None


def test_forward_rejects_overlong_sequence():
    cfg = _tiny_cfg()
    model = GLSModel(cfg)
    with pytest.raises(ValueError):
        model(torch.zeros(1, cfg.max_seq_len + 1, dtype=torch.long))


def test_attention_is_causal():
    cfg = _tiny_cfg()
    model = GLSModel(cfg).eval()
    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab, (1, 16))
    with torch.no_grad():
        base = model(ids)[0]
        t = 9
        perturbed = ids.clone()
        perturbed[0, t] = (perturbed[0, t] + 1) % cfg.vocab
        after = model(perturbed)[0]
    # positions before t cannot see the change
    assert torch.allclose(base[:, :t], after[:, :t], atol=1e-5)
    # position t itself does change
    assert not torch.allclose(base[:, t], after[:, t], atol=1e-5)


# --- Llama-expressible: the converter is a rename -----------------------


def test_llama_roundtrip_is_a_rename():
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM  # pyright: ignore[reportMissingImports]

    cfg = replace(PRESETS["small"], vocab=512, n_layers=2, max_seq_len=64)
    model = GLSModel(cfg).eval()

    lcfg = LlamaConfig(**llama_config_dict(cfg))
    lmodel = LlamaForCausalLM(lcfg).eval()
    missing, unexpected = lmodel.load_state_dict(to_llama_state_dict(model), strict=False)
    assert not unexpected
    assert set(missing) <= {"lm_head.weight"}  # tied, filled from embed_tokens

    ids = torch.randint(0, cfg.vocab, (2, 24))
    with torch.no_grad():
        ours = model(ids)[0]
        theirs = lmodel(ids).logits
    # fp32, same weights, pure rename -> they match. Formal tolerance belongs to
    # the inference gate; this is the early warning that the rotate_half RoPE
    # layout is right.
    assert torch.allclose(ours, theirs, atol=1e-4, rtol=1e-4)


# --- the model can actually learn -------------------------------------


def test_training_reduces_loss(tmp_path):
    import numpy as np

    from gls.data import PackedData

    cfg = _tiny_cfg()
    # period-32 repeating stream: learnable in a few dozen steps.
    rng = np.random.default_rng(0)
    period = np.tile(rng.integers(0, cfg.vocab, size=32, dtype=np.uint16), 5_000)
    path = tmp_path / "toy.bin"
    period.tofile(path)
    data = PackedData(path, block_size=16)

    torch.manual_seed(0)
    model = GLSModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)

    x, y = data.batch(16, "cpu")
    with torch.no_grad():
        start_loss = model(x, y)[1].item()

    loss = None
    for _ in range(80):
        x, y = data.batch(16, "cpu")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    assert loss is not None and loss.item() < start_loss - 1.0


# --- checkpoint round-trip -------------------------------------------


def test_checkpoint_roundtrip(tmp_path):
    from gls.checkpoint import load_model_dir, save_model_dir

    cfg = _tiny_cfg()
    model = GLSModel(cfg)
    save_model_dir(model, tmp_path / "ckpt", step=42)
    restored = load_model_dir(tmp_path / "ckpt")

    assert restored.cfg == cfg
    for (n, a), (_, b) in zip(model.named_parameters(), restored.named_parameters(), strict=True):
        assert torch.equal(a, b), n
    ids = torch.randint(0, cfg.vocab, (1, 12))
    with torch.no_grad():
        assert torch.allclose(model(ids)[0], restored(ids)[0])
