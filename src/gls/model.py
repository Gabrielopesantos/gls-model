"""Decoder-only transformer.

Grouped-query attention, rotary position encoding in the ``rotate_half``
(GPT-NeoX) layout, RMSNorm, SwiGLU, tied input/output embeddings, no bias
terms. Together these keep every preset expressible as a ``transformers``
``LlamaConfig``, so a greedy-decode equivalence gate gets a reference at
matching shape. ``llama_config_dict``/``to_llama_state_dict`` are that
converter, and they are a pure rename table by construction.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gls.tokenizer import SPECIAL_TOKENS, VOCAB_SIZE

N_SPECIAL = len(SPECIAL_TOKENS)


# --------------------------------------------------------------------------- #
# config                                                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelConfig:
    """The architecture dimensions. Travels verbatim in the checkpoint dir so a
    model directory is loadable without external context."""

    vocab: int = VOCAB_SIZE
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 2
    head_dim: int = 64
    d_ff: int = 1024
    max_seq_len: int = 512
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    tie_embeddings: bool = True
    # Presets are all biasless (matches LlamaConfig defaults, keeps the
    # Llama converter a rename). The flag exists because Qwen2/2.5 puts a bias
    # on q/k/v, and Qwen2.5-0.5B is a secondary-check candidate whose weights
    # would not otherwise load into this module.
    attention_bias: bool = False

    def __post_init__(self) -> None:
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {self.head_dim}")
        if self.n_heads * self.head_dim != self.d_model:
            raise ValueError(
                f"n_heads*head_dim ({self.n_heads}*{self.head_dim}) != d_model ({self.d_model}); "
                "the Llama-expressible invariant needs them equal"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be a multiple of n_kv_heads ({self.n_kv_heads})"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        """Load a config.json back. An unrecognised key is an error, not a
        silent drop - a typo must not quietly load a wrong architecture."""
        known = set(cls.__dataclass_fields__)
        out: dict[str, Any] = {}
        unknown: list[str] = []
        for key, value in d.items():
            mapped = _ALIASES.get(key, key)
            if mapped in known:
                out[mapped] = value
            else:
                unknown.append(key)
        if unknown:
            raise ValueError(
                f"unknown ModelConfig keys {sorted(unknown)} - checkpoint from a newer gls, "
                "or a typo"
            )
        return cls(**out)


# Old field name -> current, for checkpoints written before a rename. Empty so
# far; the single place backward-compatibility logic is allowed to live.
_ALIASES: dict[str, str] = {}


# This is the source of truth for tier dimensions - gls.tokenizer's break-even print
# reads d_model/KV shape back from here rather than duplicating them.
PRESETS: dict[str, ModelConfig] = {
    "small": ModelConfig(
        d_model=384,
        n_layers=6,
        n_heads=6,
        n_kv_heads=2,
        head_dim=64,
        d_ff=1024,
        max_seq_len=512,
    ),
    "medium": ModelConfig(
        d_model=1024,
        n_layers=24,
        n_heads=16,
        n_kv_heads=4,
        head_dim=64,
        d_ff=2816,
        max_seq_len=2048,
    ),
    "large": ModelConfig(
        d_model=2048,
        n_layers=24,
        n_heads=16,
        n_kv_heads=4,
        head_dim=128,
        d_ff=5504,
        max_seq_len=2048,
    ),
}


def kv_bytes_per_token(cfg: ModelConfig, dtype_bytes: int = 2) -> int:
    """KV-cache bytes for one token at one position: K and V, every layer, every
    KV head. The unit a paged KV allocator is budgeted in."""
    return 2 * cfg.n_layers * cfg.n_kv_heads * cfg.head_dim * dtype_bytes


# --------------------------------------------------------------------------- #
# building blocks                                                             #
# --------------------------------------------------------------------------- #


class RMSNorm(nn.Module):
    """Weight only, no bias, no mean-centring. Computed in fp32 then cast back,
    matching transformers' LlamaRMSNorm exactly."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def build_rope_cache(head_dim: int, max_seq_len: int, theta: float) -> tuple[Tensor, Tensor]:
    """cos/sin tables for the ``rotate_half`` layout: inv_freq over even indices,
    each frequency duplicated so ``emb = cat([freqs, freqs])``. Same construction
    as transformers' LlamaRotaryEmbedding, which is what makes the converter a
    rename. Returned as buffers by the model, never parameters."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """q, k: (B, n_heads, T, head_dim). cos, sin: (T, head_dim) already sliced to
    the positions in play."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot.to(q.dtype), k_rot.to(k.dtype)


class Attention(nn.Module):
    """Grouped-query attention. q/o stay full width; k/v shrink to n_kv_heads -
    this is what cuts the KV cache 4x. It sets the KV-cache budget and is a hard
    architecture requirement, not a tuning knob."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads

        q_out = cfg.n_heads * cfg.head_dim
        kv_out = cfg.n_kv_heads * cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, q_out, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(cfg.d_model, kv_out, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(cfg.d_model, kv_out, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(q_out, cfg.d_model, bias=False)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        past_kv: tuple[Tensor, Tensor] | None = None,
        return_kv: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE positions run from the cached prefix length onward. With no cache
        # this is arange(T); the branch is here so incremental decode lands in
        # the signature, not as a later rewrite.
        offset = 0 if past_kv is None else past_kv[0].shape[2]
        q, k = apply_rope(q, k, cos[offset : offset + T], sin[offset : offset + T])

        if past_kv is not None:
            k = torch.cat((past_kv[0], k), dim=2)
            v = torch.cat((past_kv[1], v), dim=2)
        new_kv = (k, v) if return_kv else None

        # GQA: replicate each KV head n_rep times to face all query heads. Kept
        # explicit - the incremental-decode path reads it closely.
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        # Three shapes reach this line. Prefill from an empty cache: q and k span
        # the same positions, is_causal handles it. A single decode step (T == 1)
        # against a warm cache attends the whole cache and needs no mask. A
        # multi-token prefill onto a warm cache - turn 2 of a chat - is the odd
        # one out: the new tokens may see the entire cached prefix but only their
        # own causal prefix among themselves, which is neither is_causal nor
        # no mask. Build it explicitly (True == attend); `offset` is the cached
        # length, computed above for RoPE.
        if past_kv is not None and T > 1:
            attend_prefix = torch.ones(T, offset, dtype=torch.bool, device=x.device)
            causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            attn_mask = torch.cat((attend_prefix, causal), dim=1)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=past_kv is None)
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out), new_kv


class MLP(nn.Module):
    """SwiGLU: down(silu(gate(x)) * up(x)). No biases."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    """Pre-norm decoder block: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        past_kv: tuple[Tensor, Tensor] | None = None,
        return_kv: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        attn_out, new_kv = self.self_attn(
            self.input_layernorm(x), cos, sin, past_kv=past_kv, return_kv=return_kv
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_kv


# --------------------------------------------------------------------------- #
# model                                                                       #
# --------------------------------------------------------------------------- #


class GLSModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embed_tokens = nn.Embedding(cfg.vocab, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        cos, sin = build_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        # persistent=False: a non-parameter buffer, absent from state_dict and
        # from the parameter count. A learned position table reintroduced by
        # accident would fail the per-preset count test.
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Residual-branch output projections: shrink the init by 1/sqrt(2*n_layers)
        # so the residual stream variance does not grow with depth (GPT-2 trick).
        scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 * scale)
        # Special-token embedding rows: the 12 reserved slots and <|pad|> are
        # never emitted by data.prepare, so they get no gradient; with tied
        # embeddings a random row would still emit spurious logits from the
        # output head. Zero them.
        # (<|endoftext|>, row 0, is trained - it joins every document pair.)
        with torch.no_grad():
            self.embed_tokens.weight[:N_SPECIAL].zero_()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        T = input_ids.shape[1]
        if T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}")

        cos = self.rope_cos.to(dtype=torch.float32)
        sin = self.rope_sin.to(dtype=torch.float32)

        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h, _ = layer(h, cos, sin)
        h = self.norm(h)
        logits = self.lm_head(h)

        loss = None
        if targets is not None:
            # cross_entropy is an autocast-to-fp32 op, so bf16 logits are
            # promoted inside it - no need for an explicit .float() copy, which
            # at vocab 32000 is a multi-GiB tensor held for the whole backward.
            # ignore_index=-100 is cross_entropy's default; named here because SFT
            # leans on it - prompt and padding positions carry -100 so loss lands
            # on the response span only. Inert for pretraining (the packed stream
            # never emits -100).
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    def forward_cached(
        self,
        input_ids: Tensor,
        past_kvs: list[tuple[Tensor, Tensor]] | None = None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """Incremental forward for generation: run ``input_ids`` on top of the
        per-layer keys and values in ``past_kvs`` and return ``(logits, new_kvs)``
        where ``logits`` is ``(B, vocab)`` for the final position only and
        ``new_kvs`` is the extended cache to pass to the next call.

        This is the decode path; ``forward`` stays the training/scoring one. Not
        ``torch.compile``d - the cache grows a position per step, so a compiled
        version would retrace every call. A preallocated cache is what makes it
        compilable.
        """
        B, T = input_ids.shape
        offset = 0 if past_kvs is None else past_kvs[0][0].shape[2]
        total = offset + T
        if total > self.cfg.max_seq_len:
            raise ValueError(f"context length {total} exceeds max_seq_len {self.cfg.max_seq_len}")

        cos = self.rope_cos.to(dtype=torch.float32)
        sin = self.rope_sin.to(dtype=torch.float32)

        h = self.embed_tokens(input_ids)
        new_kvs: list[tuple[Tensor, Tensor]] = []
        for i, layer in enumerate(self.layers):
            past = None if past_kvs is None else past_kvs[i]
            h, kv = layer(h, cos, sin, past_kv=past, return_kv=True)
            assert kv is not None  # return_kv=True
            new_kvs.append(kv)

        # Only the last position feeds sampling; normalise and project just that
        # row rather than the whole T.
        h = self.norm(h[:, -1:, :])
        logits = self.lm_head(h)
        return logits[:, -1, :], new_kvs

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Tied weights are counted once (nn.Module.parameters dedupes)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)


def build_model(tier: str) -> GLSModel:
    if tier not in PRESETS:
        raise KeyError(f"unknown tier {tier!r}; choose from {list(PRESETS)}")
    return GLSModel(PRESETS[tier])


# --------------------------------------------------------------------------- #
# Llama-expressible converter - a pure rename table                           #
# --------------------------------------------------------------------------- #


def llama_config_dict(cfg: ModelConfig) -> dict[str, Any]:
    """The kwargs for a transformers LlamaConfig equivalent to this ModelConfig.
    Field-for-field; no derived geometry, because head_dim*n_heads == d_model in
    every preset."""
    return {
        "vocab_size": cfg.vocab,
        "hidden_size": cfg.d_model,
        "intermediate_size": cfg.d_ff,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "max_position_embeddings": cfg.max_seq_len,
        "rope_theta": cfg.rope_theta,
        "rms_norm_eps": cfg.rms_norm_eps,
        "tie_word_embeddings": cfg.tie_embeddings,
        "hidden_act": "silu",
        "attention_bias": cfg.attention_bias,
        "mlp_bias": False,
    }


def to_llama_state_dict(model: GLSModel) -> dict[str, Tensor]:
    """Our state_dict -> the keys LlamaForCausalLM expects. The whole transform
    is a prefix: our transformer stack lives under HF's ``model.`` submodule,
    and ``lm_head`` sits at the top level (tied to the embedding). If this ever
    needs arithmetic, the architecture has drifted off the invariant."""
    out: dict[str, Tensor] = {}
    for name, tensor in model.state_dict().items():
        if name.startswith("lm_head."):
            out[name] = tensor  # top-level in LlamaForCausalLM, not under model.
        else:
            out[f"model.{name}"] = tensor
    out.setdefault("lm_head.weight", model.lm_head.weight)
    return out
