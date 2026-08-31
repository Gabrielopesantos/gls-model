"""Architecture inspection: a per-layer table (``torchinfo``) and an optional
module diagram (``torchview``), both driven straight off ``gls.model.PRESETS``.

Neither library is a runtime dependency - they live in the ``viz`` group
(``uv sync --group viz``), and each entry point fails with an install hint when
its backend is missing, the same way ``gls.evaluate`` treats ``lm-eval`` and
``transformers``.

This is the per-layer shape and parameter view. The instance-sizing number is
the measured ``train/peak_mem_gib`` the loop logs, not torchinfo's "Estimated
Total Size" footer, which overestimates the training residency (it sums every
module output tensor and models neither autocast nor SDPA nor compile fusion).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from gls.model import PRESETS, ModelConfig, build_model, kv_bytes_per_token


def _note(msg: str) -> None:
    print(f"[viz] {msg}", file=sys.stderr)


def _resolve(tier: str) -> ModelConfig:
    if tier not in PRESETS:
        raise SystemExit(f"unknown tier {tier!r}; choose from {list(PRESETS)}")
    return PRESETS[tier]


def summary(tier: str, *, batch_size: int = 8, block_size: int | None = None) -> None:
    """Print ``torchinfo.summary`` for a preset at a concrete ``(batch, seq)``."""
    try:
        from torchinfo import summary as ti_summary  # pyright: ignore[reportMissingImports]
    except ImportError:
        raise SystemExit("torchinfo not installed - run `uv sync --group viz`") from None

    cfg = _resolve(tier)
    seq = block_size or cfg.max_seq_len
    model = build_model(tier).eval()
    ids = torch.zeros(batch_size, seq, dtype=torch.long)  # real token ids, not floats

    stats = ti_summary(
        model,
        input_data=ids,
        col_names=("input_size", "output_size", "num_params", "params_percent"),
        depth=3,
        verbose=0,
    )
    print(stats)
    # torchinfo's "Total params" counts the tied lm_head separately from the
    # embedding - num_parameters() dedupes, and is the number the tier tables use.
    _note(
        f"tier={tier}  batch={batch_size}  block_size={seq}  "
        f"params={model.num_parameters():,} (tied; torchinfo's total double-counts the "
        f"embedding)  KV={kv_bytes_per_token(cfg) / 1024:.1f} KiB/token"
    )


def graph(
    tier: str, out: str | Path, *, batch_size: int = 1, block_size: int | None = None
) -> None:
    """Render a module diagram to ``out`` (extension picks the format). Needs the
    ``graphviz`` binary on PATH as well as the ``torchview`` package."""
    try:
        from torchview import draw_graph  # pyright: ignore[reportMissingImports]
    except ImportError:
        raise SystemExit("torchview not installed - run `uv sync --group viz`") from None

    cfg = _resolve(tier)
    seq = block_size or cfg.max_seq_len
    ids = torch.zeros(batch_size, seq, dtype=torch.long)
    out = Path(out)

    drawing = draw_graph(
        build_model(tier).eval(),
        input_data=ids,
        graph_name=out.stem,
        expand_nested=True,
        depth=3,
    )
    try:
        drawing.visual_graph.render(
            out.with_suffix(""), format=out.suffix.lstrip(".") or "svg", cleanup=True
        )
    except Exception as exc:  # graphviz raises ExecutableNotFound from a deep import
        if "dot" in str(exc) or "ExecutableNotFound" in type(exc).__name__:
            raise SystemExit(
                "graphviz binary not found - `dot` must be on PATH (devenv.nix adds it)"
            ) from None
        raise
    _note(f"wrote {out}")
