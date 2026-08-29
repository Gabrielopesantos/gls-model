"""From-scratch decoder-only transformer: training and (later) inference infra.

Flat layout, one concept per file - the module list is the architecture diagram:

    model       GLSModel, ModelConfig, PRESETS - architecture + forward, no optimizer
    trainer     Trainer, Runtime            - the optimizer, the step, the eval pass
    train       TrainConfig, train, lr_at   - the hand-written loop
    data        PackedData, StreamingTokens, TokenSource
    tokenizer   byte-level BPE + the bake-off
    checkpoint  save/load, rotation, resume
    tracking    JSONL metrics + optional W&B mirror
    config      dataclass <-> argparse <-> TOML
    paths       one root, every on-disk location
    env         torch/CUDA/device facts
    cli         the `gls` command (argparse lives here, nowhere else)

The names below are re-exported lazily (PEP 562): ``from gls import Trainer``
works, but ``import gls.model`` still does not drag in the training stack, so
inference code stays optimizer-free.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.1.0"

_EXPORTS = {
    "GLSModel": "gls.model",
    "ModelConfig": "gls.model",
    "PRESETS": "gls.model",
    "build_model": "gls.model",
    "TrainConfig": "gls.train",
    "train": "gls.train",
    "Trainer": "gls.trainer",
    "Runtime": "gls.trainer",
    "PackedData": "gls.data",
    "StreamingTokens": "gls.data",
    "TokenSource": "gls.data",
}

__all__ = [
    "GLSModel",
    "ModelConfig",
    "PRESETS",
    "Runtime",
    "PackedData",
    "StreamingTokens",
    "TokenSource",
    "TrainConfig",
    "Trainer",
    "__version__",
    "build_model",
    "train",
]


def __getattr__(name: str) -> object:
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from gls.data import PackedData, StreamingTokens, TokenSource
    from gls.model import PRESETS, GLSModel, ModelConfig, build_model
    from gls.train import TrainConfig, train
    from gls.trainer import Runtime, Trainer
