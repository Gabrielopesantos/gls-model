# From-scratch decoder-only transformer: training and (later) inference infra.

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
