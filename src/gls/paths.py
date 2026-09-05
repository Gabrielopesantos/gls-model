"""Filesystem layout: every on-disk location the package reads or writes, derived
from one root.

The root is ``$GLS_ROOT`` when set, else the repository that contains this
source file.
"""

from __future__ import annotations

import os
from pathlib import Path

GLS_ROOT_ENV = "GLS_ROOT"


def repo_root() -> Path:
    override = os.environ.get(GLS_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def artifacts_dir() -> Path:
    return repo_root() / "artifacts"


def tokenizer_artifact() -> Path:
    return artifacts_dir() / "tokenizer" / "tokenizer.json"


def data_dir() -> Path:
    return repo_root() / "data"


def packed_dir() -> Path:
    return data_dir() / "packed"


def runs_dir() -> Path:
    return repo_root() / "runs"
