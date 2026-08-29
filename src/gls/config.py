"""Dataclass <-> argparse <-> TOML glue.

A run config is a plain dataclass (see ``TrainConfig`` in ``gls.train``). This
module turns that one definition into a CLI and a file format so a field is
declared exactly once:

    parser = argparse.ArgumentParser()
    add_dataclass_args(parser, TrainConfig)
    cfg = resolve(TrainConfig, parser, argv)

Precedence, low to high: dataclass defaults < ``--config file.toml`` < explicit
CLI flags. Only flags the user actually passed override the file, so
``--config foo.toml --lr 1e-4`` does the obvious thing.

``ModelConfig`` deliberately does *not* go through here - its fields are an
architecture contract carried in the checkpoint, not run knobs, and the three
``PRESETS`` are the only sanctioned combinations.
"""

from __future__ import annotations

import argparse
import dataclasses
import tomllib
import types
import typing
from pathlib import Path
from typing import Any, get_args, get_origin

_SENTINEL = argparse.SUPPRESS


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
    """``X | None`` -> ``(X, True)``; anything else -> ``(tp, False)``."""
    if get_origin(tp) in (typing.Union, types.UnionType):
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return tp, False


def _field_flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def add_dataclass_args(parser: argparse.ArgumentParser, cls: type) -> None:
    """Add one CLI flag per dataclass field.

    ``int``/``float``/``str`` map straight through; ``bool`` becomes a
    ``--flag``/``--no-flag`` pair; ``X | None`` becomes a nullable flag;
    ``tuple[T, T]`` becomes ``nargs=2``. Every flag defaults to a sentinel so
    ``resolve`` can tell "passed" from "left alone". Help text comes from
    ``field.metadata["help"]``.
    """
    hints = typing.get_type_hints(cls)
    for f in dataclasses.fields(cls):
        tp, _optional = _unwrap_optional(hints[f.name])
        flag = _field_flag(f.name)
        help_txt = f.metadata.get("help", "")
        origin = get_origin(tp)

        if tp is bool:
            parser.add_argument(
                flag, dest=f.name, action="store_true", default=_SENTINEL, help=help_txt
            )
            parser.add_argument(
                _field_flag("no_" + f.name),
                dest=f.name,
                action="store_false",
                default=_SENTINEL,
                help=argparse.SUPPRESS,
            )
        elif origin in (tuple, list):
            (elem_tp,) = set(get_args(tp)) or (float,)
            parser.add_argument(
                flag,
                dest=f.name,
                type=elem_tp,
                nargs=len(get_args(tp)) if origin is tuple else "+",
                default=_SENTINEL,
                help=help_txt,
            )
        else:
            parser.add_argument(flag, dest=f.name, type=tp, default=_SENTINEL, help=help_txt)


def _coerce(cls: type, name: str, value: Any) -> Any:
    """TOML gives us lists where the dataclass wants tuples; nothing else needs
    massaging (tomllib already produces int/float/str/bool)."""
    hints = typing.get_type_hints(cls)
    tp, _ = _unwrap_optional(hints[name])
    if get_origin(tp) is tuple and isinstance(value, list):
        return tuple(value)
    return value


def load_toml(path: str | Path, cls: type) -> dict[str, Any]:
    data = tomllib.loads(Path(path).read_text())
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise SystemExit(f"{path}: unknown keys {sorted(unknown)}; valid: {sorted(known)}")
    return {k: _coerce(cls, k, v) for k, v in data.items()}


def resolve(cls: type, parser: argparse.ArgumentParser, argv: list[str] | None = None) -> Any:
    """Merge dataclass defaults, an optional ``--config`` TOML, and explicit CLI
    flags into an instance of ``cls``.

    ``parser`` must already have been through ``add_dataclass_args`` and must
    define a ``--config`` argument.
    """
    args = parser.parse_args(argv)
    values: dict[str, Any] = {}

    config_path = getattr(args, "config", None)
    if config_path:
        values.update(load_toml(config_path, cls))

    for f in dataclasses.fields(cls):
        passed = getattr(args, f.name, _SENTINEL)
        if passed is not _SENTINEL:
            values[f.name] = passed

    return cls(**{k: _coerce(cls, k, v) for k, v in values.items()})
