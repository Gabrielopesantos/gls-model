"""gls.config - dataclass <-> argparse <-> TOML, declared once."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import pytest

from gls.config import add_dataclass_args, resolve


@dataclass
class Cfg:
    name: str = "base"
    steps: int = 10
    lr: float = 1e-3
    flag: bool = False
    block: int | None = None
    betas: tuple[float, float] = (0.9, 0.95)
    tag: str | None = field(default=None, metadata={"help": "a label"})


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    add_dataclass_args(p, Cfg)
    return p


def test_defaults_when_no_args():
    assert resolve(Cfg, _parser(), []) == Cfg()


def test_cli_overrides_defaults():
    got = resolve(Cfg, _parser(), ["--steps", "50", "--lr", "3e-4", "--name", "x"])
    assert (got.steps, got.lr, got.name) == (50, 3e-4, "x")


def test_bool_flag_pair():
    assert resolve(Cfg, _parser(), ["--flag"]).flag is True
    assert resolve(Cfg, _parser(), ["--no-flag"]).flag is False
    assert resolve(Cfg, _parser(), []).flag is False


def test_optional_int():
    assert resolve(Cfg, _parser(), []).block is None
    assert resolve(Cfg, _parser(), ["--block", "512"]).block == 512


def test_tuple_from_cli_is_coerced():
    got = resolve(Cfg, _parser(), ["--betas", "0.8", "0.99"])
    assert got.betas == (0.8, 0.99)
    assert isinstance(got.betas, tuple)


def test_toml_then_cli_precedence(tmp_path):
    toml = tmp_path / "run.toml"
    toml.write_text('name = "fromfile"\nsteps = 99\nbetas = [0.5, 0.6]\n')
    p = _parser()
    got = resolve(Cfg, p, ["--config", str(toml), "--steps", "7"])
    assert got.name == "fromfile"  # file over default
    assert got.steps == 7  # cli over file
    assert got.betas == (0.5, 0.6)  # list -> tuple


def test_toml_rejects_unknown_key(tmp_path):
    toml = tmp_path / "bad.toml"
    toml.write_text("nope = 1\n")
    with pytest.raises(SystemExit):
        resolve(Cfg, _parser(), ["--config", str(toml)])
