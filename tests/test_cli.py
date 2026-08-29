"""gls.cli - the one argparse surface. Preconditions fail before any real work."""

from __future__ import annotations

import pytest

from gls import cli


def test_bare_invocation_lists_subcommands(capsys):
    with pytest.raises(SystemExit):
        cli.main([])
    assert "{train,data,tokenizer,eval,env}" in capsys.readouterr().err


def test_train_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        cli.main(["train", "--help"])
    assert exc.value.code == 0


def test_unknown_tier_rejected_before_training(monkeypatch):
    called = False

    def _fail(_cfg):
        nonlocal called
        called = True

    monkeypatch.setattr(cli, "train", _fail)
    with pytest.raises(SystemExit) as exc:
        cli.main(["train", "--tier", "enormous"])
    assert "unknown tier" in str(exc.value)
    assert not called  # rejected at the boundary, model never built


def test_config_then_cli_flag_precedence(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "train", lambda cfg: seen.setdefault("cfg", cfg))

    toml = tmp_path / "run.toml"
    toml.write_text("lr = 0.5\nsteps = 99\n")
    cli.main(["train", "--config", str(toml), "--lr", "0.001"])

    assert seen["cfg"].lr == 0.001  # CLI over file
    assert seen["cfg"].steps == 99  # file over default


def test_init_from_with_resume_is_not_an_error(tmp_path, monkeypatch):
    # resume-after-kill re-runs the same config, which carries init_from; the
    # run's own checkpoint wins in train() and the seed is ignored - no conflict.
    seen = {}
    monkeypatch.setattr(cli, "train", lambda cfg: seen.setdefault("cfg", cfg))
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"")  # resolve_init only checks existence
    cli.main(["train", "--init-from", str(ckpt), "--resume", "auto"])
    assert seen["cfg"].init_from == str(ckpt) and seen["cfg"].resume == "auto"


def test_init_from_nonexistent_path_rejected_before_training(monkeypatch):
    monkeypatch.setattr(cli, "train", lambda cfg: pytest.fail("train ran with a bad --init-from"))
    with pytest.raises(SystemExit, match="no checkpoint"):
        cli.main(["train", "--init-from", "/no/such/checkpoint"])
