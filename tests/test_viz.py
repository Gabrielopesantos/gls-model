"""gls.viz - the `gls model summary` table.

`torchinfo` comes from the `viz` group (`uv sync --group viz`); the tests that
need it skip when it is absent, and one checks the install-hint path with it
faked out.
"""

from __future__ import annotations

import sys

import pytest

from gls import viz
from gls.model import build_model


def test_summary_prints_activation_footer_and_deduped_param_count(capsys):
    pytest.importorskip("torchinfo")
    viz.summary("small", batch_size=2, block_size=32)
    out = capsys.readouterr()
    assert "Estimated Total Size" in out.out  # the number the command exists for
    # the note carries num_parameters(), which dedupes the tied lm_head - not
    # torchinfo's "Total params", which counts it twice
    assert f"{build_model('small').num_parameters():,}" in out.err


def test_summary_rejects_unknown_tier():
    pytest.importorskip("torchinfo")
    with pytest.raises(SystemExit, match="unknown tier"):
        viz.summary("gigantic")


def test_summary_without_torchinfo_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "torchinfo", None)
    with pytest.raises(SystemExit, match="torchinfo not installed"):
        viz.summary("small")


def test_cli_model_summary_dispatches(monkeypatch):
    from gls import cli

    seen = {}
    monkeypatch.setattr(viz, "summary", lambda tier, **kw: seen.update(tier=tier, **kw))
    cli.main(["model", "summary", "--tier", "small", "--batch-size", "4"])
    assert seen == {"tier": "small", "batch_size": 4, "block_size": None}
