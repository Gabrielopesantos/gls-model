"""gls.train - loop-level behaviour: best-checkpoint capture, early stopping,
SIGTERM checkpointing, and the post-checkpoint sync hook.

Every test drives the real ``train()`` against a tiny preset and a toy packed
``.bin`` under a ``GLS_ROOT`` tmp dir, so nothing here touches the network or the
real ``runs/`` tree. ``Trainer.eval_step`` is scripted where a specific val curve
is the thing under test.
"""

from __future__ import annotations

import json
import os
import signal
from dataclasses import replace

import numpy as np
import pytest

from gls import train as T
from gls.checkpoint import latest_dir
from gls.model import PRESETS
from gls.train import TrainConfig


def _tiny():
    return replace(
        PRESETS["small"],
        vocab=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        head_dim=16,
        n_kv_heads=2,
        d_ff=128,
        max_seq_len=32,
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    """GLS_ROOT tmp dir, a tiny `small` preset, and a toy tinystories train pack."""
    monkeypatch.setenv("GLS_ROOT", str(tmp_path))
    monkeypatch.setitem(T.PRESETS, "small", _tiny())
    packed = tmp_path / "data" / "packed"
    packed.mkdir(parents=True)
    rng = np.random.default_rng(0)
    np.tile(rng.integers(0, 256, size=32, dtype=np.uint16), 4_000).tofile(
        packed / "tinystories-train.bin"
    )
    return tmp_path


def _cfg(**over) -> TrainConfig:
    base = TrainConfig(
        tier="small",
        corpus="tinystories",
        device="cpu",
        steps=5,
        batch_size=4,
        grad_accum=1,
        block_size=16,
        warmup_steps=1,
        eval_interval=1,
        eval_iters=2,
        ckpt_interval=100,
        keep_last=3,
        val_fraction=0.2,
    )
    return replace(base, **over)


def _events(run_dir) -> list[dict]:
    return [json.loads(x) for x in (run_dir / "log.jsonl").read_text().splitlines()]


def _script_val(monkeypatch, values):
    it = iter(values)
    monkeypatch.setattr(T.Trainer, "eval_step", lambda self, data: {"train": 1.0, "val": next(it)})


def test_best_checkpoint_captured_between_ckpt_marks(env, monkeypatch):
    # eval every step, but ckpt_interval only fires on the last step. The best val
    # is the second eval; without the save-on-improvement path it would never
    # reach best.json.
    _script_val(monkeypatch, [5.0, 3.0, 4.0, 4.0, 4.0])
    T.train(_cfg(run_name="r", steps=5, ckpt_interval=5, keep_last=2))

    best = json.loads((env / "runs" / "r" / "checkpoints" / "best.json").read_text())
    assert best["step"] == 2
    assert best["val_loss"] == pytest.approx(3.0)


def test_patience_stops_before_max_steps(env, monkeypatch):
    _script_val(monkeypatch, [3.0] * 20)  # first eval sets best; none improve after
    T.train(_cfg(run_name="e", steps=50, patience=2, min_improvement=0.01))

    events = _events(env / "runs" / "e")
    stop = [e for e in events if e.get("event") == "early_stop"]
    assert stop and stop[0]["step"] < 50
    assert not any(e.get("event") == "done" for e in events)


def test_patience_zero_runs_to_completion(env, monkeypatch):
    _script_val(monkeypatch, [3.0] * 20)
    T.train(_cfg(run_name="z", steps=6, patience=0))

    events = _events(env / "runs" / "z")
    done = [e for e in events if e.get("event") == "done"]
    assert done and done[0]["step"] == 6


def test_sigterm_checkpoints_then_exits_clean(env, monkeypatch):
    real = T.Trainer.train_step
    n = {"i": 0}

    def step_then_term(self, data, gen):
        n["i"] += 1
        out = real(self, data, gen)
        if n["i"] == 3:
            os.kill(os.getpid(), signal.SIGTERM)
        return out

    monkeypatch.setattr(T.Trainer, "train_step", step_then_term)
    T.train(_cfg(run_name="s", steps=100, eval_interval=2, ckpt_interval=100))

    events = _events(env / "runs" / "s")
    assert any(e.get("event") == "sigterm" for e in events)
    assert not any(e.get("event") == "done" for e in events)
    assert latest_dir(env / "runs" / "s") is not None
    # handler is restored, so a later SIGTERM is the default action again
    assert signal.getsignal(signal.SIGTERM) in (signal.SIG_DFL, signal.default_int_handler)


def test_sync_cmd_runs_after_each_checkpoint(env):
    dest = env / "synced"
    dest.mkdir()
    T.train(
        _cfg(
            run_name="y",
            steps=3,
            eval_interval=1,
            ckpt_interval=3,
            sync_cmd=f"cp -r {{ckpt}} {dest}/",
        )
    )
    assert list(dest.glob("*/model.safetensors"))


def test_sync_cmd_failure_does_not_abort_run(env):
    T.train(_cfg(run_name="x", steps=3, ckpt_interval=3, sync_cmd="false"))
    assert (env / "runs" / "x" / "checkpoints" / "latest.json").exists()


def test_sync_cmd_run_tree_carries_pointers_and_final_log(env):
    # A {run}-tree sync_cmd: the final sync fires after run.finish(), so the
    # copied log.jsonl must include the terminating `done` row and the pointer
    # files must be present.
    dest = env / "synced"
    dest.mkdir()
    T.train(
        _cfg(
            run_name="y",
            steps=3,
            eval_interval=1,
            ckpt_interval=3,
            sync_cmd=f"cp -r {{run}}/. {dest}/",
        )
    )
    assert (dest / "checkpoints" / "latest.json").exists()
    assert (dest / "train_config.json").exists()
    synced_events = [json.loads(x) for x in (dest / "log.jsonl").read_text().splitlines()]
    assert synced_events[-1].get("event") == "done"


def test_final_sync_waits_out_an_in_flight_upload(env):
    # A slow hook still running when the loop ends: the teardown sync must wait
    # for it and then run once more, so the newest checkpoint lands. Before the
    # `final=True` path this checkpoint was dropped.
    dest = env / "synced"
    dest.mkdir()
    T.train(
        _cfg(
            run_name="w",
            steps=3,
            eval_interval=1,
            ckpt_interval=1,
            sync_cmd=f"sleep 1; cp -r {{run}}/. {dest}/",
        )
    )
    latest = json.loads((env / "runs" / "w" / "checkpoints" / "latest.json").read_text())
    assert (dest / "checkpoints" / latest["path"] / "model.safetensors").exists()
