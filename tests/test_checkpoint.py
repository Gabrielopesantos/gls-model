"""gls.checkpoint - atomic saves, rotation, resume equivalence."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from gls.checkpoint import CKPT_SUBDIR, TrainerState, latest_dir, resolve_resume, save
from gls.data import PackedData
from gls.model import PRESETS, GLSModel
from gls.train import TrainConfig, lr_at
from gls.trainer import Runtime, Trainer


def _tiny():
    from dataclasses import replace

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


def _state(step, opt, gen, best=None):
    return TrainerState.capture(step, opt, {"tier": "small"}, gen, best)


def test_save_is_atomic_and_updates_pointers(tmp_path):
    model = GLSModel(_tiny())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)

    save(tmp_path, 100, model, _state(100, opt, gen), val_loss=2.0, keep_last=3)
    d = latest_dir(tmp_path)
    assert d is not None and d.name == "step-000100"
    assert (d / "model.safetensors").exists() and (d / "trainer.pt").exists()
    assert json.loads((tmp_path / CKPT_SUBDIR / "best.json").read_text())["val_loss"] == 2.0
    assert not list((tmp_path / CKPT_SUBDIR).glob("*.tmp"))


def test_best_pointer_only_moves_on_improvement(tmp_path):
    model = GLSModel(_tiny())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)
    save(tmp_path, 1, model, _state(1, opt, gen), val_loss=2.0, keep_last=5)
    save(tmp_path, 2, model, _state(2, opt, gen), val_loss=3.0, keep_last=5)  # worse
    save(tmp_path, 3, model, _state(3, opt, gen), val_loss=1.0, keep_last=5)  # better
    best = json.loads((tmp_path / CKPT_SUBDIR / "best.json").read_text())
    assert best["step"] == 3 and best["val_loss"] == 1.0


def test_rotation_keeps_last_n_plus_best(tmp_path):
    model = GLSModel(_tiny())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)
    save(tmp_path, 10, model, _state(10, opt, gen), val_loss=0.5, keep_last=2)  # the best
    for s in (20, 30, 40, 50):
        save(tmp_path, s, model, _state(s, opt, gen), val_loss=9.0, keep_last=2)
    kept = sorted(d.name for d in (tmp_path / CKPT_SUBDIR).glob("step-*"))
    assert kept == ["step-000010", "step-000040", "step-000050"]


def test_resolve_resume_auto(tmp_path):
    assert resolve_resume(tmp_path, None) is None
    assert resolve_resume(tmp_path, "auto") is None  # fresh run, no latest.json
    model = GLSModel(_tiny())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)
    save(tmp_path, 5, model, _state(5, opt, gen), keep_last=3)
    resumed = resolve_resume(tmp_path, "auto")
    assert resumed is not None and resumed.name == "step-000005"


def test_state_restores_optimizer_and_sampler(tmp_path):
    from gls.checkpoint import load

    torch.manual_seed(0)
    model = GLSModel(_tiny())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(123)
    # a couple of fake steps so AdamW has moment buffers
    for _ in range(3):
        model(torch.randint(0, 256, (2, 16)), torch.randint(0, 256, (2, 16)))[1].backward()
        opt.step()
        opt.zero_grad()
    save(tmp_path, 3, model, _state(3, opt, gen), keep_last=3)
    # what gen produces from here is the continuation a resume must reproduce
    expected = torch.randint(0, 100, (8,), generator=gen).tolist()

    ckpt = latest_dir(tmp_path)
    assert ckpt is not None
    _, state = load(ckpt)
    assert state.step == 3
    gen2 = torch.Generator()
    state.restore_rng(gen2)
    assert torch.randint(0, 100, (8,), generator=gen2).tolist() == expected

    # optimizer moments came back too
    _, restored_state = load(ckpt)
    assert restored_state.optimizer["state"]


def test_trainer_init_from_is_fresh_optimizer_inherited_weights(tmp_path):
    from gls.checkpoint import load, resolve_init

    torch.manual_seed(0)
    model = GLSModel(_tiny())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)
    for _ in range(3):  # give AdamW real moment buffers in the source run
        model(torch.randint(0, 256, (2, 16)), torch.randint(0, 256, (2, 16)))[1].backward()
        opt.step()
        opt.zero_grad()
    save(tmp_path / "base", 7, model, _state(7, opt, gen), keep_last=3)

    cfg = TrainConfig(steps=10, batch_size=2, grad_accum=1, block_size=16)
    trainer = Trainer.init_from(resolve_init(str(tmp_path / "base")), cfg, Runtime.resolve("cpu"))

    # weights inherited...
    base_ckpt = latest_dir(tmp_path / "base")
    assert base_ckpt is not None
    src_model, _ = load(base_ckpt)
    for (_, a), (_, b) in zip(
        trainer.model.state_dict().items(), src_model.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b)
    # ...optimizer and step are not: a fine-tune starts at zero.
    assert all(not st for st in trainer.opt.state_dict()["state"].values())


def test_lr_schedule_is_step_addressable():
    # resume relies on lr_at(step) being a pure function, not a stateful sched
    cfg = TrainConfig(lr=1e-3, warmup_steps=10, steps=100)
    assert lr_at(400, cfg) == lr_at(400, cfg)
    assert lr_at(5, cfg) < lr_at(10, cfg)  # still warming up


def test_trainer_resume_reproduces_uninterrupted_run(tmp_path):
    """The Model/Trainer split's load-bearing property: 2 steps + save + resume +
    2 steps lands on the same losses as 4 steps straight through."""
    cfg = TrainConfig(steps=4, batch_size=2, grad_accum=1, block_size=16, warmup_steps=1, seed=0)
    rt = Runtime.resolve("cpu")

    period = np.tile(np.arange(32, dtype=np.uint16), 2_000)
    bin_path = tmp_path / "toy.bin"
    period.tofile(bin_path)

    def run(trainer, gen, lo, hi):
        data = PackedData(bin_path, block_size=16)
        out = []
        for step in range(lo, hi):
            trainer.set_lr(lr_at(step, cfg))
            out.append(trainer.train_step(data, gen)[0])
        return out

    torch.manual_seed(0)
    straight = run(Trainer.fresh(_tiny(), cfg, rt), torch.Generator().manual_seed(0), 0, 4)

    torch.manual_seed(0)
    part = Trainer.fresh(_tiny(), cfg, rt)
    gen = torch.Generator().manual_seed(0)
    run(part, gen, 0, 2)
    part.save(
        tmp_path / "r", 2, gen, PackedData(bin_path, block_size=16), val_loss=None, best_val=None
    )

    ckpt = latest_dir(tmp_path / "r")
    assert ckpt is not None
    gen_r = torch.Generator()
    resumed, state = Trainer.resume(ckpt, cfg, rt, gen_r)  # restores gen_r in place
    assert state.step == 2
    tail = run(resumed, gen_r, 2, 4)

    assert tail == pytest.approx(straight[2:], rel=1e-4)
