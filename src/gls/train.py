"""Hand-written pretraining loop. No ``Trainer``, no ``accelerate`` - every
mechanic (forward, backward, grad accumulation, clip, optimizer step, LR
schedule, eval, checkpoint) is visible in ``train()`` below.

Config comes from a dataclass turned into a CLI and a TOML format by
``gls.config`` (declare a field once). Checkpoints, rotation and resume live in
``gls.checkpoint``; metric logging and the optional W&B mirror in
``gls.tracking``. Phase 1.5 replaces this single-process loop with hand-rolled
multi-GPU orchestration - easier on top of a loop you own than a framework's.

    python -m gls.train --config configs/small-tinystories.toml
    python -m gls.train --config configs/medium-fineweb.toml --resume auto
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import torch
from torch import nn

from gls import checkpoint, env
from gls.checkpoint import TrainerState
from gls.checkpoint import load_model_dir as load_checkpoint
from gls.checkpoint import save_model_dir as save_checkpoint
from gls.config import add_dataclass_args, resolve
from gls.data import PackedData, StreamingTokens, resolve_source
from gls.model import PRESETS, GLSModel
from gls.tracking import Run

# load_checkpoint/save_checkpoint are re-exported: the Phase 0 public names.
__all__ = ["TrainConfig", "train", "lr_at", "param_groups", "load_checkpoint", "save_checkpoint"]

# --------------------------------------------------------------------------- #
# run config                                                                  #
# --------------------------------------------------------------------------- #


def _f(help_txt: str, **kw):
    return field(metadata={"help": help_txt}, **kw)


@dataclass
class TrainConfig:
    tier: str = _f("model preset", default="small")
    corpus: str = _f("corpus name (see gls.data.CORPORA)", default="tinystories")
    steps: int = _f("optimizer steps to run to", default=3_000)
    # batch 16 x 8 accum x 512 ctx = 65k tokens/step. batch is deliberately
    # modest: at vocab 32000 the cross-entropy logit buffer, not the 21.7M
    # weights, is what sets `small`'s memory on an 11.6 GiB card.
    batch_size: int = _f("micro-batch rows", default=16)
    grad_accum: int = _f("micro-batches per optimizer step", default=8)
    block_size: int | None = _f("context length; default = tier max_seq_len", default=None)

    lr: float = _f("peak learning rate", default=3e-4)
    min_lr_frac: float = _f("floor LR as a fraction of peak", default=0.1)
    warmup_steps: int = _f("linear warmup length", default=100)
    weight_decay: float = _f("AdamW weight decay (2D params only)", default=0.1)
    grad_clip: float = _f("grad-norm clip; <=0 disables", default=1.0)
    betas: tuple[float, float] = _f("AdamW betas", default=(0.9, 0.95))

    stream: bool = _f("tokenize from the HF stream instead of packed shards", default=False)
    val_fraction: float = _f("carve this tail fraction of train as val", default=0.0)

    log_interval: int = _f("steps between train-metric rows", default=10)
    eval_interval: int = _f("steps between eval passes", default=250)
    eval_iters: int = _f("batches per eval pass", default=50)
    ckpt_interval: int = _f("steps between checkpoints", default=1_000)
    keep_last: int = _f("checkpoints to retain (best is always kept)", default=3)

    resume: str | None = _f("'auto', a checkpoint dir, or unset", default=None)
    wandb: bool = _f("mirror metrics to Weights & Biases", default=False)
    wandb_project: str = _f("W&B project", default="gls-model")
    wandb_run_name: str | None = _f("W&B run name; default = run dir name", default=None)

    seed: int = _f("RNG seed", default=1337)
    device: str | None = _f("torch device; default = cuda if available", default=None)
    run_name: str | None = _f("run dir name under runs/; default = timestamp", default=None)
    out: str | None = _f("explicit run dir path (overrides run_name)", default=None)


def _resolve_device(name: str | None) -> str:
    if name:
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay to ``min_lr_frac * lr``. A plain function
    of the step - no torch scheduler object to hide the curve."""
    min_lr = cfg.lr * cfg.min_lr_frac
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.steps:
        return min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (cfg.lr - min_lr)


def param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Decay 2D+ tensors (matmuls, embeddings), leave 1D tensors (norms, biases)
    undecayed - the standard split."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# --------------------------------------------------------------------------- #
# eval                                                                        #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def evaluate(model: GLSModel, data, cfg: TrainConfig, device: str, autocast, step: int) -> dict:
    model.eval()
    out = {}
    # A per-step generator: eval is reproducible and never perturbs the training
    # sample stream (so resume determinism does not depend on the eval cadence).
    gen = torch.Generator().manual_seed(cfg.seed + 1_000_003 * step)
    splits = [("train", False)]
    if data.has_val:
        splits.append(("val", True))
    for name, is_val in splits:
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = data.batch(cfg.batch_size, device, val=is_val, generator=gen)
            with autocast:
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


# --------------------------------------------------------------------------- #
# loop                                                                        #
# --------------------------------------------------------------------------- #


def _make_data(cfg: TrainConfig, block_size: int, skip_docs: int):
    if cfg.stream:
        return StreamingTokens(cfg.corpus, block_size, skip_docs=skip_docs)
    source = resolve_source(cfg.corpus, "train")
    if source is None:
        raise SystemExit(
            f"nothing prepared for {cfg.corpus!r}; "
            "run `python -m gls.data prepare --corpus {cfg.corpus} --split train`"
        )
    return PackedData(
        source,
        block_size,
        val_source=resolve_source(cfg.corpus, "val"),
        val_fraction=cfg.val_fraction,
    )


def train(cfg: TrainConfig) -> Path:
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")  # TF32 for the fp32 matmuls outside autocast
    device = _resolve_device(cfg.device)
    model_cfg = PRESETS[cfg.tier]
    block_size = cfg.block_size or model_cfg.max_seq_len

    if cfg.out:
        run_dir = Path(cfg.out)
    else:
        name = cfg.run_name or time.strftime("%Y%m%d-%H%M%S")
        run_dir = Path("runs") / name
    run_dir.mkdir(parents=True, exist_ok=True)

    resume_from = checkpoint.resolve_resume(run_dir, cfg.resume)
    resuming = resume_from is not None

    run = Run(run_dir)
    info = env.describe()
    (run_dir / "env.json").write_text(_json(info))
    (run_dir / "train_config.json").write_text(_json(_asdict(cfg)))

    # data sampler RNG: an explicit generator so its state travels in the
    # checkpoint and no window is replayed on resume.
    sampler_gen = torch.Generator().manual_seed(cfg.seed)

    start_step = 0
    best_val: float | None = None
    skip_docs = 0
    state = None
    if resuming:
        model, state = checkpoint.load(resume_from, device)
        state.restore_rng(sampler_gen)
        start_step = state.step
        best_val = state.best_val
        skip_docs = state.train_config.get("_stream_docs", 0) if cfg.stream else 0
        print(f"resuming {run_dir.name} from step {start_step}")
    else:
        model = GLSModel(model_cfg).to(device)

    # Optimizer is built against the final model object, then its state loaded -
    # rebuilding it before the resume swap would leave it pointing at orphaned
    # parameter tensors.
    opt = torch.optim.AdamW(
        param_groups(model, cfg.weight_decay),
        lr=cfg.lr,
        betas=cfg.betas,
        fused=(device == "cuda"),
    )
    if state is not None:
        opt.load_state_dict(state.optimizer)

    if start_step >= cfg.steps:
        print(f"already at step {start_step} >= steps {cfg.steps}; nothing to do")
        return run_dir

    data = _make_data(cfg, block_size, skip_docs)

    run.event(
        {
            "event": "resume" if resuming else "start",
            "step": start_step,
            "train_cfg": _asdict(cfg),
            "model_cfg": model_cfg.to_dict(),
            "block_size": block_size,
            "device": device,
            "env": info,
        }
    )
    if cfg.wandb:
        run.start_wandb(
            project=cfg.wandb_project,
            run_name=cfg.wandb_run_name or run_dir.name,
            config={**_asdict(cfg), "model": model_cfg.to_dict()},
            resuming=resuming,
        )

    n_params = model.num_parameters()
    print(f"tier={cfg.tier}  params={n_params:,}  device={device}  block_size={block_size}")

    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else contextlib.nullcontext()
    )
    tokens_per_step = cfg.batch_size * cfg.grad_accum * block_size
    t0 = time.time()
    last_eval: dict = {}

    def _cadence(step: int, interval: int) -> bool:
        return (step + 1) % interval == 0 or step == cfg.steps - 1

    for step in range(start_step, cfg.steps):
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        step_t0 = time.time()
        loss_accum = torch.zeros((), device=device)
        for _ in range(cfg.grad_accum):
            x, y = data.batch(cfg.batch_size, device, generator=sampler_gen)
            with autocast:
                _, loss = model(x, y)
            (loss / cfg.grad_accum).backward()
            loss_accum += loss.detach() / cfg.grad_accum

        grad_norm = (
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
            if cfg.grad_clip > 0
            else _global_grad_norm(model)
        )
        opt.step()
        opt.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.synchronize()

        loss_val = loss_accum.item()
        if not math.isfinite(loss_val):
            run.event({"event": "abort", "step": step, "reason": f"non-finite loss {loss_val}"})
            raise SystemExit(f"non-finite loss at step {step}: {loss_val}")

        dt = time.time() - step_t0
        is_first = step == start_step

        if _cadence(step, cfg.log_interval) and not is_first:
            run.log(
                {
                    "train/loss": round(loss_val, 4),
                    "train/lr": lr,
                    "train/grad_norm": round(grad_norm, 3),
                    "train/tokens_per_sec": round(tokens_per_step / dt),
                    "elapsed_s": round(time.time() - t0, 1),
                },
                step=step,
            )

        if _cadence(step, cfg.eval_interval):
            last_eval = evaluate(model, data, cfg, device, autocast, step)
            run.log({f"eval/{k}_loss": round(v, 4) for k, v in last_eval.items()}, step=step)
            print(
                f"step {step:>6}  loss {loss_val:6.4f}  eval {last_eval}  lr {lr:.2e}  "
                f"gnorm {grad_norm:5.2f}"
            )

        if _cadence(step, cfg.ckpt_interval):
            val_now = last_eval.get("val")
            if val_now is not None and (best_val is None or val_now < best_val):
                best_val = val_now
            state = TrainerState.capture(
                step=step + 1,
                optimizer=opt,
                train_config=_ckpt_cfg(cfg, data),
                sampler_gen=sampler_gen,
                best_val=best_val,
            )
            checkpoint.save(
                run_dir, step + 1, model, state, val_loss=val_now, keep_last=cfg.keep_last
            )

    run.event({"event": "done", "total_s": round(time.time() - t0, 1), "step": cfg.steps})
    run.finish()
    if isinstance(data, StreamingTokens):
        data.close()
    print(f"done in {time.time() - t0:.1f}s -> {run_dir}")
    return run_dir


# --------------------------------------------------------------------------- #
# small helpers                                                               #
# --------------------------------------------------------------------------- #


def _json(obj) -> str:
    import json

    return json.dumps(obj, indent=2, default=str)


def _asdict(cfg: TrainConfig) -> dict:
    return {f.name: getattr(cfg, f.name) for f in fields(cfg)}


def _ckpt_cfg(cfg: TrainConfig, data) -> dict:
    d = _asdict(cfg)
    if isinstance(data, StreamingTokens):
        d["_stream_docs"] = data.docs_consumed
    return d


def _global_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().norm() ** 2
    return float(total**0.5)


# --------------------------------------------------------------------------- #
# cli                                                                         #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gls.train", description=__doc__.splitlines()[0])
    p.add_argument("--config", default=None, help="TOML run config; CLI flags override it")
    add_dataclass_args(p, TrainConfig)
    cfg = resolve(TrainConfig, p, argv)
    if cfg.tier not in PRESETS:
        raise SystemExit(f"unknown tier {cfg.tier!r}; choose from {list(PRESETS)}")
    train(cfg)
    return 0


if __name__ == "__main__":
    _rc = main()
    # --stream keeps HF's parquet reader alive on a background thread that can
    # fault during interpreter shutdown. The run is fully written and W&B
    # already finished by here, so bypass finalization (mirrors gls.data).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc)
