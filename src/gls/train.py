"""Hand-written pretraining loop. The LR schedule, cadence gates, logging
and the teardown are all visible in ``train()`` below.
The optimizer, the step and the eval pass live in
``gls.trainer`` (``Trainer``/``Runtime``); the loop calls ``train_step``/
``eval_step``/``save`` and owns nothing else.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from gls import checkpoint, env, paths
from gls.data import PackedData, StreamingTokens, TokenSource, resolve_source
from gls.model import PRESETS
from gls.tracking import Run
from gls.trainer import Runtime, Trainer

__all__ = ["TrainConfig", "train", "lr_at"]

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
    # batch 16 x 8 accum x 512 ctx = 65k tokens/step. batch is modest:
    # at vocab 32000 the cross-entropy logit buffer, not the 21.7M
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


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay to ``min_lr_frac * lr``. A plain function
    of the step and pure so a resumed run lands on the same value."""
    min_lr = cfg.lr * cfg.min_lr_frac
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.steps:
        return min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (cfg.lr - min_lr)


# --------------------------------------------------------------------------- #
# loop                                                                        #
# --------------------------------------------------------------------------- #


def _make_data(cfg: TrainConfig, block_size: int, skip_docs: int) -> TokenSource:
    if cfg.stream:
        return StreamingTokens(cfg.corpus, block_size, skip_docs=skip_docs)
    source = resolve_source(cfg.corpus, "train")
    if source is None:
        raise SystemExit(
            f"nothing prepared for {cfg.corpus!r}; "
            f"run `gls data prepare --corpus {cfg.corpus} --split train`"
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
    rt = Runtime.resolve(cfg.device)
    model_cfg = PRESETS[cfg.tier]
    block_size = cfg.block_size or model_cfg.max_seq_len

    if cfg.out:
        run_dir = Path(cfg.out)
    else:
        run_dir = paths.runs_dir() / (cfg.run_name or time.strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    resume_from = checkpoint.resolve_resume(run_dir, cfg.resume)
    resuming = resume_from is not None

    run = Run(run_dir)
    info = env.describe()
    (run_dir / "env.json").write_text(_json(info))
    (run_dir / "train_config.json").write_text(_json(asdict(cfg)))

    # data sampler RNG: an explicit generator so its state travels in the
    # checkpoint and no window is replayed on resume.
    sampler_gen = torch.Generator().manual_seed(cfg.seed)

    if resuming:
        trainer, state = Trainer.resume(resume_from, cfg, rt, sampler_gen)
        start_step = state.step
        best_val = state.best_val
        skip_docs = state.train_config.get("_stream_docs", 0)
        _note(f"resuming {run_dir.name} from step {start_step}")
    else:
        trainer = Trainer.fresh(model_cfg, cfg, rt)
        start_step, best_val, skip_docs = 0, None, 0

    if start_step >= cfg.steps:
        _note(f"already at step {start_step} >= steps {cfg.steps}; nothing to do")
        return run_dir

    data = _make_data(cfg, block_size, skip_docs)

    run.event(
        {
            "event": "resume" if resuming else "start",
            "step": start_step,
            "train_cfg": asdict(cfg),
            "model_cfg": model_cfg.to_dict(),
            "block_size": block_size,
            "device": rt.device,
            "env": info,
        }
    )
    if cfg.wandb:
        run.start_wandb(
            project=cfg.wandb_project,
            run_name=cfg.wandb_run_name or run_dir.name,
            config={**asdict(cfg), "model": model_cfg.to_dict()},
            resuming=resuming,
        )

    _note(
        f"tier={cfg.tier}  params={trainer.model.num_parameters():,}  "
        f"device={rt.device}  block_size={block_size}"
    )

    tokens_per_step = cfg.batch_size * cfg.grad_accum * block_size
    t0 = time.time()
    last_eval: dict[str, float] = {}

    def _due(step: int, interval: int) -> bool:
        return (step + 1) % interval == 0 or step == cfg.steps - 1

    for step in range(start_step, cfg.steps):
        lr = lr_at(step, cfg)
        trainer.set_lr(lr)

        step_t0 = time.time()
        loss_val, grad_norm = trainer.train_step(data, sampler_gen)

        if not math.isfinite(loss_val):
            run.event({"event": "abort", "step": step, "reason": f"non-finite loss {loss_val}"})
            raise SystemExit(f"non-finite loss at step {step}: {loss_val}")

        dt = time.time() - step_t0
        is_first = step == start_step

        if _due(step, cfg.log_interval) and not is_first:
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

        if _due(step, cfg.eval_interval):
            last_eval = trainer.eval_step(data, step)
            run.log({f"eval/{k}_loss": round(v, 4) for k, v in last_eval.items()}, step=step)
            _note(
                f"step {step:>6}  loss {loss_val:6.4f}  eval {last_eval}  "
                f"lr {lr:.2e}  gnorm {grad_norm:5.2f}"
            )

        if _due(step, cfg.ckpt_interval):
            val_now = last_eval.get("val")
            if val_now is not None and (best_val is None or val_now < best_val):
                best_val = val_now
            trainer.save(run_dir, step + 1, sampler_gen, data, val_loss=val_now, best_val=best_val)

    run.event({"event": "done", "total_s": round(time.time() - t0, 1), "step": cfg.steps})
    run.finish()
    data.close()
    _note(f"done in {time.time() - t0:.1f}s -> {run_dir}")
    return run_dir


# --------------------------------------------------------------------------- #
# small helpers                                                               #
# --------------------------------------------------------------------------- #


def _json(obj: object) -> str:
    return json.dumps(obj, indent=2, default=str)


def _note(msg: str) -> None:
    """Progress/diagnostic line to stderr - stdout is reserved for program
    output."""
    print(f"[train] {msg}", file=sys.stderr)
