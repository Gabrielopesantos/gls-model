"""Hand-written pretraining loop. The LR schedule, cadence gates, logging
and the teardown are all visible in ``train()`` below.
The optimizer, the step and the eval pass live in
``gls.trainer`` (``Trainer``/``Runtime``); the loop calls ``train_step``/
``eval_step``/``save`` and owns nothing else.
"""

from __future__ import annotations

import json
import math
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from gls import checkpoint, env, paths, sft
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
    compile: bool = _f("wrap the forward pass in torch.compile", default=False)

    stream: bool = _f("tokenize from the HF stream instead of packed shards", default=False)
    val_fraction: float = _f("carve this tail fraction of train as val", default=0.0)

    log_interval: int = _f("steps between train-metric rows", default=10)
    eval_interval: int = _f("steps between eval passes", default=250)
    eval_iters: int = _f("batches per eval pass", default=50)
    eval_batch_size: int | None = _f(
        "rows per eval batch; default = batch_size. Set it so the eval sample "
        "size stops tracking a batch_size change made for a memory reason.",
        default=None,
    )
    patience: int = _f("stop after N evals with no val improvement; 0 disables", default=0)
    min_improvement: float = _f("val-loss drop that counts as an improvement", default=0.0)
    ckpt_interval: int = _f("steps between checkpoints", default=1_000)
    keep_last: int = _f("checkpoints to retain (best is always kept)", default=3)
    sync_cmd: str | None = _f(
        "shell command run after each checkpoint save; {ckpt}/{run} substituted", default=None
    )

    resume: str | None = _f("'auto', a checkpoint dir, or unset", default=None)
    init_from: str | None = _f(
        "seed weights from a checkpoint/run dir (fine-tune); fresh optimizer, step 0", default=None
    )
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
    if cfg.corpus in sft.SFT_CORPORA:
        return sft.SFTData(
            cfg.corpus,
            block_size,
            val_fraction=cfg.val_fraction or sft.DEFAULT_VAL_FRACTION,
            seed=cfg.seed,
        )
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
        if cfg.init_from:
            _note(
                f"--init-from {cfg.init_from} ignored: resuming {run_dir.name} from its own state"
            )
        trainer, state = Trainer.resume(resume_from, cfg, rt, sampler_gen)
        start_step = state.step
        best_val = state.best_val
        skip_docs = state.train_config.get("_stream_docs", 0)
        _note(f"resuming {run_dir.name} from step {start_step}")
    elif cfg.init_from:
        # A fine-tune: inherit weights, nothing else. --resume on this same run
        # dir still wins (an interrupted SFT continues from its own checkpoint),
        # which is why this is the `elif` and not checked first.
        src = checkpoint.resolve_init(cfg.init_from)
        trainer = Trainer.init_from(src, cfg, rt)
        start_step, best_val, skip_docs = 0, None, 0
        if trainer.model.cfg != model_cfg:
            _note(
                f"warning: --tier {cfg.tier} != the checkpoint architecture; using the checkpoint"
            )
        _note(f"init weights from {src}")
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
    # train/mfu: achieved FLOP/s as a fraction of the card's dense bf16 peak.
    # Logged only when the peak is known (env._PEAK_BF16_FLOPS) - a guessed peak
    # would make the number meaningless.
    flops_per_token = trainer.model.flops_per_token(block_size)
    peak_flops = env.peak_bf16_flops() if rt.is_cuda else None
    t0 = time.time()
    last_eval: dict[str, float] = {}
    stale_evals = 0  # consecutive evals with no significant val improvement
    sync_proc: subprocess.Popen | None = None
    sync_logged = False

    def _due(step: int, interval: int) -> bool:
        return (step + 1) % interval == 0 or step == cfg.steps - 1

    def _sync(ckpt_dir: Path, *, final: bool = False) -> None:
        """Fire ``cfg.sync_cmd`` for a just-written checkpoint. Non-blocking and
        best-effort: a rented box needs each checkpoint copied off its ephemeral
        disk, but a slow or failing upload must never reach into the loop - same
        contract ``gls.tracking`` gives W&B. One upload in flight at a time; a
        slow one skips a checkpoint rather than piling processes up.

        ``final=True`` is the teardown call: it waits out an in-flight upload
        instead of skipping, then blocks on its own. The last checkpoint - the
        whole point of the SIGTERM save - has to land, and by this point the loop
        is over so blocking costs nothing."""
        nonlocal sync_proc
        if not cfg.sync_cmd:
            return
        if sync_proc is not None and sync_proc.poll() is None:
            if not final:
                _note("sync_cmd from the previous checkpoint still running; skipping this one")
                return
            _note("waiting on the in-flight checkpoint sync...")
            sync_proc.wait()
        cmd = cfg.sync_cmd.format(ckpt=str(ckpt_dir), run=str(run_dir))
        nonlocal sync_logged
        if not sync_logged:
            # Log the resolved command once. A sync_cmd that hardcodes a run name
            # instead of using {run} sends this run's checkpoints into another
            # run's destination, and does it silently at every save.
            _note(f"sync_cmd -> {cmd}")
            sync_logged = True
        try:
            sync_proc = subprocess.Popen(cmd, shell=True)
        except OSError as exc:
            _note(f"sync_cmd failed to launch: {exc}")
            return
        if final:
            _note("waiting on the final checkpoint sync...")
            sync_proc.wait()

    def _checkpoint(step_after: int, val_now: float | None) -> None:
        nonlocal best_val
        if val_now is not None and (best_val is None or val_now < best_val):
            best_val = val_now
        ckpt = trainer.save(
            run_dir, step_after, sampler_gen, data, val_loss=val_now, best_val=best_val
        )
        _sync(ckpt)

    # SIGTERM (the preemption or `scancel` signal) checkpoints and exits through
    # the normal teardown. SIGINT keeps its hard-abort default on purpose - a
    # Ctrl-C at the keyboard is "stop now", not "wrap up".
    stop = threading.Event()
    prev_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    outcome = "done"
    last_step = start_step
    try:
        for step in range(start_step, cfg.steps):
            last_step = step
            lr = lr_at(step, cfg)
            trainer.set_lr(lr)

            step_t0 = time.time()
            loss_val, grad_norm = trainer.train_step(data, sampler_gen)

            if not math.isfinite(loss_val):
                run.event({"event": "abort", "step": step, "reason": f"non-finite loss {loss_val}"})
                raise SystemExit(f"non-finite loss at step {step}: {loss_val}")

            dt = time.time() - step_t0
            is_first = step == start_step
            done_steps = step - start_step + 1
            eta_s = (time.time() - t0) / done_steps * (cfg.steps - step - 1)

            # The first step pays the torch.compile trace and its transient
            # allocations; reset the peak once it is behind us so the logged
            # figure is the steady-state training residency the rental is sized
            # against.
            if is_first and rt.is_cuda:
                torch.cuda.reset_peak_memory_stats()

            if _due(step, cfg.log_interval) and not is_first:
                row = {
                    "train/loss": round(loss_val, 4),
                    "train/lr": lr,
                    "train/grad_norm": round(grad_norm, 3),
                    "train/tokens_per_sec": round(tokens_per_step / dt),
                    "train/eta_s": round(eta_s),
                    "elapsed_s": round(time.time() - t0, 1),
                }
                if rt.is_cuda:
                    row["train/peak_mem_gib"] = round(
                        torch.cuda.max_memory_allocated() / 1024**3, 3
                    )
                if peak_flops is not None:
                    row["train/mfu"] = round(flops_per_token * tokens_per_step / dt / peak_flops, 4)
                run.log(row, step=step)

            saved_after: int | None = None

            if _due(step, cfg.eval_interval):
                last_eval = trainer.eval_step(data)
                run.log({f"eval/{k}_loss": round(v, 4) for k, v in last_eval.items()}, step=step)
                _note(
                    f"step {step:>6}  loss {loss_val:6.4f}  eval {last_eval}  "
                    f"lr {lr:.2e}  gnorm {grad_norm:5.2f}  eta {eta_s / 60:.0f}m"
                )
                val_now = last_eval.get("val")
                if val_now is not None:
                    if best_val is None or val_now < best_val - cfg.min_improvement:
                        stale_evals = 0
                    else:
                        stale_evals += 1
                    # Save on a significant improvement, not only at
                    # ckpt_interval marks: a best val landing between two marks
                    # must still reach best.json (checkpoint.save's own compare
                    # only runs when we actually write a checkpoint).
                    #
                    # min_improvement gates this the same way it gates the
                    # patience counter above. Without it a bare `<` chases eval
                    # noise.
                    if best_val is None or val_now < best_val - cfg.min_improvement:
                        _checkpoint(step + 1, val_now)
                        saved_after = step + 1

            if _due(step, cfg.ckpt_interval) and saved_after != step + 1:
                _checkpoint(step + 1, last_eval.get("val"))
                saved_after = step + 1

            if cfg.patience and stale_evals >= cfg.patience:
                outcome = "early_stop"
                break

            if stop.is_set():
                if saved_after != step + 1:
                    _checkpoint(step + 1, last_eval.get("val"))
                outcome = "sigterm"
                break
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)

    total_s = round(time.time() - t0, 1)
    if outcome == "early_stop":
        run.event(
            {
                "event": "early_stop",
                "step": last_step + 1,
                "total_s": total_s,
                "stale_evals": stale_evals,
            }
        )
        _note(f"early stop at step {last_step}: {stale_evals} evals with no val improvement")
    elif outcome == "sigterm":
        run.event({"event": "sigterm", "step": last_step + 1, "total_s": total_s})
        _note(f"SIGTERM at step {last_step}: checkpointed, exiting clean")
    else:
        run.event({"event": "done", "total_s": total_s, "step": cfg.steps})

    run.finish()
    data.close()
    # After run.finish() so the run-tree sync carries the terminal log.jsonl row
    # (the done / sigterm / early_stop event) and the final pointer files.
    _sync(run_dir, final=True)
    _note(f"done in {total_s}s -> {run_dir}")
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
