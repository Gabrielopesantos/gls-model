"""The training-only half of the stack: the optimizer, the step, the eval pass.

``gls.model.GLSModel`` is architecture plus forward - usable for inference with
no optimizer in sight. ``Trainer`` is that model plus the AdamW state, the
resolved device/autocast ``Runtime``, and the three verbs the loop calls:
``train_step``, ``eval_step``, ``save``. The loop itself - schedule, cadence,
logging - stays in ``gls.train`` where every mechanic is visible.
"""

from __future__ import annotations

import contextlib
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from gls import checkpoint
from gls.checkpoint import TrainerState
from gls.model import GLSModel, ModelConfig

if TYPE_CHECKING:
    from gls.data import TokenSource
    from gls.train import TrainConfig


# --------------------------------------------------------------------------- #
# runtime - device-dependent choices, resolved once                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Runtime:
    """Device-dependent choices, resolved once per run."""

    device: str
    autocast: AbstractContextManager
    fused_optimizer: bool
    pin_memory: bool

    @classmethod
    def resolve(cls, device: str | None) -> Runtime:
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cuda = dev.startswith("cuda")
        return cls(
            device=dev,
            autocast=(
                torch.autocast("cuda", dtype=torch.bfloat16) if cuda else contextlib.nullcontext()
            ),
            fused_optimizer=cuda,
            pin_memory=cuda,
        )

    @property
    def is_cuda(self) -> bool:
        return self.device.startswith("cuda")


# --------------------------------------------------------------------------- #
# optimizer helpers                                                           #
# --------------------------------------------------------------------------- #


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


def global_grad_norm(model: nn.Module) -> float:
    """Total L2 norm over all gradients, for logging when clipping is disabled
    (``clip_grad_norm_`` returns this for free when it is on)."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().norm().item() ** 2
    return total**0.5


# --------------------------------------------------------------------------- #
# trainer                                                                     #
# --------------------------------------------------------------------------- #


class Trainer:
    """Model + optimizer + runtime. ``train`` owns the loop; this owns the step."""

    def __init__(self, model: GLSModel, cfg: TrainConfig, rt: Runtime) -> None:
        self.model = model
        self.cfg = cfg
        self.rt = rt
        self.opt = torch.optim.AdamW(
            param_groups(model, cfg.weight_decay),
            lr=cfg.lr,
            betas=cfg.betas,
            fused=rt.fused_optimizer,
        )

    # -- construction --------------------------------------------------- #

    @classmethod
    def fresh(cls, model_cfg: ModelConfig, cfg: TrainConfig, rt: Runtime) -> Trainer:
        return cls(GLSModel(model_cfg).to(rt.device), cfg, rt)

    @classmethod
    def resume(
        cls,
        ckpt_dir: Path,
        cfg: TrainConfig,
        rt: Runtime,
        sampler_gen: torch.Generator,
    ) -> tuple[Trainer, TrainerState]:
        """Load weights, then build the optimizer against that exact model object
        and load its moments into it. Building the optimizer before the weight
        load would leave it pointing at orphaned tensors.
        """
        model, state = checkpoint.load(ckpt_dir, rt.device)
        state.restore_rng(sampler_gen)
        trainer = cls(model, cfg, rt)
        trainer.opt.load_state_dict(state.optimizer)
        return trainer, state

    # -- the step ----------------------------------------------------- #

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr

    def train_step(self, data: TokenSource, gen: torch.Generator) -> tuple[float, float]:
        """One optimizer step over ``grad_accum`` micro-batches: forward, scaled
        backward, clip, step, zero. Returns ``(mean micro-batch loss, grad norm)``."""
        cfg = self.cfg
        self.model.train()
        loss_accum = torch.zeros((), device=self.rt.device)
        for _ in range(cfg.grad_accum):
            x, y = data.batch(
                cfg.batch_size, self.rt.device, generator=gen, pin_memory=self.rt.pin_memory
            )
            with self.rt.autocast:
                _, loss = self.model(x, y)
            (loss / cfg.grad_accum).backward()
            loss_accum += loss.detach() / cfg.grad_accum

        grad_norm = (
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip).item()
            if cfg.grad_clip > 0
            else global_grad_norm(self.model)
        )
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        if self.rt.is_cuda:
            torch.cuda.synchronize()
        return loss_accum.item(), grad_norm

    @torch.no_grad()
    def eval_step(self, data: TokenSource, step: int) -> dict[str, float]:
        """Mean loss over ``eval_iters`` batches, on train and (if present) val.

        A per-step generator: eval is reproducible and never touches the training
        sample stream, so resume determinism does not depend on the eval cadence.
        """
        cfg = self.cfg
        self.model.eval()
        gen = torch.Generator().manual_seed(cfg.seed + 1_000_003 * step)
        out: dict[str, float] = {}
        splits = [("train", False)]
        if data.has_val:
            splits.append(("val", True))
        for name, is_val in splits:
            losses = torch.zeros(cfg.eval_iters)
            for i in range(cfg.eval_iters):
                x, y = data.batch(
                    cfg.batch_size,
                    self.rt.device,
                    val=is_val,
                    generator=gen,
                    pin_memory=self.rt.pin_memory,
                )
                with self.rt.autocast:
                    _, loss = self.model(x, y)
                losses[i] = loss.item()
            out[name] = losses.mean().item()
        self.model.train()
        return out

    # -- persistence ------------------------------------------------ #

    def save(
        self,
        run_dir: Path,
        step: int,
        sampler_gen: torch.Generator,
        data: TokenSource,
        *,
        val_loss: float | None,
        best_val: float | None,
    ) -> Path:
        """Capture trainer state (optimizer moments, RNG, resolved config plus
        whatever the data source needs to resume) and write the checkpoint dir."""
        state = TrainerState.capture(
            step=step,
            optimizer=self.opt,
            train_config={**asdict(self.cfg), **data.checkpoint_state()},
            sampler_gen=sampler_gen,
            best_val=best_val,
        )
        return checkpoint.save(
            run_dir, step, self.model, state, val_loss=val_loss, keep_last=self.cfg.keep_last
        )
