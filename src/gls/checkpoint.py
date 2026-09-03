"""Checkpointing and resume.

A checkpoint is a directory ``checkpoints/step-NNNNNN/`` holding:

* ``model.safetensors`` + ``config.json`` - weights plus the architecture dims,
  loadable on their own by ``load_model_dir`` and by the Llama converter. This
  pair is the deliverable a checkpoint is.
* ``trainer.pt`` - everything else needed to continue the run byte-for-byte:
  optimizer moments, the step counter, CPU + CUDA RNG, the data sampler's
  generator state, and the resolved ``TrainConfig`` dict.

``checkpoints/latest.json`` and ``checkpoints/best.json`` are pointer files.
Saves are written to a ``.tmp`` sibling and renamed, so an interrupted save
never leaves a half-written directory that ``latest.json`` references.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from gls.model import GLSModel, ModelConfig

CKPT_SUBDIR = "checkpoints"


# --------------------------------------------------------------------------- #
# model dir - weights + config, kept stable                                   #
# --------------------------------------------------------------------------- #


def save_model_dir(model: GLSModel, out_dir: Path, step: int) -> None:
    """``safetensors`` weights + ``config.json`` in one directory - loadable
    without external context."""
    from safetensors.torch import save_model

    out_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, str(out_dir / "model.safetensors"), metadata={"step": str(step)})
    (out_dir / "config.json").write_text(json.dumps(model.cfg.to_dict(), indent=2))


def load_model_dir(out_dir: Path, device: str = "cpu") -> GLSModel:
    from safetensors.torch import load_model

    cfg = ModelConfig.from_dict(json.loads((out_dir / "config.json").read_text()))
    model = GLSModel(cfg).to(device)
    load_model(model, str(out_dir / "model.safetensors"))
    return model


# --------------------------------------------------------------------------- #
# trainer state                                                               #
# --------------------------------------------------------------------------- #


@dataclass
class TrainerState:
    """The non-weight half of a checkpoint. ``step`` is the number of optimizer
    steps already completed, so training resumes at ``step``."""

    step: int
    optimizer: dict[str, Any]
    train_config: dict[str, Any]
    cpu_rng: torch.Tensor
    cuda_rng: list[torch.Tensor]
    sampler_rng: torch.Tensor
    best_val: float | None = None

    @classmethod
    def capture(
        cls,
        step: int,
        optimizer: torch.optim.Optimizer,
        train_config: dict[str, Any],
        sampler_gen: torch.Generator,
        best_val: float | None,
    ) -> TrainerState:
        return cls(
            step=step,
            optimizer=optimizer.state_dict(),
            train_config=train_config,
            cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            sampler_rng=sampler_gen.get_state(),
            best_val=best_val,
        )

    def restore_rng(self, sampler_gen: torch.Generator) -> None:
        torch.set_rng_state(self.cpu_rng)
        if self.cuda_rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.cuda_rng)
        sampler_gen.set_state(self.sampler_rng)


# --------------------------------------------------------------------------- #
# pointer files                                                               #
# --------------------------------------------------------------------------- #


def _write_pointer(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.rename(path)


def _read_pointer(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def latest_dir(run_dir: Path) -> Path | None:
    ptr = _read_pointer(run_dir / CKPT_SUBDIR / "latest.json")
    if ptr is None:
        return None
    d = run_dir / CKPT_SUBDIR / ptr["path"]
    return d if d.exists() else None


def best_dir(run_dir: Path) -> Path | None:
    ptr = _read_pointer(run_dir / CKPT_SUBDIR / "best.json")
    if ptr is None:
        return None
    d = run_dir / CKPT_SUBDIR / ptr["path"]
    return d if d.exists() else None


# --------------------------------------------------------------------------- #
# save/load/rotate                                                        #
# --------------------------------------------------------------------------- #


def save(
    run_dir: Path,
    step: int,
    model: GLSModel,
    state: TrainerState,
    *,
    val_loss: float | None = None,
    keep_last: int = 3,
    min_improvement: float = 0.0,
) -> Path:
    """Write ``checkpoints/step-NNNNNN/`` atomically, update ``latest.json``,
    update ``best.json`` when ``val_loss`` improves by more than
    ``min_improvement``, then rotate.

    The threshold matters because ``best.json`` is what ``resolve_init`` hands a
    fine-tune. An unguarded ``<`` chases eval noise and can pin a checkpoint that
    never finished its LR schedule - see the note in ``gls.train``.
    """
    ckpt_root = run_dir / CKPT_SUBDIR
    ckpt_root.mkdir(parents=True, exist_ok=True)
    name = f"step-{step:06d}"
    final = ckpt_root / name
    tmp = ckpt_root / f"{name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)

    save_model_dir(model, tmp, step)
    torch.save(state, tmp / "trainer.pt")
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)

    _write_pointer(ckpt_root / "latest.json", {"path": name, "step": step})

    prev_best = _read_pointer(ckpt_root / "best.json")
    if val_loss is not None and (
        prev_best is None or val_loss < prev_best["val_loss"] - min_improvement
    ):
        _write_pointer(ckpt_root / "best.json", {"path": name, "step": step, "val_loss": val_loss})

    _rotate(ckpt_root, keep_last)
    return final


def _rotate(ckpt_root: Path, keep_last: int) -> None:
    """Keep the ``keep_last`` newest ``step-*`` dirs plus whatever ``best.json``
    pins. ``keep_last <= 0`` disables rotation."""
    if keep_last <= 0:
        return
    protected: set[str] = set()
    for ptr_name in ("latest.json", "best.json"):
        ptr = _read_pointer(ckpt_root / ptr_name)
        if ptr:
            protected.add(ptr["path"])

    dirs = sorted(
        (d for d in ckpt_root.glob("step-*") if d.is_dir() and not d.name.endswith(".tmp")),
        key=lambda d: int(d.name.split("-")[1]),
    )
    keep = {d.name for d in dirs[-keep_last:]} | protected
    for d in dirs:
        if d.name not in keep:
            shutil.rmtree(d)


def load(ckpt_dir: Path, device: str = "cpu") -> tuple[GLSModel, TrainerState]:
    model = load_model_dir(ckpt_dir, device)
    state: TrainerState = torch.load(ckpt_dir / "trainer.pt", weights_only=False)
    return model, state


def resolve_init(spec: str) -> Path:
    """Resolve a weights-only spec to a checkpoint directory. Backs both
    ``gls train --init-from`` and ``gls eval --ckpt``.

    ``spec`` is either a checkpoint dir (holds ``model.safetensors``) or a run
    dir, in which case ``best.json`` is preferred over ``latest.json``. Unlike
    ``resolve_resume`` this loads weights only - the optimizer, step and RNG all
    start fresh - so a fine-tune is a new run that inherits weights, not a
    continuation.
    """
    p = Path(spec)
    if (p / "model.safetensors").exists():
        return p
    for finder in (best_dir, latest_dir):
        d = finder(p)
        if d is not None:
            return d
    raise SystemExit(f"{spec}: no checkpoint here or under {p / CKPT_SUBDIR}")


def resolve_resume(run_dir: Path, spec: str | None) -> Path | None:
    """``None`` -> no resume. ``"auto"`` -> ``latest.json`` under ``run_dir`` (or
    ``None`` if there is none - a fresh run). Anything else -> that path."""
    if spec is None:
        return None
    if spec == "auto":
        return latest_dir(run_dir)
    p = Path(spec)
    if not p.exists():
        raise SystemExit(f"--resume {spec}: no such checkpoint directory")
    return p
