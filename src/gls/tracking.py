"""Metric tracking: a JSONL file that is always the source of truth, plus an
optional Weights & Biases mirror.

``_log`` used to be a bare function in ``train.py`` whose docstring promised a
W&B mirror would be "one conditional here". This is that conditional, lifted out
so the loop stays readable. The JSONL contract is unchanged - one JSON object
per line, appended - so every tool that already reads ``runs/*/log.jsonl`` keeps
working.

W&B is opt-in (``--wandb``) and best-effort: a missing API key or a network
outage downgrades to file-only with a warning printed once. It never raises into
the training loop. The W&B run id is persisted in the run dir so ``--resume``
reattaches to the same curve instead of drawing a second one.

Named ``tracking`` rather than ``logging`` on purpose - a module named
``gls.logging`` shadows the stdlib inside the package.
"""

from __future__ import annotations

import json
import secrets
import sys
from pathlib import Path
from typing import Any

_RUN_ID_FILE = "wandb_run_id"


class Run:
    """One training run's metric sink.

    ``event(record)`` writes a structural row (``start``/``done``/``abort``)
    to JSONL only. ``log(record, step=...)`` writes a metric row to JSONL and,
    when W&B is live, mirrors the numeric entries under their namespaced keys
    (``train/loss``, ``eval/val_loss``, ...).
    """

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.log_path = self.run_dir / "log.jsonl"
        self._wandb = None

    # -- jsonl ------------------------------------------------------------- #

    def _append(self, record: dict) -> None:
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    def event(self, record: dict) -> None:
        self._append(record)

    def log(self, record: dict[str, Any], step: int) -> None:
        self._append({"step": step, **record})
        if self._wandb is not None:
            scalars = {k: v for k, v in record.items() if isinstance(v, int | float)}
            try:
                self._wandb.log(scalars, step=step)
            except Exception as exc:  # noqa: BLE001 - tracking must not kill a run
                print(f"[tracking] wandb.log failed, file-only from here: {exc}", file=sys.stderr)
                self._wandb = None

    # -- wandb ------------------------------------------------------------- #

    def start_wandb(
        self,
        *,
        project: str,
        run_name: str | None,
        config: dict,
        resuming: bool,
    ) -> None:
        try:
            import wandb
        except ImportError:
            print(
                "[tracking] --wandb given but wandb is not installed (`uv sync --group track`); "
                "file-only.",
                file=sys.stderr,
            )
            return

        id_path = self.run_dir / _RUN_ID_FILE
        if resuming and id_path.exists():
            run_id = id_path.read_text().strip()
        else:
            run_id = secrets.token_hex(4)
        try:
            self._wandb = wandb.init(
                project=project,
                name=run_name,
                id=run_id,
                resume="allow",
                dir=str(self.run_dir),
                config=config,
            )
            id_path.write_text(run_id)
        except Exception as exc:  # noqa: BLE001 - see above
            print(f"[tracking] wandb.init failed, file-only: {exc}", file=sys.stderr)
            self._wandb = None

    def finish(self) -> None:
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:  # noqa: BLE001
                pass
