"""Package-level guarantees: the inference/training import seam and lazy exports."""

from __future__ import annotations

import subprocess
import sys


def test_importing_model_does_not_pull_the_training_stack():
    """Inference is ``gls.model`` + ``gls.checkpoint.load_model_dir``. A
    fresh interpreter that imports only those must not drag in the optimizer,
    the loop, or the CLI - run out-of-process so nothing this suite already
    imported hides a regression."""
    code = (
        "import sys; import gls.model, gls.checkpoint; "
        "leaked = [m for m in ('gls.trainer', 'gls.train', 'gls.cli') if m in sys.modules]; "
        "print(leaked); sys.exit(1 if leaked else 0)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, f"training modules leaked into the inference path: {out.stdout}"


def test_lazy_exports_resolve():
    import gls

    assert gls.Trainer is gls.Trainer  # __getattr__ resolves and is stable
    assert gls.__version__
    assert "Trainer" in dir(gls)
