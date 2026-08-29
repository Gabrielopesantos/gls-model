"""Environment introspection.

Backs ``gls env``, and is called at the start of every training or serving run
so the exact torch/CUDA/device combination ends up in the run log alongside the
loss curve. The human-readable rendering lives in ``gls.cli``.
"""

from __future__ import annotations

import platform
from typing import Any

import torch


def describe() -> dict[str, Any]:
    """Collect the torch/CUDA/device facts worth recording for a run."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }

    for i in range(info["device_count"]):
        props = torch.cuda.get_device_properties(i)
        info["devices"].append(
            {
                "index": i,
                "name": props.name,
                "capability": f"{props.major}.{props.minor}",
                "total_memory_gib": round(props.total_memory / 1024**3, 2),
                "multi_processor_count": props.multi_processor_count,
            }
        )

    return info
