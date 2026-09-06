"""Environment introspection.

Backs ``gls env``, and is called at the start of every training or serving run
so the exact torch/CUDA/device combination ends up in the run log alongside the
loss curve.
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


# Dense bf16 tensor-core peak (FP32 accumulate), FLOP/s, keyed on
# ``torch.cuda.get_device_name``. Only the datacenter cards a real training run
# lands on are listed - a card not here logs no ``train/mfu`` rather than a wrong
# one. Consumer Ada parts are deliberately absent: their marketed tensor figure
# is 2:4-sparse, and the dense bf16 number is ambiguous enough that a guess would
# defeat the point of the metric.
_PEAK_BF16_FLOPS: dict[str, float] = {
    "NVIDIA A100-SXM4-40GB": 312e12,
    "NVIDIA A100-SXM4-80GB": 312e12,
    "NVIDIA A100 80GB PCIe": 312e12,
    "NVIDIA H100 80GB HBM3": 989e12,
    "NVIDIA H100 PCIe": 756e12,
    "NVIDIA L40S": 362e12,
}


def peak_bf16_flops(device_name: str | None = None) -> float | None:
    """Known dense bf16 peak for ``device_name`` (default: the first visible
    CUDA device), or ``None`` when the card is unlisted or there is no CUDA."""
    if device_name is None:
        if not torch.cuda.is_available():
            return None
        device_name = torch.cuda.get_device_name(0)
    return _PEAK_BF16_FLOPS.get(device_name)
