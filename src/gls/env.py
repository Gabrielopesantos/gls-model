"""Environment introspection.

Backs the `gpu-check` script, and is meant to be called at the start of every
training or serving run so the exact torch/CUDA/device combination ends up in
the run log alongside the loss curve.
"""

from __future__ import annotations

import platform
import sys
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


def main() -> int:
    info = describe()

    for key in ("python", "torch", "torch_cuda", "cuda_available", "device_count"):
        print(f"{key}: {info[key]}")

    for device in info["devices"]:
        print(
            f"  [{device['index']}] {device['name']} "
            f"sm_{device['capability'].replace('.', '')} "
            f"{device['total_memory_gib']} GiB "
            f"{device['multi_processor_count']} SMs"
        )

    if not info["cuda_available"]:
        print("cuda unavailable, CPU only", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
