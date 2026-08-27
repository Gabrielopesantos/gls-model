import torch

from gls.env import describe


def test_describe_reports_expected_keys():
    info = describe()

    assert set(info) == {
        "python",
        "torch",
        "torch_cuda",
        "cuda_available",
        "device_count",
        "devices",
    }
    assert info["torch"].startswith("2.")
    assert info["device_count"] == len(info["devices"])


def test_cuda_is_usable():
    """Skips rather than fails, so the suite stays green on CPU-only hosts."""
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("no CUDA device visible")

    device = describe()["devices"][0]
    assert device["total_memory_gib"] > 0

    x = torch.randn(256, 256, device="cuda")
    assert torch.isfinite(x @ x).all()
