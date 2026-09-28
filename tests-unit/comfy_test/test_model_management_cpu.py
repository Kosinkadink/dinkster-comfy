import pytest
import torch

from dinkster_inference import model_management


def test_cpu_only_runtime_uses_cpu():
    if (
        torch.cuda.is_available()
        or torch.backends.mps.is_available()
        or model_management.xpu_available
        or model_management.npu_available
        or model_management.mlu_available
        or model_management.ixuca_available
    ):
        pytest.skip("runtime has an accelerator")

    assert model_management.get_torch_device() == torch.device("cpu")
