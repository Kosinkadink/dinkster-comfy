from unittest import mock

import torch

import dinkster_inference.model_patcher
import dinkster_inference.sd


def test_model_patcher_can_leave_mapped_offload_weights_unpinned():
    patcher = dinkster_inference.model_patcher.ModelPatcher(
        torch.nn.Linear(1, 1), torch.device("cpu"), torch.device("cpu")
    )
    patcher.pin_offloaded_weights = False

    with mock.patch.object(dinkster_inference.model_patcher, "get_key_weight") as get_weight:
        patcher.pin_weight_to_device("weight")

    get_weight.assert_not_called()


def test_assign_loaded_weights_preserves_mmap_storage(tmp_path):
    weights = tmp_path / "weights.bin"
    weights.write_bytes(torch.arange(4, dtype=torch.float32).numpy().tobytes())
    source = torch.from_file(str(weights), shared=False, size=4).view(2, 2)

    class Model:
        def __init__(self):
            self.diffusion_model = torch.nn.Linear(2, 2, bias=False)

        def load_model_weights(self, state, _prefix, assign=False):
            self.diffusion_model.load_state_dict(state, strict=False, assign=assign)

    class Config:
        supported_inference_dtypes = [torch.float32]
        quant_config = None
        optimizations = {"fp8": False}

        def set_inference_dtype(self, *_args, **_kwargs):
            pass

        def get_model(self, _state, _prefix):
            return Model()

    class Patcher:
        def __init__(self, model, **_kwargs):
            self.model = model
            self.pin_offloaded_weights = True

        def is_dynamic(self):
            return False

    with (
        mock.patch.object(
            dinkster_inference.sd.model_detection,
            "unet_prefix_from_state_dict",
            return_value="",
        ),
        mock.patch.object(
            dinkster_inference.sd.model_detection,
            "model_config_from_unet",
            return_value=Config(),
        ),
        mock.patch.object(
            dinkster_inference.sd.dinkster_inference.utils,
            "convert_old_quants",
            side_effect=lambda state, _prefix, metadata=None: (state, metadata),
        ),
        mock.patch.object(
            dinkster_inference.sd.model_management,
            "get_torch_device",
            return_value=torch.device("cpu"),
        ),
        mock.patch.object(
            dinkster_inference.sd.model_management,
            "unet_offload_device",
            return_value=torch.device("cpu"),
        ),
        mock.patch.object(
            dinkster_inference.sd.model_management,
            "unet_dtype",
            return_value=torch.float32,
        ),
        mock.patch.object(
            dinkster_inference.sd.model_management,
            "unet_manual_cast",
            return_value=None,
        ),
        mock.patch.object(
            dinkster_inference.sd.dinkster_inference.model_patcher,
            "CoreModelPatcher",
            Patcher,
        ),
    ):
        copied = dinkster_inference.sd.load_diffusion_model_state_dict(
            {"weight": source}
        )
        assigned = dinkster_inference.sd.load_diffusion_model_state_dict(
            {"weight": source}, model_options={"assign_loaded_weights": True}
        )

    assert copied.model.diffusion_model.weight.untyped_storage().data_ptr() != (
        source.untyped_storage().data_ptr()
    )
    assert copied.pin_offloaded_weights is True
    assert assigned.model.diffusion_model.weight.untyped_storage().data_ptr() == (
        source.untyped_storage().data_ptr()
    )
    assert assigned.pin_offloaded_weights is False
