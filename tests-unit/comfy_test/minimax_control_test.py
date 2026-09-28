import torch

from dinkster_comfy.cli_args import args

args.cpu = True

from dinkster_comfy.minimax_control import (
    MiniMaxH3FunControlPatch,
    _control_config,
    _video_window_layout,
)
from dinkster_comfy.window_execution import gather_window_tensor
from dinkster_comfy.window_plan import (
    IntegerAffineIndexMap,
    KindAxisMap,
    LayerWindow,
    MediaAxis,
    MergeDeclaration,
    WindowIndexList,
    WindowKind,
    WindowPlanLayer,
    WindowWeightKind,
    WindowWeightProfile,
    compile_window_plan,
)


def _layer(axis, windows):
    return WindowPlanLayer(
        (axis,),
        tuple(LayerWindow((WindowIndexList(window),)) for window in windows),
        (WindowWeightProfile(WindowWeightKind.FLAT),),
        MergeDeclaration(),
    )


def test_control_config_reads_union_v2_metadata_and_tensor_widths():
    state_dict = {
        "control_proj_in.weight": torch.empty(12, 196),
        "control_blocks.0.attn.qkv_proj.weight": torch.empty(24, 4),
        "control_blocks.0.attn.q_norm.weight": torch.empty(2),
        "control_blocks.0.mlp.fc1.weight": torch.empty(16, 4),
    }
    for index in range(10):
        state_dict[f"control_blocks.{index}.after_proj.weight"] = torch.empty(1)

    config = _control_config(
        state_dict,
        {
            "minimax_h3_fun_controlnet": "adaln_basis",
            "inpaint_masked_pixel_mode": "post_norm",
        },
    )

    assert config == {
        "control_in_dim": 49,
        "injection_layers": tuple(range(0, 50, 5)),
        "inpaint_post_norm": True,
        "hidden_size": 12,
        "num_attention_heads": 4,
        "attention_head_dim": 2,
        "ffn_hidden_size": 8,
        "time_embed_dim": 8,
        "use_adaln_curves": True,
    }


def test_control_latent_encodes_full_domain_once_and_gathers_each_joint_window(monkeypatch):
    temporal = MediaAxis("temporal", 3)
    height = MediaAxis("height", 2)
    width = MediaAxis("width", 2)
    plan = compile_window_plan(
        axes=(height, temporal, width),
        kinds=(
            WindowKind(
                "video",
                tuple(
                    KindAxisMap(axis.name, axis.extent, IntegerAffineIndexMap(1))
                    for axis in (height, temporal, width)
                ),
            ),
        ),
        layers=(
            _layer("temporal", ((0, 1), (1, 2))),
            _layer("height", ((0, 1),)),
            _layer("width", ((0, 1),)),
        ),
    )

    class VAE:
        encode_calls = 0

        def spacial_compression_encode(self):
            return 1

        def encode(self, frames):
            del frames
            self.encode_calls += 1
            return torch.full((1, 24, 3, 2, 2), float(self.encode_calls))

    class Control:
        inpaint_post_norm = False
        injection_layers = (0,)

    class ModelPatch:
        model = Control()

    monkeypatch.setattr("dinkster_comfy.minimax_control.model_management.loaded_models", lambda **kwargs: [])
    monkeypatch.setattr("dinkster_comfy.minimax_control.model_management.load_models_gpu", lambda models: None)
    vae = VAE()
    mask = torch.tensor(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )
    patch = MiniMaxH3FunControlPatch(
        ModelPatch(),
        vae,
        torch.zeros((5, 3, 2, 2)),
        mask,
        torch.zeros((5, 3, 2, 2)),
        1.0,
        1.0,
        0.0,
    )
    x = [torch.zeros((1, 24, 2, 2, 2)), torch.zeros((1, 32, 2, 3))]
    gathered = []

    for window in plan.joint_windows:
        options = {
            "sigmas": torch.tensor([0.5]),
            "window_plan": plan,
            "window": window,
        }

        def executor(*call_args, **call_kwargs):
            del call_args, call_kwargs
            gathered.append(patch.window_control_latent.clone())
            return x

        patch.diffusion_model_wrapper(
            executor,
            x,
            torch.tensor([500.0]),
            torch.empty(1),
            options,
        )

    assert vae.encode_calls == 2
    assert patch.control_latent_shape == (1, 24, 3, 2, 2)
    assert patch.control_latent.shape == (1, 49, 3, 2, 2)
    assert torch.equal(patch.control_latent[:, :24], torch.ones((1, 24, 3, 2, 2)))
    assert torch.equal(patch.control_latent[:, 25:], torch.full((1, 24, 3, 2, 2), 2.0))
    assert torch.equal(
        patch.control_latent[:, 24],
        torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]] * 3]),
    )
    layout = _video_window_layout(plan)
    for window, actual in zip(plan.joint_windows, gathered, strict=True):
        assert torch.equal(
            actual,
            gather_window_tensor(patch.control_latent, layout, window),
        )
