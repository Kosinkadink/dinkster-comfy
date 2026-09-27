import torch

from dinkster_comfy.window_execution import (
    CompiledWindowField,
    WindowTensorLayout,
    gather_window_tensor,
    merge_window_tensors,
)
from dinkster_comfy.window_plan import (
    LayerWindow,
    KindAxisMap,
    MediaAxis,
    MergeDeclaration,
    WindowIndexList,
    WindowKind,
    WindowPlanLayer,
    WindowWeightKind,
    WindowWeightProfile,
    compile_window_plan,
    IntegerAffineIndexMap,
)


def _layer(axis, windows):
    return WindowPlanLayer(
        (axis,),
        tuple(LayerWindow((WindowIndexList(window),)) for window in windows),
        (WindowWeightProfile(WindowWeightKind.FLAT),),
        MergeDeclaration(),
    )


def _kind(name, axes):
    return WindowKind(
        name,
        tuple(KindAxisMap(axis.name, axis.extent, IntegerAffineIndexMap(1)) for axis in sorted(axes, key=lambda item: item.name)),
    )


def test_tensor_gather_uses_declared_media_axes_instead_of_dimension_order():
    temporal = MediaAxis("temporal", 3)
    width = MediaAxis("width", 4)
    plan = compile_window_plan(
        axes=(temporal, width),
        kinds=(_kind("latent", (temporal, width)),),
        layers=(_layer("temporal", ((2, 0), (1,))), _layer("width", ((3, 1), (0, 2)))),
    )
    tensor = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    layout = WindowTensorLayout("latent", (("temporal", 2), ("width", 1)))

    gathered = gather_window_tensor(tensor, layout, plan.joint_windows[0])

    indices = dict(plan.joint_windows[0].axis_indices)
    expected = tensor.index_select(2, torch.tensor(indices["temporal"])).index_select(
        1, torch.tensor(indices["width"])
    )
    assert torch.equal(gathered, expected)


def test_full_domain_window_round_trips_byte_identically():
    temporal = MediaAxis("temporal", 3)
    plan = compile_window_plan(
        axes=(temporal,),
        kinds=(_kind("latent", (temporal,)),),
        layers=(_layer("temporal", ((0, 1, 2),)),),
    )
    tensor = torch.tensor([[[1.25, -2.5, 7.0]]], dtype=torch.float16)
    layout = WindowTensorLayout("latent", (("temporal", 2),))

    gathered = gather_window_tensor(tensor, layout, plan.joint_windows[0])
    merged = merge_window_tensors(plan, layout, (gathered,), tuple(tensor.shape))

    assert torch.equal(merged, tensor)
    assert merged.numpy().tobytes() == tensor.numpy().tobytes()


def test_repeated_occurrences_accumulate_outputs_and_full_domain_mask_gathers_once():
    temporal = MediaAxis("temporal", 3, wrappable=True)
    layer = WindowPlanLayer(
        ("temporal",),
        (
            LayerWindow((WindowIndexList((0, 3, 0), modular=True),)),
            LayerWindow((WindowIndexList((1, 2),),)),
        ),
        (WindowWeightProfile(WindowWeightKind.FLAT),),
        MergeDeclaration(),
    )
    kinds = (_kind("latent", (temporal,)), _kind("effect_mask", (temporal,)))
    plan = compile_window_plan(axes=(temporal,), kinds=kinds, layers=(layer,))
    latent_layout = WindowTensorLayout("latent", (("temporal", 2),))
    mask = CompiledWindowField(
        torch.tensor([[[0.25, 0.5, 1.0]]]),
        WindowTensorLayout("effect_mask", (("temporal", 2),)),
    )

    assert mask.gather(plan.joint_windows[0]).tolist() == [[[0.25, 0.25, 0.25]]]
    first = torch.tensor([[[2.0, 4.0, 8.0]]])
    second = torch.tensor([[[10.0, 12.0]]])
    merged = merge_window_tensors(plan, latent_layout, (first, second), (1, 1, 3))

    assert torch.equal(merged, torch.tensor([[[14.0 / 3.0, 10.0, 12.0]]]))
