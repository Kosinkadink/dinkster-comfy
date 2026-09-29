import torch

from dinkster_inference.cli_args import args

args.cpu = True

from dinkster_inference.context_windows import (
    ContextFuseMethods,
    ContextSchedules,
    TemporalWindowPlan,
    get_matching_context_schedule,
    get_matching_fuse_method,
)
from dinkster_inference.conds import CONDConstant
from dinkster_inference.contribution_gain import ContributionGain
from dinkster_inference.samplers import compile_window_masks, get_area_and_mult
from dinkster_inference import patcher_extension, utils
from dinkster_inference.window_execution import (
    CompiledWindowField,
    WindowPlanExecutor,
    WindowTensorLayout,
    gather_window_tensor,
    merge_window_tensors,
)
from dinkster_inference.window_plan import (
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
    ProportionalRangeIndexMap,
)
from dinkster_inference.nested_tensor import NestedTensor


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


def test_executor_stacks_spatial_and_temporal_layers_and_gathers_masks_per_window():
    temporal = MediaAxis("temporal", 2)
    height = MediaAxis("height", 2)
    kinds = (
        _kind("effect_mask", (temporal, height)),
        _kind("latent", (temporal, height)),
    )
    plan = compile_window_plan(
        axes=(temporal, height),
        kinds=kinds,
        layers=(
            _layer("temporal", ((0,), (1,))),
            _layer("height", ((0,), (1,))),
        ),
    )
    layout = WindowTensorLayout("latent", (("height", 2), ("temporal", 3)))
    mask = CompiledWindowField(
        torch.tensor([[[[1.0, 0.5], [0.25, 0.0]]]]),
        WindowTensorLayout("effect_mask", (("height", 2), ("temporal", 3))),
    )
    x = torch.arange(4.0).reshape(1, 1, 2, 2)
    seen_masks = []

    def evaluate(model, conds, sub_x, timestep, options):
        del model, timestep
        seen_masks.append(conds[0][0]["mask"].item())
        assert options["transformer_options"]["window"] in plan.joint_windows
        return [sub_x + conds[0][0]["mask"]]

    result = WindowPlanExecutor(plan, layout).execute(
        evaluate,
        object(),
        [[{"mask": mask}]],
        x,
        torch.tensor([1.0]),
        {"unrelated": "preserved"},
    )

    assert sorted(seen_masks) == [0.0, 0.25, 0.5, 1.0]
    assert torch.equal(result[0], x + mask.tensor)


def test_window_execute_wrapper_can_distribute_evaluation_without_changing_merge_order():
    temporal = MediaAxis("temporal", 3)
    plan = compile_window_plan(
        axes=(temporal,),
        kinds=(_kind("latent", (temporal,)),),
        layers=(_layer("temporal", ((0,), (1,), (2,))),),
    )
    window_executor = WindowPlanExecutor(
        plan,
        WindowTensorLayout("latent", (("temporal", 2),)),
    )
    visited = []

    def evaluate(model, conds, sub_x, timestep, options):
        del model, conds, timestep
        index = options["transformer_options"]["window"].index
        visited.append(index)
        return [sub_x + index + 1]

    def reverse_windows(executor, evaluate, model, conds, template, timestep, options, packed):
        outputs = [[None] * len(executor.class_obj.plan.joint_windows) for _ in conds]
        for index in reversed(range(len(executor.class_obj.plan.joint_windows))):
            values = executor.class_obj.evaluate_window(
                index, evaluate, model, conds, template, timestep, options, packed
            )
            for condition_index, value in enumerate(values):
                outputs[condition_index][index] = value
        return outputs

    model_options = {}
    patcher_extension.add_wrapper(
        patcher_extension.WrappersMP.WINDOW_EXECUTE,
        reverse_windows,
        model_options,
        is_model_options=True,
    )
    result = window_executor.execute(
        evaluate,
        object(),
        [[{}]],
        torch.zeros((1, 1, 3)),
        torch.tensor([1.0]),
        model_options,
    )

    assert visited == [2, 1, 0]
    assert torch.equal(result[0], torch.tensor([[[1.0, 2.0, 3.0]]]))


def test_conditioning_and_control_effect_masks_compile_once_and_gather_per_window():
    temporal = MediaAxis("temporal", 2)
    height = MediaAxis("height", 2)
    plan = compile_window_plan(
        axes=(temporal, height),
        kinds=(_kind("latent", (temporal, height)),),
        layers=(
            _layer("temporal", ((0,), (1,))),
            _layer("height", ((0,), (1,))),
        ),
    )
    executor = WindowPlanExecutor(
        plan,
        WindowTensorLayout("latent", (("height", 2), ("temporal", 3))),
    )
    mask = torch.tensor([[[1.0, 0.5], [0.25, 0.0]]])
    control_mask = torch.tensor([[[0.2, 0.4], [0.6, 0.8]]])
    control = object()
    gain = ContributionGain(effect_masks=(mask, control_mask)).realize(
        torch.tensor([1.0, 0.0])
    )
    conds = [[
        {
            "control": control,
            "mask": mask,
            "model_conds": {},
            "realized_contribution_gain": gain,
            "uuid": "masked-control",
        }
    ]]
    compile_window_masks({"positive": conds[0]}, executor)
    assert type(conds[0][0]["mask"]) is CompiledWindowField
    assert all(
        type(value) is CompiledWindowField
        for value in conds[0][0]["window_effect_masks"]
    )
    x = torch.zeros((1, 1, 2, 2))
    seen = []

    def evaluate(model, sub_conds, sub_x, timestep, options):
        del model, options
        prepared = get_area_and_mult(sub_conds[0][0], sub_x, timestep)
        assert prepared.control is control
        seen.append(prepared.mult.item())
        return [prepared.mult]

    result = executor.execute(
        evaluate,
        object(),
        conds,
        x,
        torch.tensor([1.0]),
        {},
    )[0]

    assert torch.allclose(torch.tensor(sorted(seen)), torch.tensor([0.0, 0.15, 0.2, 0.2]))
    assert torch.allclose(result, (mask * control_mask).unsqueeze(1))


def test_temporal_adapter_compiles_stock_context_schedule_without_raw_dimension_claims():
    temporal = TemporalWindowPlan(
        get_matching_context_schedule(ContextSchedules.UNIFORM_LOOPED),
        get_matching_fuse_method(ContextFuseMethods.PYRAMID),
        context_length=4,
        context_overlap=1,
        context_stride=2,
        closed_loop=True,
        _step=3,
    )
    expected = temporal.context_schedule.func(6, temporal, {})

    executor = temporal.executor(extent=6, latent_dimension=2, model_options={})

    actual = [dict(window.axis_indices)["temporal"] for window in executor.plan.joint_windows]
    assert actual == [tuple(window) for window in expected]
    assert executor.latent_layout == WindowTensorLayout("latent", (("temporal", 2),))


def test_temporal_adapter_stacks_with_a_spatial_layer():
    temporal = TemporalWindowPlan(
        get_matching_context_schedule(ContextSchedules.STATIC_STANDARD),
        get_matching_fuse_method(ContextFuseMethods.FLAT),
        context_length=2,
    )
    height = MediaAxis("height", 2)
    kinds = (_kind("latent", (height, MediaAxis("temporal", 2))),)

    executor = temporal.executor(
        extent=2,
        latent_dimension=3,
        model_options={},
        axes=(height,),
        layers=(_layer("height", ((0,), (1,))),),
        kinds=kinds,
        latent_axis_dimensions=(("height", 2),),
    )

    assert len(executor.plan.joint_windows) == 2
    assert executor.latent_layout == WindowTensorLayout(
        "latent", (("height", 2), ("temporal", 3))
    )


def test_executor_maps_asymmetric_nested_video_and_audio_streams():
    temporal = MediaAxis("temporal", 3)
    layer = _layer("temporal", ((0, 1), (1, 2)))
    kinds = (
        WindowKind(
            "audio",
            (KindAxisMap("temporal", 5, ProportionalRangeIndexMap()),),
        ),
        _kind("video", (temporal,)),
    )
    plan = compile_window_plan(axes=(temporal,), kinds=kinds, layers=(layer,))
    executor = WindowPlanExecutor(
        plan,
        (
            WindowTensorLayout("video", (("temporal", 2),)),
            WindowTensorLayout("audio", (("temporal", 3),)),
        ),
    )
    video = torch.tensor([[[1.0, 2.0, 3.0]]])
    audio = torch.tensor([[[[10.0, 20.0, 30.0, 40.0, 50.0]]]])
    latent = NestedTensor((video, audio))
    seen_shapes = []

    def evaluate(model, conds, sub_x, timestep, options):
        del model, conds, timestep, options
        seen_shapes.append(tuple(tuple(stream.shape) for stream in sub_x.unbind()))
        return [sub_x + 100.0 * len(seen_shapes)]

    result = executor.execute(evaluate, object(), [[]], latent, torch.tensor([1.0]), {})[0]

    assert seen_shapes == [((1, 1, 2), (1, 1, 1, 3)), ((1, 1, 2), (1, 1, 1, 3))]
    assert torch.equal(result.unbind()[0], torch.tensor([[[101.0, 152.0, 203.0]]]))
    assert torch.equal(
        result.unbind()[1],
        torch.tensor([[[[110.0, 120.0, 180.0, 240.0, 250.0]]]]),
    )


def test_executor_unpacks_windows_and_repacks_sampler_latents():
    temporal = MediaAxis("temporal", 3)
    plan = compile_window_plan(
        axes=(temporal,),
        kinds=(
            WindowKind(
                "audio",
                (KindAxisMap("temporal", 5, ProportionalRangeIndexMap()),),
            ),
            _kind("video", (temporal,)),
        ),
        layers=(_layer("temporal", ((0, 1), (1, 2))),),
    )
    executor = WindowPlanExecutor(
        plan,
        (
            WindowTensorLayout("video", (("temporal", 2),)),
            WindowTensorLayout("audio", (("temporal", 3),)),
        ),
    )
    video = torch.tensor([[[1.0, 2.0, 3.0]]])
    audio = torch.tensor([[[[10.0, 20.0, 30.0, 40.0, 50.0]]]])
    packed, shapes = utils.pack_latents((video, audio))

    class Model:
        latent_shapes = shapes

    conds = [[{"model_conds": {"latent_shapes": CONDConstant(shapes)}}]]
    seen_shapes = []

    def evaluate(model, sub_conds, sub_x, timestep, options):
        del model, timestep, options
        sub_shapes = sub_conds[0][0]["model_conds"]["latent_shapes"].cond
        seen_shapes.append(sub_shapes)
        assert sub_x.shape == (1, 1, sum(torch.Size(shape[1:]).numel() for shape in sub_shapes))
        return [sub_x + 100.0 * len(seen_shapes)]

    result = executor.execute(evaluate, Model(), conds, packed, torch.tensor([1.0]), {})[0]
    video_result, audio_result = utils.unpack_latents(result, shapes)

    assert seen_shapes == [
        [torch.Size((1, 1, 2)), torch.Size((1, 1, 1, 3))],
        [torch.Size((1, 1, 2)), torch.Size((1, 1, 1, 3))],
    ]
    assert torch.equal(video_result, torch.tensor([[[101.0, 152.0, 203.0]]]))
    assert torch.equal(
        audio_result,
        torch.tensor([[[[110.0, 120.0, 180.0, 240.0, 250.0]]]]),
    )


def test_executor_packs_primary_effect_mask_for_multistream_conditioning():
    temporal = MediaAxis("temporal", 3)
    plan = compile_window_plan(
        axes=(temporal,),
        kinds=(
            WindowKind(
                "audio",
                (KindAxisMap("temporal", 5, ProportionalRangeIndexMap()),),
            ),
            _kind("video", (temporal,)),
        ),
        layers=(_layer("temporal", ((0, 1), (1, 2))),),
    )
    executor = WindowPlanExecutor(
        plan,
        (
            WindowTensorLayout("video", (("temporal", 2),)),
            WindowTensorLayout("audio", (("temporal", 3),)),
        ),
    )
    video = torch.zeros((1, 2, 3))
    audio = torch.zeros((1, 3, 1, 5))
    packed, shapes = utils.pack_latents((video, audio))

    class Model:
        latent_shapes = shapes

    mask = torch.tensor([[0.25, 0.5, 0.75]])
    conds = [[
        {
            "mask": executor.compile_mask(mask),
            "model_conds": {"latent_shapes": CONDConstant(shapes)},
            "uuid": "masked",
        }
    ]]
    seen = []

    def evaluate(model, sub_conds, sub_x, timestep, options):
        del model, options
        prepared = get_area_and_mult(sub_conds[0][0], sub_x, timestep)
        seen.append(prepared.mult)
        return [prepared.mult]

    executor.execute(evaluate, Model(), conds, packed, torch.tensor([1.0]), {})

    first_video, first_audio = utils.unpack_latents(seen[0], [
        torch.Size((1, 2, 2)),
        torch.Size((1, 3, 1, 3)),
    ])
    assert torch.equal(first_video, torch.tensor([[[0.25, 0.5], [0.25, 0.5]]]))
    assert torch.equal(first_audio, torch.ones((1, 3, 1, 3)))


def test_executor_merges_nested_streams_over_invariant_spatial_axes():
    temporal = MediaAxis("temporal", 2)
    height = MediaAxis("height", 2)
    plan = compile_window_plan(
        axes=(height, temporal),
        kinds=(
            WindowKind(
                "audio",
                (KindAxisMap("temporal", 2, IntegerAffineIndexMap(1)),),
                ("height",),
            ),
            _kind("video", (height, temporal)),
        ),
        layers=(
            _layer("temporal", ((0, 1),)),
            _layer("height", ((0,), (1,))),
        ),
    )
    executor = WindowPlanExecutor(
        plan,
        (
            WindowTensorLayout("video", (("height", 2), ("temporal", 3))),
            WindowTensorLayout("audio", (("temporal", 3),)),
        ),
    )
    video = torch.zeros((1, 1, 2, 2))
    audio = torch.zeros((1, 1, 1, 2))
    latent = NestedTensor((video, audio))
    calls = 0

    def evaluate(model, conds, sub_x, timestep, options):
        nonlocal calls
        del model, conds, timestep, options
        calls += 1
        return [sub_x + 100.0 * calls]

    result = executor.execute(evaluate, object(), [[]], latent, torch.tensor([1.0]), {})[0]

    assert torch.equal(
        result.unbind()[0],
        torch.tensor([[[[100.0, 100.0], [200.0, 200.0]]]]),
    )
    assert torch.equal(result.unbind()[1], torch.tensor([[[[150.0, 150.0]]]]))
