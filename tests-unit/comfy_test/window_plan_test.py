"""Windowed-evaluation plan compilation and pure merge tests."""

from __future__ import annotations

import hashlib
import itertools
import json
import struct
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest
from dinkster_inference.window_plan import (
    AccumulationDType,
    CompositeWindowPlan,
    IntegerAffineIndexMap,
    KindAxisMap,
    LayerWindow,
    MediaAxis,
    MergeDeclaration,
    ProportionalRangeIndexMap,
    WindowIndexList,
    WindowKind,
    WindowMergeError,
    WindowPlanLayer,
    WindowPlanRefusal,
    WindowPlanRefusalCode,
    WindowWeightKind,
    WindowWeightProfile,
    compile_window_plan,
    map_window_indices,
    merge_window_outputs,
    window_cache_digest,
)


def _window(*index_lists: tuple[int, ...], modular: bool = False) -> LayerWindow:
    return LayerWindow(tuple(WindowIndexList(indices, modular) for indices in index_lists))


def _layer(
    axes: tuple[str, ...],
    windows: tuple[LayerWindow, ...],
    *,
    weight: WindowWeightKind = WindowWeightKind.FLAT,
    axis_weights: tuple[WindowWeightKind, ...] | None = None,
    overlaps: tuple[int, ...] = (),
    merge: MergeDeclaration | None = None,
    passthrough: tuple[tuple[int, ...], ...] = (),
) -> WindowPlanLayer:
    axis_overlaps = overlaps or (0,) * len(axes)
    if len(axis_overlaps) != len(axes):
        raise ValueError("test layer helper requires one overlap per axis")
    profile_kinds = axis_weights or (weight,) * len(axes)
    if len(profile_kinds) != len(axes):
        raise ValueError("test layer helper requires one weight profile per axis")
    return WindowPlanLayer(
        axes,
        windows,
        tuple(
            WindowWeightProfile(kind, overlap)
            for kind, overlap in zip(profile_kinds, axis_overlaps, strict=True)
        ),
        MergeDeclaration() if merge is None else merge,
        passthrough,
    )


def _full_layer(axis: str, extent: int, **kwargs: Any) -> WindowPlanLayer:
    return _layer((axis,), (_window(tuple(range(extent))),), **kwargs)


def _assert_refusal(
    expected: WindowPlanRefusalCode,
    call: Any,
) -> None:
    with pytest.raises(WindowPlanRefusal) as raised:
        call()
    assert raised.value.code is expected
    assert str(raised.value).startswith(f"{expected.value}:")


def _comfy_proportional_reference(
    indices: tuple[int, ...], primary_extent: int, kind_extent: int
) -> tuple[int, ...]:
    mapped: list[int] = []
    seen: set[int] = set()
    for index in indices:
        start = min(int(round(index * kind_extent / primary_extent)), kind_extent - 1)
        stop = min(int(round((index + 1) * kind_extent / primary_extent)), kind_extent)
        if stop <= start:
            stop = start + 1
        for mapped_index in range(start, stop):
            if mapped_index not in seen:
                seen.add(mapped_index)
                mapped.append(mapped_index)
    return tuple(mapped)


def _float32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


@pytest.mark.parametrize(
    ("indices", "primary_extent", "kind_extent"),
    (
        ((0, 1, 2, 3, 4), 5, 2),
        ((4, 0, 4, 1), 5, 7),
        ((0, 2, 5), 6, 1),
        ((1, 2, 3), 5, 11),
        ((0, 1, 2, 3), 4, 3),
    ),
)
def test_proportional_index_map_matches_comfyui_multimodal_slicing(
    indices: tuple[int, ...], primary_extent: int, kind_extent: int
) -> None:
    assert map_window_indices(
        ProportionalRangeIndexMap(),
        indices,
        primary_extent=primary_extent,
        kind_extent=kind_extent,
    ) == _comfy_proportional_reference(indices, primary_extent, kind_extent)


def test_proportional_index_map_clamps_before_forcing_nonempty_and_deduplicates() -> None:
    assert map_window_indices(
        ProportionalRangeIndexMap(),
        (0, 1, 2, 3, 4),
        primary_extent=5,
        kind_extent=2,
    ) == (0, 1)
    assert map_window_indices(
        ProportionalRangeIndexMap(),
        (4,),
        primary_extent=5,
        kind_extent=2,
    ) == (1,)


def test_integer_affine_index_map_preserves_ordered_repeated_occurrences() -> None:
    assert map_window_indices(
        IntegerAffineIndexMap(2, 1),
        (0, 1, 1),
        primary_extent=3,
        kind_extent=6,
    ) == (1, 3, 3)

    _assert_refusal(
        WindowPlanRefusalCode.INDEX_MAP_OUT_OF_RANGE,
        lambda: map_window_indices(
            IntegerAffineIndexMap(2, 1),
            (2,),
            primary_extent=3,
            kind_extent=5,
        ),
    )


def test_compiler_maps_each_kind_and_requires_mapped_or_invariant_axes() -> None:
    temporal = _full_layer("temporal", 5)
    audio = WindowKind(
        "audio",
        (KindAxisMap("temporal", 2, ProportionalRangeIndexMap()),),
    )
    metadata = WindowKind("metadata", invariant_axes=("temporal",))

    plan = compile_window_plan(
        axes=(MediaAxis("temporal", 5),),
        kinds=(metadata, audio),
        layers=(temporal,),
    )

    assert plan.kinds == (audio, metadata)
    assert plan.joint_windows[0].kind_indices[0].axes == (("temporal", (0, 1)),)
    assert plan.joint_windows[0].kind_indices[1].axes == ()

    missing = WindowKind("missing")
    _assert_refusal(
        WindowPlanRefusalCode.MISSING_AXIS_MAP,
        lambda: compile_window_plan(
            axes=(MediaAxis("temporal", 5),),
            kinds=(missing,),
            layers=(temporal,),
        ),
    )


def test_unknown_index_map_profile_is_a_named_compile_time_refusal() -> None:
    mapping = KindAxisMap("temporal", 3, IntegerAffineIndexMap(1))
    object.__setattr__(mapping, "profile", cast(Any, object()))

    _assert_refusal(
        WindowPlanRefusalCode.UNSUPPORTED_INDEX_MAP,
        lambda: compile_window_plan(
            axes=(MediaAxis("temporal", 3),),
            kinds=(WindowKind("bad", (mapping,)),),
            layers=(_full_layer("temporal", 3),),
        ),
    )


def test_layer_arrival_order_is_non_semantic_and_digest_order_is_canonical() -> None:
    temporal = _layer(
        ("temporal",),
        (_window((0, 1)), _window((1, 2))),
        weight=WindowWeightKind.PYRAMID,
    )
    spatial = _layer(
        ("height", "width"),
        (_window((0, 1), (0, 1)),),
    )
    axes = (MediaAxis("width", 2), MediaAxis("temporal", 3), MediaAxis("height", 2))

    first = compile_window_plan(axes=axes, layers=(temporal, spatial))
    second = compile_window_plan(axes=tuple(reversed(axes)), layers=(spatial, temporal))

    assert first == second
    assert first.digest == second.digest
    assert first.layer_digests == tuple(sorted((temporal.digest, spatial.digest)))
    assert tuple(layer.digest for layer in first.layers) == first.layer_digests
    assert tuple(axis.name for axis in first.axes) == ("height", "temporal", "width")


def test_composite_occurrences_follow_canonical_layer_local_position_order() -> None:
    height = _layer(("height",), (_window((1, 0)),))
    width = _layer(("width",), (_window((1, 0)),))
    plan = compile_window_plan(
        axes=(MediaAxis("height", 2), MediaAxis("width", 2)),
        layers=(height, width),
    )

    assert tuple(
        occurrence.local_positions for occurrence in plan.joint_windows[0].occurrences
    ) == ((0, 0), (0, 1), (1, 0), (1, 1))


def test_disjoint_layers_form_flat_cartesian_joint_windows() -> None:
    temporal = _layer(
        ("temporal",),
        (_window((0, 1)), _window((1, 2))),
    )
    height = _layer(
        ("height",),
        (_window((0,)), _window((1,))),
    )
    plan = compile_window_plan(
        axes=(MediaAxis("temporal", 3), MediaAxis("height", 2)),
        layers=(temporal, height),
    )

    assert len(plan.joint_windows) == 4
    expected: list[tuple[tuple[str, tuple[int, ...]], ...]] = []
    windows_by_digest = {
        temporal.digest: ((("temporal", (0, 1)),), (("temporal", (1, 2)),)),
        height.digest: ((("height", (0,)),), (("height", (1,)),)),
    }
    layer_windows = tuple(windows_by_digest[digest] for digest in plan.layer_digests)
    for components in itertools.product(*layer_windows):
        expected.append(tuple(sorted(itertools.chain.from_iterable(components))))
    assert tuple(window.axis_indices for window in plan.joint_windows) == tuple(expected)


def test_overlapping_axis_claim_is_a_named_refusal() -> None:
    first = _full_layer("temporal", 2)
    second = _layer(("temporal",), (_window((0,)), _window((1,))))

    _assert_refusal(
        WindowPlanRefusalCode.OVERLAPPING_AXIS_CLAIM,
        lambda: compile_window_plan(
            axes=(MediaAxis("temporal", 2),),
            layers=(first, second),
        ),
    )


def test_modular_windows_normalize_indices_and_keep_repeated_occurrences() -> None:
    layer = _layer(
        ("temporal",),
        (
            _window((3, 4, 7), modular=True),
            _window((1, 2)),
        ),
    )
    plan = compile_window_plan(
        axes=(MediaAxis("temporal", 4, wrappable=True),),
        layers=(layer,),
    )

    assert plan.joint_windows[0].axis_indices == (("temporal", (3, 0, 3)),)
    assert tuple(occurrence.coordinate for occurrence in plan.joint_windows[0].occurrences) == (
        (3,),
        (0,),
        (3,),
    )
    assert merge_window_outputs(plan, ((2.0, 4.0, 8.0), (10.0, 12.0))) == (
        4.0,
        10.0,
        12.0,
        5.0,
    )


def test_wrapping_requires_axis_and_merge_capabilities_and_binds_identity() -> None:
    modular = _layer(("temporal",), (_window((0, 1), modular=True),))
    plain = _layer(("temporal",), (_window((0, 1)),))

    _assert_refusal(
        WindowPlanRefusalCode.INVALID_WRAP,
        lambda: compile_window_plan(
            axes=(MediaAxis("temporal", 2),),
            layers=(modular,),
        ),
    )
    no_wrap_merge = _layer(
        ("temporal",),
        (_window((0, 1), modular=True),),
        merge=MergeDeclaration(supports_wrapping=False),
    )
    _assert_refusal(
        WindowPlanRefusalCode.WRAP_MERGE_UNSUPPORTED,
        lambda: compile_window_plan(
            axes=(MediaAxis("temporal", 2, wrappable=True),),
            layers=(no_wrap_merge,),
        ),
    )

    modular_plan = compile_window_plan(
        axes=(MediaAxis("temporal", 2, wrappable=True),),
        layers=(modular,),
    )
    plain_plan = compile_window_plan(
        axes=(MediaAxis("temporal", 2, wrappable=True),),
        layers=(plain,),
    )
    assert modular_plan.joint_windows[0].axis_indices == plain_plan.joint_windows[0].axis_indices
    assert modular.digest != plain.digest
    assert modular_plan.digest != plain_plan.digest


def test_product_weights_and_repeated_indices_merge_in_declared_order() -> None:
    repeated = _layer(
        ("temporal",),
        (_window((0, 0, 1)),),
        weight=WindowWeightKind.PYRAMID,
    )
    plan = compile_window_plan(
        axes=(MediaAxis("temporal", 2),),
        layers=(repeated,),
    )

    assert tuple(occurrence.weight for occurrence in plan.joint_windows[0].occurrences) == (
        1.0,
        2.0,
        1.0,
    )
    assert plan.total_weights == (3.0, 1.0)
    assert merge_window_outputs(plan, ((1.0, 3.0, 5.0),)) == (7.0 / 3.0, 5.0)

    temporal = _layer(
        ("temporal",),
        (_window((0, 1, 2)),),
        weight=WindowWeightKind.PYRAMID,
    )
    height = _layer(
        ("height",),
        (_window((0, 1)),),
        weight=WindowWeightKind.PYRAMID,
    )
    stacked = compile_window_plan(
        axes=(MediaAxis("temporal", 3), MediaAxis("height", 2)),
        layers=(temporal, height),
    )
    assert {
        occurrence.coordinate: occurrence.weight
        for occurrence in stacked.joint_windows[0].occurrences
    } == {
        (0, 0): 1.0,
        (0, 1): 2.0,
        (0, 2): 1.0,
        (1, 0): 1.0,
        (1, 1): 2.0,
        (1, 2): 1.0,
    }


def test_multi_axis_layer_uses_separable_per_axis_profiles() -> None:
    single_layer = _layer(
        ("height", "width"),
        (_window((0, 1, 2), (0, 1)),),
        axis_weights=(WindowWeightKind.PYRAMID, WindowWeightKind.FLAT),
    )
    split_height = _layer(
        ("height",),
        (_window((0, 1, 2)),),
        weight=WindowWeightKind.PYRAMID,
    )
    split_width = _full_layer("width", 2)
    axes = (MediaAxis("height", 3), MediaAxis("width", 2))

    combined = compile_window_plan(axes=axes, layers=(single_layer,))
    split = compile_window_plan(axes=axes, layers=(split_height, split_width))
    combined_weights = {
        occurrence.coordinate: occurrence.weight
        for occurrence in combined.joint_windows[0].occurrences
    }
    split_weights = {
        occurrence.coordinate: occurrence.weight
        for occurrence in split.joint_windows[0].occurrences
    }

    assert combined_weights == split_weights
    assert combined.digest != split.digest


@pytest.mark.parametrize("dtype", (AccumulationDType.FLOAT32, AccumulationDType.FLOAT64))
def test_overlap_linear_preserves_positive_endpoints(dtype: AccumulationDType) -> None:
    layer = _layer(
        ("temporal",),
        (_window((0, 1), modular=True),),
        weight=WindowWeightKind.OVERLAP_LINEAR,
        overlaps=(2,),
        merge=MergeDeclaration(dtype),
    )
    plan = compile_window_plan(
        axes=(MediaAxis("temporal", 2, wrappable=True),),
        layers=(layer,),
    )
    endpoint = _float32(1e-37) if dtype is AccumulationDType.FLOAT32 else 1e-37

    assert tuple(occurrence.weight for occurrence in plan.joint_windows[0].occurrences) == (
        1.0,
        endpoint,
    )
    assert plan.total_weights == (1.0, endpoint)


def test_float32_factors_and_outputs_are_rounded_before_each_operation() -> None:
    merge = MergeDeclaration(AccumulationDType.FLOAT32)
    single_axis = _layer(
        ("temporal",),
        (_window((0, 1, 2, 3), modular=True),),
        weight=WindowWeightKind.OVERLAP_LINEAR,
        overlaps=(4,),
        merge=merge,
    )
    single_plan = compile_window_plan(
        axes=(MediaAxis("temporal", 4, wrappable=True),),
        layers=(single_axis,),
    )
    assert merge_window_outputs(single_plan, ((0.0, 1.0 / 3.0, 0.0, 0.0),))[1] == _float32(
        1.0 / 3.0
    )

    two_axes = _layer(
        ("height", "width"),
        (
            _window((0, 1, 2, 3), (0, 1, 2, 3), modular=True),
            _window((0, 1, 2, 3), (0, 1, 2, 3)),
        ),
        weight=WindowWeightKind.OVERLAP_LINEAR,
        overlaps=(4, 4),
        merge=merge,
    )
    product_plan = compile_window_plan(
        axes=(
            MediaAxis("height", 4, wrappable=True),
            MediaAxis("width", 4, wrappable=True),
        ),
        layers=(two_axes,),
    )
    factor = _float32(2.0 / 3.0)
    occurrence = next(
        occurrence
        for occurrence in product_plan.joint_windows[0].occurrences
        if occurrence.coordinate == (1, 1)
    )

    assert occurrence.weight == _float32(factor * factor)


def test_totality_refuses_uncovered_and_float32_zero_weight_coordinates() -> None:
    _assert_refusal(
        WindowPlanRefusalCode.NON_TOTAL_MERGE,
        lambda: compile_window_plan(
            axes=(MediaAxis("temporal", 3),),
            layers=(_layer(("temporal",), (_window((0, 1)),)),),
        ),
    )

    merge = MergeDeclaration(AccumulationDType.FLOAT32)
    windows = (_window((0, 1)), _window((1, 2)), _window((2, 3)))
    first = _layer(
        ("height",),
        windows,
        weight=WindowWeightKind.OVERLAP_LINEAR,
        overlaps=(2,),
        merge=merge,
    )
    second = _layer(
        ("width",),
        windows,
        weight=WindowWeightKind.OVERLAP_LINEAR,
        overlaps=(2,),
        merge=merge,
    )
    _assert_refusal(
        WindowPlanRefusalCode.NON_TOTAL_MERGE,
        lambda: compile_window_plan(
            axes=(MediaAxis("height", 4), MediaAxis("width", 4)),
            layers=(first, second),
        ),
    )


def test_merge_declaration_conflicts_are_refused() -> None:
    float64_layer = _full_layer("temporal", 2)
    float32_layer = _full_layer(
        "width",
        2,
        merge=MergeDeclaration(AccumulationDType.FLOAT32),
    )

    _assert_refusal(
        WindowPlanRefusalCode.CONFLICTING_MERGE_DECLARATION,
        lambda: compile_window_plan(
            axes=(MediaAxis("temporal", 2), MediaAxis("width", 2)),
            layers=(float64_layer, float32_layer),
        ),
    )


def test_passthrough_claims_lift_once_and_conflicts_are_refused() -> None:
    temporal = _full_layer("temporal", 2, passthrough=((0,),))
    height = _full_layer("height", 2)
    plan = compile_window_plan(
        axes=(MediaAxis("temporal", 2), MediaAxis("height", 2)),
        layers=(temporal, height),
    )

    assert plan.passthrough_coordinates == ((0, 0), (1, 0))

    conflicting_height = _full_layer("height", 2, passthrough=((1,),))
    _assert_refusal(
        WindowPlanRefusalCode.CONFLICTING_STRUCTURAL_CLAIM,
        lambda: compile_window_plan(
            axes=(MediaAxis("temporal", 2), MediaAxis("height", 2)),
            layers=(temporal, conflicting_height),
        ),
    )


def test_mesh_feasibility_visits_every_joint_window_before_refusing() -> None:
    temporal = _layer(
        ("temporal",),
        (_window((0,)), _window((1,))),
    )
    height = _layer(
        ("height",),
        (_window((0,)), _window((1,)), _window((2,))),
    )
    visited: list[int] = []

    def feasible(window: Any) -> bool:
        visited.append(window.index)
        return window.index not in (1, 4)

    _assert_refusal(
        WindowPlanRefusalCode.INFEASIBLE_WINDOW,
        lambda: compile_window_plan(
            axes=(MediaAxis("temporal", 2), MediaAxis("height", 3)),
            layers=(temporal, height),
            mesh_feasible=feasible,
        ),
    )
    assert visited == list(range(6))


def test_nested_or_staged_plan_is_a_named_refusal() -> None:
    layer = _full_layer("temporal", 2)
    plan = compile_window_plan(axes=(MediaAxis("temporal", 2),), layers=(layer,))

    _assert_refusal(
        WindowPlanRefusalCode.NESTED_OR_STAGED_PLAN,
        lambda: compile_window_plan(
            axes=(MediaAxis("temporal", 2),),
            layers=(plan,),
        ),
    )


def test_digest_domains_bind_layer_vector_and_distinguish_layerings() -> None:
    temporal = _full_layer("temporal", 2)
    height = _full_layer("height", 2)
    stacked = compile_window_plan(
        axes=(MediaAxis("temporal", 2), MediaAxis("height", 2)),
        layers=(temporal, height),
    )
    one_layer = _layer(
        ("height", "temporal"),
        (_window((0, 1), (0, 1)),),
    )
    flat = compile_window_plan(
        axes=(MediaAxis("temporal", 2), MediaAxis("height", 2)),
        layers=(one_layer,),
    )

    layer_preimage = json.loads(temporal.canonical_preimage)
    composite_preimage = json.loads(stacked.canonical_preimage)
    assert layer_preimage[0] == "dinkster.window-plan.layer.v1"
    assert temporal.digest == hashlib.sha256(temporal.canonical_preimage.encode()).hexdigest()
    assert composite_preimage[:3] == [
        "dinkster.window-plan.composite.v1",
        2,
        list(stacked.layer_digests),
    ]
    assert stacked.digest == hashlib.sha256(stacked.canonical_preimage.encode()).hexdigest()
    assert stacked.joint_windows[0].axis_indices == flat.joint_windows[0].axis_indices
    assert sorted(
        occurrence.coordinate for occurrence in stacked.joint_windows[0].occurrences
    ) == sorted(occurrence.coordinate for occurrence in flat.joint_windows[0].occurrences)
    assert stacked.digest != flat.digest


def test_cache_digest_binds_joint_window_and_composite_plan_digests() -> None:
    flat_layer = _layer(
        ("temporal",),
        (_window((0,)), _window((1,))),
    )
    pyramid_layer = _layer(
        ("temporal",),
        (_window((0,)), _window((1,))),
        weight=WindowWeightKind.PYRAMID,
    )
    axes = (MediaAxis("temporal", 2),)
    flat = compile_window_plan(axes=axes, layers=(flat_layer,))
    pyramid = compile_window_plan(axes=axes, layers=(pyramid_layer,))

    assert window_cache_digest(flat, 0) != window_cache_digest(flat, 1)
    assert flat.joint_windows[0].digest == pyramid.joint_windows[0].digest
    assert flat.digest != pyramid.digest
    assert window_cache_digest(flat, 0) != window_cache_digest(pyramid, 0)


def test_composite_plan_is_immutable_factory_owned_and_merge_checks_shapes() -> None:
    plan = compile_window_plan(
        axes=(MediaAxis("temporal", 2),),
        layers=(_full_layer("temporal", 2),),
    )

    with pytest.raises(FrozenInstanceError):
        plan.digest = "bad"  # type: ignore[misc]
    with pytest.raises(TypeError, match="produced only"):
        CompositeWindowPlan()
    with pytest.raises(WindowMergeError, match="one value sequence"):
        merge_window_outputs(plan, ())
    with pytest.raises(WindowMergeError, match="requires 2 outputs"):
        merge_window_outputs(plan, ((1.0,),))
    with pytest.raises(WindowMergeError, match="exact ints or floats"):
        merge_window_outputs(plan, ((1.0, cast(Any, "bad")),))
