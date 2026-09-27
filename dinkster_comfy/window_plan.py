"""Torch-free compilation and deterministic merge for windowed evaluation."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias, cast

__all__ = [
    "AccumulationDType",
    "CompositeWindowPlan",
    "IntegerAffineIndexMap",
    "JointWindow",
    "KindAxisMap",
    "KindWindowIndices",
    "LayerWindow",
    "MediaAxis",
    "MergeDeclaration",
    "ProportionalRangeIndexMap",
    "WindowIndexList",
    "WindowKind",
    "WindowMergeError",
    "WindowOccurrence",
    "WindowPlanLayer",
    "WindowPlanRefusal",
    "WindowPlanRefusalCode",
    "WindowWeightKind",
    "WindowWeightProfile",
    "compile_window_plan",
    "map_window_indices",
    "merge_window_outputs",
    "window_cache_digest",
]

_LAYER_DOMAIN = "dinkster.window-plan.layer.v1"
_COMPOSITE_DOMAIN = "dinkster.window-plan.composite.v1"
_JOINT_WINDOW_DOMAIN = "dinkster.window-plan.joint-window.v1"
_CACHE_DOMAIN = "dinkster.window-plan.cache.v1"
_MERGE_ALGORITHM = "accumulate-normalize.v1"
_AFFINE_PROFILE = "integer-affine.v1"
_PROPORTIONAL_PROFILE = "proportional-range-nearest-even-clamp-nonempty-dedup.v1"


def _canonical_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("window-plan values must be finite")
    return format(value, ".17g")


class WindowPlanRefusalCode(StrEnum):
    INVALID_PLAN = "invalid-plan"
    UNSUPPORTED_INDEX_MAP = "unsupported-index-map"
    INDEX_MAP_OUT_OF_RANGE = "index-map-out-of-range"
    MISSING_AXIS_MAP = "missing-axis-map"
    INVALID_WRAP = "invalid-wrap"
    INDEX_OUT_OF_RANGE = "index-out-of-range"
    OVERLAPPING_AXIS_CLAIM = "overlapping-axis-claim"
    CONFLICTING_MERGE_DECLARATION = "conflicting-merge-declaration"
    CONFLICTING_STRUCTURAL_CLAIM = "conflicting-structural-claim"
    NESTED_OR_STAGED_PLAN = "nested-or-staged-plan"
    WRAP_MERGE_UNSUPPORTED = "wrap-merge-unsupported"
    NON_TOTAL_MERGE = "non-total-merge"
    INFEASIBLE_WINDOW = "infeasible-window"


class WindowPlanRefusal(ValueError):
    def __init__(self, code: WindowPlanRefusalCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


class WindowMergeError(ValueError):
    pass


def _nonempty_string(name: str, value: str) -> None:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty exact string")


def _positive_int(name: str, value: int) -> None:
    if type(value) is not int or value < 1:
        raise TypeError(f"{name} must be an exact int >= 1")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(preimage: str) -> str:
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class MediaAxis:
    name: str
    extent: int
    wrappable: bool = False

    def __post_init__(self) -> None:
        _nonempty_string("axis name", self.name)
        _positive_int("axis extent", self.extent)
        if type(self.wrappable) is not bool:
            raise TypeError("wrappable must be an exact bool")


@dataclass(frozen=True, slots=True)
class IntegerAffineIndexMap:
    scale: int
    offset: int = 0

    def __post_init__(self) -> None:
        if type(self.scale) is not int or type(self.offset) is not int:
            raise TypeError("affine scale and offset must be exact ints")

    @property
    def identity(self) -> str:
        return _AFFINE_PROFILE


@dataclass(frozen=True, slots=True)
class ProportionalRangeIndexMap:
    @property
    def identity(self) -> str:
        return _PROPORTIONAL_PROFILE


IndexMapProfile: TypeAlias = IntegerAffineIndexMap | ProportionalRangeIndexMap


@dataclass(frozen=True, slots=True)
class KindAxisMap:
    axis: str
    extent: int
    profile: IndexMapProfile

    def __post_init__(self) -> None:
        _nonempty_string("kind axis map axis", self.axis)
        _positive_int("kind axis map extent", self.extent)


@dataclass(frozen=True, slots=True)
class WindowKind:
    name: str
    axis_maps: tuple[KindAxisMap, ...] = ()
    invariant_axes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty_string("window kind name", self.name)
        if type(self.axis_maps) is not tuple or any(
            type(mapping) is not KindAxisMap for mapping in self.axis_maps
        ):
            raise TypeError("axis_maps must be a tuple of exact KindAxisMap values")
        map_axes = tuple(mapping.axis for mapping in self.axis_maps)
        if map_axes != tuple(sorted(map_axes)) or len(map_axes) != len(set(map_axes)):
            raise ValueError("kind axis maps must be sorted by axis and unique")
        if type(self.invariant_axes) is not tuple or any(
            type(axis) is not str or not axis for axis in self.invariant_axes
        ):
            raise TypeError("invariant_axes must be a tuple of non-empty exact strings")
        if self.invariant_axes != tuple(sorted(self.invariant_axes)) or len(
            self.invariant_axes
        ) != len(set(self.invariant_axes)):
            raise ValueError("invariant_axes must be sorted and unique")
        if set(map_axes) & set(self.invariant_axes):
            raise ValueError("a kind axis cannot be both mapped and invariant")


@dataclass(frozen=True, slots=True)
class WindowIndexList:
    indices: tuple[int, ...]
    modular: bool = False

    def __post_init__(self) -> None:
        if type(self.indices) is not tuple or any(type(index) is not int for index in self.indices):
            raise TypeError("window indices must be a tuple of exact ints")
        if not self.indices:
            raise ValueError("window index lists must not be empty")
        if type(self.modular) is not bool:
            raise TypeError("window modular flag must be an exact bool")


@dataclass(frozen=True, slots=True)
class LayerWindow:
    index_lists: tuple[WindowIndexList, ...]

    def __post_init__(self) -> None:
        if type(self.index_lists) is not tuple or any(
            type(index_list) is not WindowIndexList for index_list in self.index_lists
        ):
            raise TypeError("index_lists must be a tuple of exact WindowIndexList values")
        if not self.index_lists:
            raise ValueError("a layer window must claim at least one axis")


class WindowWeightKind(StrEnum):
    FLAT = "flat.v1"
    PYRAMID = "pyramid.v1"
    OVERLAP_LINEAR = "overlap-linear.v1"


@dataclass(frozen=True, slots=True)
class WindowWeightProfile:
    kind: WindowWeightKind
    overlap: int = 0

    def __post_init__(self) -> None:
        if type(self.kind) is not WindowWeightKind:
            raise TypeError("weight kind must be an exact WindowWeightKind")
        if type(self.overlap) is not int or self.overlap < 0:
            raise TypeError("weight overlap must be an exact int >= 0")
        if self.kind is not WindowWeightKind.OVERLAP_LINEAR and self.overlap:
            raise ValueError("only overlap-linear weights accept an overlap parameter")


class AccumulationDType(StrEnum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"


@dataclass(frozen=True, slots=True)
class MergeDeclaration:
    accumulation_dtype: AccumulationDType = AccumulationDType.FLOAT64
    supports_wrapping: bool = True
    algorithm: str = _MERGE_ALGORITHM

    def __post_init__(self) -> None:
        if type(self.accumulation_dtype) is not AccumulationDType:
            raise TypeError("accumulation_dtype must be an exact AccumulationDType")
        if type(self.supports_wrapping) is not bool:
            raise TypeError("supports_wrapping must be an exact bool")
        if self.algorithm != _MERGE_ALGORITHM:
            raise ValueError(f"merge algorithm must be {_MERGE_ALGORITHM!r}")


@dataclass(frozen=True, slots=True)
class WindowPlanLayer:
    axes: tuple[str, ...]
    windows: tuple[LayerWindow, ...]
    weight_profiles: tuple[WindowWeightProfile, ...]
    merge: MergeDeclaration
    passthrough_coordinates: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        if type(self.axes) is not tuple or any(
            type(axis) is not str or not axis for axis in self.axes
        ):
            raise TypeError("layer axes must be a tuple of non-empty exact strings")
        if not self.axes:
            raise ValueError("a window-plan layer must claim at least one axis")
        if self.axes != tuple(sorted(self.axes)) or len(self.axes) != len(set(self.axes)):
            raise ValueError("layer axes must be sorted and unique")
        if type(self.windows) is not tuple or any(
            type(window) is not LayerWindow for window in self.windows
        ):
            raise TypeError("windows must be a tuple of exact LayerWindow values")
        if not self.windows:
            raise ValueError("a window-plan layer must contain at least one window")
        if any(len(window.index_lists) != len(self.axes) for window in self.windows):
            raise ValueError("every layer window must contain one index list per layer axis")
        if type(self.weight_profiles) is not tuple or any(
            type(profile) is not WindowWeightProfile for profile in self.weight_profiles
        ):
            raise TypeError("weight_profiles must be a tuple of exact WindowWeightProfile values")
        if len(self.weight_profiles) != len(self.axes):
            raise ValueError("a layer requires one weight profile per claimed axis")
        if type(self.merge) is not MergeDeclaration:
            raise TypeError("merge must be an exact MergeDeclaration")
        if type(self.passthrough_coordinates) is not tuple or any(
            type(coordinate) is not tuple
            or len(coordinate) != len(self.axes)
            or any(type(index) is not int for index in coordinate)
            for coordinate in self.passthrough_coordinates
        ):
            raise TypeError("passthrough coordinates must be tuples of exact axis indices")
        if len(self.passthrough_coordinates) != len(set(self.passthrough_coordinates)):
            raise ValueError("passthrough coordinates must be unique")

    def _facts(self) -> dict[str, object]:
        return {
            "axes": self.axes,
            "merge": {
                "accumulation_dtype": self.merge.accumulation_dtype.value,
                "algorithm": self.merge.algorithm,
                "supports_wrapping": self.merge.supports_wrapping,
            },
            "passthrough_coordinates": self.passthrough_coordinates,
            "weight_profiles": tuple(
                {"identity": profile.kind.value, "overlap": profile.overlap}
                for profile in self.weight_profiles
            ),
            "windows": tuple(
                tuple(
                    {"indices": index_list.indices, "modular": index_list.modular}
                    for index_list in window.index_lists
                )
                for window in self.windows
            ),
        }

    @property
    def canonical_preimage(self) -> str:
        return _canonical_json((_LAYER_DOMAIN, self._facts()))

    @property
    def digest(self) -> str:
        return _digest(self.canonical_preimage)


@dataclass(frozen=True, slots=True)
class KindWindowIndices:
    kind: str
    axes: tuple[tuple[str, tuple[int, ...]], ...]


@dataclass(frozen=True, slots=True)
class WindowOccurrence:
    coordinate: tuple[int, ...]
    local_positions: tuple[int, ...]
    weight: float


@dataclass(frozen=True, slots=True)
class JointWindow:
    index: int
    axis_indices: tuple[tuple[str, tuple[int, ...]], ...]
    kind_indices: tuple[KindWindowIndices, ...]
    occurrences: tuple[WindowOccurrence, ...]
    digest: str


@dataclass(frozen=True, slots=True, init=False)
class CompositeWindowPlan:
    axes: tuple[MediaAxis, ...]
    kinds: tuple[WindowKind, ...]
    layers: tuple[WindowPlanLayer, ...]
    layer_digests: tuple[str, ...]
    joint_windows: tuple[JointWindow, ...]
    merge: MergeDeclaration
    coordinates: tuple[tuple[int, ...], ...]
    total_weights: tuple[float, ...]
    passthrough_coordinates: tuple[tuple[int, ...], ...]
    canonical_preimage: str
    digest: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("composite window plans are produced only by compile_window_plan")


@dataclass(frozen=True, slots=True)
class _LayerOccurrence:
    coordinate: tuple[int, ...]
    local_positions: tuple[int, ...]
    weight: float


@dataclass(frozen=True, slots=True)
class _CompiledLayerWindow:
    index_lists: tuple[tuple[int, ...], ...]
    modular_axes: tuple[bool, ...]
    occurrences: tuple[_LayerOccurrence, ...]


@dataclass(frozen=True, slots=True)
class _CompiledLayer:
    declaration: WindowPlanLayer
    windows: tuple[_CompiledLayerWindow, ...]


def _index_profile_facts(profile: IndexMapProfile) -> dict[str, object]:
    if type(profile) is IntegerAffineIndexMap:
        return {
            "identity": profile.identity,
            "offset": profile.offset,
            "scale": profile.scale,
        }
    if type(profile) is ProportionalRangeIndexMap:
        return {"identity": profile.identity}
    raise WindowPlanRefusal(
        WindowPlanRefusalCode.UNSUPPORTED_INDEX_MAP,
        f"profile {type(profile).__name__} is not in the declared vocabulary",
    )


def map_window_indices(
    profile: IndexMapProfile,
    indices: tuple[int, ...],
    *,
    primary_extent: int,
    kind_extent: int,
) -> tuple[int, ...]:
    """Map one primary-axis index list through a declared profile."""

    if type(indices) is not tuple or any(type(index) is not int for index in indices):
        raise TypeError("indices must be a tuple of exact ints")
    _positive_int("primary_extent", primary_extent)
    _positive_int("kind_extent", kind_extent)
    if any(not 0 <= index < primary_extent for index in indices):
        raise WindowPlanRefusal(
            WindowPlanRefusalCode.INDEX_MAP_OUT_OF_RANGE,
            "primary index is outside its declared extent",
        )
    if type(profile) is IntegerAffineIndexMap:
        mapped = tuple(index * profile.scale + profile.offset for index in indices)
        if any(not 0 <= index < kind_extent for index in mapped):
            raise WindowPlanRefusal(
                WindowPlanRefusalCode.INDEX_MAP_OUT_OF_RANGE,
                "integer-affine mapping names an index outside the kind extent",
            )
        return mapped
    if type(profile) is ProportionalRangeIndexMap:
        mapped_list: list[int] = []
        seen: set[int] = set()
        for index in indices:
            start = round(index * kind_extent / primary_extent)
            stop = round((index + 1) * kind_extent / primary_extent)
            start = min(max(start, 0), kind_extent - 1)
            stop = min(max(stop, 0), kind_extent)
            if stop <= start:
                stop = start + 1
            for mapped_index in range(start, stop):
                if mapped_index not in seen:
                    seen.add(mapped_index)
                    mapped_list.append(mapped_index)
        return tuple(mapped_list)
    raise WindowPlanRefusal(
        WindowPlanRefusalCode.UNSUPPORTED_INDEX_MAP,
        f"profile {type(profile).__name__} is not in the declared vocabulary",
    )


def _cast_accumulation(value: float, dtype: AccumulationDType) -> float:
    if dtype is AccumulationDType.FLOAT64:
        return float(value)
    try:
        return struct.unpack("!f", struct.pack("!f", float(value)))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def _add(left: float, right: float, dtype: AccumulationDType) -> float:
    left = _cast_accumulation(left, dtype)
    right = _cast_accumulation(right, dtype)
    return _cast_accumulation(left + right, dtype)


def _multiply(left: float, right: float, dtype: AccumulationDType) -> float:
    left = _cast_accumulation(left, dtype)
    right = _cast_accumulation(right, dtype)
    return _cast_accumulation(left * right, dtype)


def _divide(numerator: float, denominator: float, dtype: AccumulationDType) -> float:
    numerator = _cast_accumulation(numerator, dtype)
    denominator = _cast_accumulation(denominator, dtype)
    return _cast_accumulation(numerator / denominator, dtype)


def _linspace(
    start: float,
    stop: float,
    count: int,
    dtype: AccumulationDType,
) -> tuple[float, ...]:
    if count == 0:
        return ()
    start = _cast_accumulation(start, dtype)
    stop = _cast_accumulation(stop, dtype)
    if count == 1:
        return (start,)
    last = count - 1
    return tuple(
        start
        if index == 0
        else stop
        if index == last
        else _cast_accumulation((start * (last - index) + stop * index) / last, dtype)
        for index in range(count)
    )


def _axis_weights(
    profile: WindowWeightProfile,
    indices: tuple[int, ...],
    *,
    extent: int,
    modular: bool,
    dtype: AccumulationDType,
) -> tuple[float, ...]:
    length = len(indices)
    if profile.kind is WindowWeightKind.FLAT:
        return (_cast_accumulation(1.0, dtype),) * length
    if profile.kind is WindowWeightKind.PYRAMID:
        return tuple(
            _cast_accumulation(float(min(position + 1, length - position)), dtype)
            for position in range(length)
        )
    if profile.overlap > length:
        raise WindowPlanRefusal(
            WindowPlanRefusalCode.INVALID_PLAN,
            "overlap-linear overlap must not exceed its window length",
        )
    weights = [1.0] * length
    if profile.overlap:
        if modular or min(indices) > 0:
            weights[: profile.overlap] = _linspace(1e-37, 1.0, profile.overlap, dtype)
        if modular or max(indices) < extent - 1:
            weights[-profile.overlap :] = _linspace(1.0, 1e-37, profile.overlap, dtype)
    return tuple(_cast_accumulation(weight, dtype) for weight in weights)


def _compile_layer(layer: WindowPlanLayer, axes: dict[str, MediaAxis]) -> _CompiledLayer:
    compiled_windows: list[_CompiledLayerWindow] = []
    for window in layer.windows:
        normalized_lists: list[tuple[int, ...]] = []
        modular_axes: list[bool] = []
        weights_by_axis: list[tuple[float, ...]] = []
        for axis_index, (axis_name, index_list) in enumerate(
            zip(layer.axes, window.index_lists, strict=True)
        ):
            axis = axes[axis_name]
            if index_list.modular:
                if not axis.wrappable:
                    raise WindowPlanRefusal(
                        WindowPlanRefusalCode.INVALID_WRAP,
                        f"axis {axis_name!r} is not declared wrappable",
                    )
                normalized = tuple(index % axis.extent for index in index_list.indices)
            else:
                if any(not 0 <= index < axis.extent for index in index_list.indices):
                    raise WindowPlanRefusal(
                        WindowPlanRefusalCode.INDEX_OUT_OF_RANGE,
                        f"axis {axis_name!r} has a non-modular index outside its extent",
                    )
                normalized = index_list.indices
            profile = layer.weight_profiles[axis_index]
            normalized_lists.append(normalized)
            modular_axes.append(index_list.modular)
            weights_by_axis.append(
                _axis_weights(
                    profile,
                    normalized,
                    extent=axis.extent,
                    modular=index_list.modular,
                    dtype=layer.merge.accumulation_dtype,
                )
            )
        occurrences: list[_LayerOccurrence] = []
        for local_positions in itertools.product(
            *(range(len(indices)) for indices in normalized_lists)
        ):
            weight = _cast_accumulation(1.0, layer.merge.accumulation_dtype)
            coordinate: list[int] = []
            for axis_index, local_position in enumerate(local_positions):
                coordinate.append(normalized_lists[axis_index][local_position])
                weight = _multiply(
                    weight,
                    weights_by_axis[axis_index][local_position],
                    layer.merge.accumulation_dtype,
                )
            occurrences.append(_LayerOccurrence(tuple(coordinate), local_positions, weight))
        compiled_windows.append(
            _CompiledLayerWindow(
                tuple(normalized_lists),
                tuple(modular_axes),
                tuple(occurrences),
            )
        )
    return _CompiledLayer(layer, tuple(compiled_windows))


def _kind_facts(kind: WindowKind) -> dict[str, object]:
    return {
        "axis_maps": tuple(
            {
                "axis": mapping.axis,
                "extent": mapping.extent,
                "profile": _index_profile_facts(mapping.profile),
            }
            for mapping in kind.axis_maps
        ),
        "invariant_axes": kind.invariant_axes,
        "name": kind.name,
    }


def _joint_window_digest(axis_indices: tuple[tuple[str, tuple[int, ...]], ...]) -> str:
    return _digest(_canonical_json((_JOINT_WINDOW_DOMAIN, axis_indices)))


def _coordinate_index(coordinate: tuple[int, ...], extents: tuple[int, ...]) -> int:
    index = 0
    for axis_index, extent in zip(coordinate, extents, strict=True):
        index = index * extent + axis_index
    return index


def _lift_passthrough_coordinates(
    layer: WindowPlanLayer,
    axis_names: tuple[str, ...],
    axes: dict[str, MediaAxis],
) -> tuple[tuple[int, ...], ...]:
    layer_positions = {axis: position for position, axis in enumerate(layer.axes)}
    lifted: list[tuple[int, ...]] = []
    for local_coordinate in layer.passthrough_coordinates:
        choices = tuple(
            (local_coordinate[layer_positions[axis]],)
            if axis in layer_positions
            else tuple(range(axes[axis].extent))
            for axis in axis_names
        )
        lifted.extend(itertools.product(*choices))
    return tuple(lifted)


def compile_window_plan(
    *,
    axes: tuple[MediaAxis, ...],
    kinds: tuple[WindowKind, ...] = (),
    layers: tuple[WindowPlanLayer | CompositeWindowPlan, ...],
    mesh_feasible: Callable[[JointWindow], bool] | None = None,
) -> CompositeWindowPlan:
    """Compile flat layers into one canonical, total composite plan."""

    if type(axes) is not tuple or any(type(axis) is not MediaAxis for axis in axes):
        raise TypeError("axes must be a tuple of exact MediaAxis values")
    if not axes:
        raise WindowPlanRefusal(WindowPlanRefusalCode.INVALID_PLAN, "at least one axis is required")
    axis_by_name = {axis.name: axis for axis in axes}
    if len(axis_by_name) != len(axes):
        raise WindowPlanRefusal(WindowPlanRefusalCode.INVALID_PLAN, "axis names must be unique")
    if type(kinds) is not tuple or any(type(kind) is not WindowKind for kind in kinds):
        raise TypeError("kinds must be a tuple of exact WindowKind values")
    ordered_kinds = tuple(sorted(kinds, key=lambda kind: kind.name))
    if len({kind.name for kind in ordered_kinds}) != len(ordered_kinds):
        raise WindowPlanRefusal(WindowPlanRefusalCode.INVALID_PLAN, "kind names must be unique")
    if type(layers) is not tuple:
        raise TypeError("layers must be a tuple")
    if not layers:
        raise WindowPlanRefusal(
            WindowPlanRefusalCode.INVALID_PLAN,
            "at least one window-plan layer is required",
        )
    if any(type(layer) is CompositeWindowPlan for layer in layers):
        raise WindowPlanRefusal(
            WindowPlanRefusalCode.NESTED_OR_STAGED_PLAN,
            "nested or staged window plans are reserved",
        )
    if any(type(layer) is not WindowPlanLayer for layer in layers):
        raise TypeError("layers must contain exact WindowPlanLayer values")
    ordered_layers = tuple(
        sorted(cast("tuple[WindowPlanLayer, ...]", layers), key=lambda layer: layer.digest)
    )

    claimed_axes: set[str] = set()
    for layer in ordered_layers:
        for axis in layer.axes:
            if axis not in axis_by_name:
                raise WindowPlanRefusal(
                    WindowPlanRefusalCode.INVALID_PLAN,
                    f"layer claims undeclared axis {axis!r}",
                )
            if axis in claimed_axes:
                raise WindowPlanRefusal(
                    WindowPlanRefusalCode.OVERLAPPING_AXIS_CLAIM,
                    f"more than one layer claims axis {axis!r}",
                )
            claimed_axes.add(axis)
    semantic_axis_names = tuple(sorted(claimed_axes))
    semantic_axes = tuple(axis_by_name[name] for name in semantic_axis_names)

    merge = ordered_layers[0].merge
    if any(layer.merge != merge for layer in ordered_layers[1:]):
        raise WindowPlanRefusal(
            WindowPlanRefusalCode.CONFLICTING_MERGE_DECLARATION,
            "all layers must declare the same merge algorithm, dtype, and wrap capability",
        )

    for kind in ordered_kinds:
        map_by_axis = {mapping.axis: mapping for mapping in kind.axis_maps}
        for mapping in kind.axis_maps:
            if mapping.axis not in axis_by_name:
                raise WindowPlanRefusal(
                    WindowPlanRefusalCode.INVALID_PLAN,
                    f"kind {kind.name!r} maps undeclared axis {mapping.axis!r}",
                )
            _index_profile_facts(mapping.profile)
        for axis in kind.invariant_axes:
            if axis not in axis_by_name:
                raise WindowPlanRefusal(
                    WindowPlanRefusalCode.INVALID_PLAN,
                    f"kind {kind.name!r} marks undeclared axis {axis!r} invariant",
                )
        for axis in semantic_axis_names:
            if axis not in map_by_axis and axis not in kind.invariant_axes:
                raise WindowPlanRefusal(
                    WindowPlanRefusalCode.MISSING_AXIS_MAP,
                    f"kind {kind.name!r} is neither mapped nor invariant on axis {axis!r}",
                )

    passthrough_layers = tuple(layer for layer in ordered_layers if layer.passthrough_coordinates)
    if len(passthrough_layers) > 1:
        raise WindowPlanRefusal(
            WindowPlanRefusalCode.CONFLICTING_STRUCTURAL_CLAIM,
            "lifted passthrough restorations from disjoint layers intersect",
        )
    for layer in passthrough_layers:
        for coordinate in layer.passthrough_coordinates:
            if any(
                not 0 <= index < axis_by_name[axis].extent
                for axis, index in zip(layer.axes, coordinate, strict=True)
            ):
                raise WindowPlanRefusal(
                    WindowPlanRefusalCode.INVALID_PLAN,
                    "passthrough coordinate is outside its declared layer geometry",
                )

    compiled_layers = tuple(_compile_layer(layer, axis_by_name) for layer in ordered_layers)
    if (
        any(
            modular
            for layer in compiled_layers
            for window in layer.windows
            for modular in window.modular_axes
        )
        and not merge.supports_wrapping
    ):
        raise WindowPlanRefusal(
            WindowPlanRefusalCode.WRAP_MERGE_UNSUPPORTED,
            "the composite merge does not support modular windows",
        )

    joint_windows: list[JointWindow] = []
    for joint_index, component_windows in enumerate(
        itertools.product(*(layer.windows for layer in compiled_layers))
    ):
        axis_indices_map: dict[str, tuple[int, ...]] = {}
        for compiled_layer, component_window in zip(
            compiled_layers, component_windows, strict=True
        ):
            axis_indices_map.update(
                zip(compiled_layer.declaration.axes, component_window.index_lists, strict=True)
            )
        axis_indices = tuple((axis, axis_indices_map[axis]) for axis in semantic_axis_names)

        kind_indices: list[KindWindowIndices] = []
        for kind in ordered_kinds:
            map_by_axis = {mapping.axis: mapping for mapping in kind.axis_maps}
            mapped_axes = tuple(
                (
                    axis,
                    map_window_indices(
                        map_by_axis[axis].profile,
                        axis_indices_map[axis],
                        primary_extent=axis_by_name[axis].extent,
                        kind_extent=map_by_axis[axis].extent,
                    ),
                )
                for axis in semantic_axis_names
                if axis in map_by_axis
            )
            kind_indices.append(KindWindowIndices(kind.name, mapped_axes))

        occurrences: list[WindowOccurrence] = []
        for component_occurrences in itertools.product(
            *(window.occurrences for window in component_windows)
        ):
            coordinate_by_axis: dict[str, int] = {}
            local_positions: list[int] = []
            weight = _cast_accumulation(1.0, merge.accumulation_dtype)
            for compiled_layer, occurrence in zip(
                compiled_layers, component_occurrences, strict=True
            ):
                coordinate_by_axis.update(
                    zip(
                        compiled_layer.declaration.axes,
                        occurrence.coordinate,
                        strict=True,
                    )
                )
                local_positions.extend(occurrence.local_positions)
                weight = _multiply(weight, occurrence.weight, merge.accumulation_dtype)
            occurrences.append(
                WindowOccurrence(
                    tuple(coordinate_by_axis[axis] for axis in semantic_axis_names),
                    tuple(local_positions),
                    weight,
                )
            )
        if len(compiled_layers) > 1:
            occurrences.sort(
                key=lambda occurrence: (occurrence.coordinate, occurrence.local_positions)
            )
        joint_windows.append(
            JointWindow(
                joint_index,
                axis_indices,
                tuple(kind_indices),
                tuple(occurrences),
                _joint_window_digest(axis_indices),
            )
        )

    infeasible: list[int] = []
    if mesh_feasible is not None:
        for window in joint_windows:
            if not mesh_feasible(window):
                infeasible.append(window.index)
    if infeasible:
        raise WindowPlanRefusal(
            WindowPlanRefusalCode.INFEASIBLE_WINDOW,
            f"mesh is infeasible for joint windows {tuple(infeasible)!r}",
        )

    extents = tuple(axis.extent for axis in semantic_axes)
    coordinates = tuple(itertools.product(*(range(extent) for extent in extents)))
    total_weights = [0.0] * len(coordinates)
    for window in joint_windows:
        for occurrence in window.occurrences:
            coordinate_index = _coordinate_index(occurrence.coordinate, extents)
            total_weights[coordinate_index] = _add(
                total_weights[coordinate_index],
                occurrence.weight,
                merge.accumulation_dtype,
            )
    invalid_coordinates = tuple(
        coordinate
        for coordinate, weight in zip(coordinates, total_weights, strict=True)
        if not math.isfinite(weight) or weight <= 0.0
    )
    if invalid_coordinates:
        raise WindowPlanRefusal(
            WindowPlanRefusalCode.NON_TOTAL_MERGE,
            f"semantic coordinates lack positive finite weight: {invalid_coordinates!r}",
        )

    passthrough_coordinates = (
        _lift_passthrough_coordinates(passthrough_layers[0], semantic_axis_names, axis_by_name)
        if passthrough_layers
        else ()
    )
    layer_digests = tuple(layer.digest for layer in ordered_layers)
    compiled_facts = {
        "axes": tuple(
            {"extent": axis.extent, "name": axis.name, "wrappable": axis.wrappable}
            for axis in semantic_axes
        ),
        "joint_windows": tuple(
            {
                "axis_indices": window.axis_indices,
                "digest": window.digest,
                "kind_indices": tuple(
                    {"axes": indices.axes, "kind": indices.kind} for indices in window.kind_indices
                ),
                "occurrences": tuple(
                    {
                        "coordinate": occurrence.coordinate,
                        "local_positions": occurrence.local_positions,
                        "weight": _canonical_float(occurrence.weight),
                    }
                    for occurrence in window.occurrences
                ),
            }
            for window in joint_windows
        ),
        "kinds": tuple(_kind_facts(kind) for kind in ordered_kinds),
        "merge": {
            "accumulation_dtype": merge.accumulation_dtype.value,
            "algorithm": merge.algorithm,
            "supports_wrapping": merge.supports_wrapping,
        },
        "passthrough_coordinates": passthrough_coordinates,
        "total_weights": tuple(_canonical_float(weight) for weight in total_weights),
    }
    canonical_preimage = _canonical_json(
        (_COMPOSITE_DOMAIN, len(ordered_layers), layer_digests, compiled_facts)
    )
    plan = object.__new__(CompositeWindowPlan)
    object.__setattr__(plan, "axes", semantic_axes)
    object.__setattr__(plan, "kinds", ordered_kinds)
    object.__setattr__(plan, "layers", ordered_layers)
    object.__setattr__(plan, "layer_digests", layer_digests)
    object.__setattr__(plan, "joint_windows", tuple(joint_windows))
    object.__setattr__(plan, "merge", merge)
    object.__setattr__(plan, "coordinates", coordinates)
    object.__setattr__(plan, "total_weights", tuple(total_weights))
    object.__setattr__(plan, "passthrough_coordinates", passthrough_coordinates)
    object.__setattr__(plan, "canonical_preimage", canonical_preimage)
    object.__setattr__(plan, "digest", _digest(canonical_preimage))
    return plan


def merge_window_outputs(
    plan: CompositeWindowPlan,
    outputs: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    """Merge semantic per-occurrence outputs in the plan's fixed traversal order."""

    if type(plan) is not CompositeWindowPlan:
        raise TypeError("plan must be an exact CompositeWindowPlan")
    output_windows = tuple(tuple(values) for values in outputs)
    if len(output_windows) != len(plan.joint_windows):
        raise WindowMergeError("outputs must contain one value sequence per joint window")
    extents = tuple(axis.extent for axis in plan.axes)
    accumulators = [0.0] * len(plan.coordinates)
    for window, values in zip(plan.joint_windows, output_windows, strict=True):
        if len(values) != len(window.occurrences):
            raise WindowMergeError(
                f"joint window {window.index} requires {len(window.occurrences)} outputs"
            )
        for occurrence, value in zip(window.occurrences, values, strict=True):
            if type(value) not in (int, float):
                raise WindowMergeError("window output values must be exact ints or floats")
            coordinate_index = _coordinate_index(occurrence.coordinate, extents)
            contribution = _multiply(
                float(value),
                occurrence.weight,
                plan.merge.accumulation_dtype,
            )
            accumulators[coordinate_index] = _add(
                accumulators[coordinate_index],
                contribution,
                plan.merge.accumulation_dtype,
            )
    return tuple(
        _divide(numerator, denominator, plan.merge.accumulation_dtype)
        for numerator, denominator in zip(accumulators, plan.total_weights, strict=True)
    )


def window_cache_digest(plan: CompositeWindowPlan, joint_window_index: int) -> str:
    """Bind cache identity to both one joint window and its composite plan."""

    if type(plan) is not CompositeWindowPlan:
        raise TypeError("plan must be an exact CompositeWindowPlan")
    if type(joint_window_index) is not int or not 0 <= joint_window_index < len(plan.joint_windows):
        raise ValueError("joint_window_index must name a window in the plan")
    return _digest(
        _canonical_json(
            (
                _CACHE_DOMAIN,
                plan.joint_windows[joint_window_index].digest,
                plan.digest,
            )
        )
    )
