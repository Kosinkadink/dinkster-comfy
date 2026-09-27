"""Tensor gathering and merge for compiled window plans."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .window_plan import AccumulationDType, CompositeWindowPlan, JointWindow

__all__ = [
    "CompiledWindowField",
    "WindowTensorLayout",
    "gather_window_tensor",
    "merge_window_tensors",
]


@dataclass(frozen=True, slots=True)
class WindowTensorLayout:
    """Bind declared media axes to dimensions of one tensor kind."""

    kind: str
    axis_dimensions: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if type(self.kind) is not str or not self.kind:
            raise TypeError("window tensor kind must be a non-empty exact string")
        if type(self.axis_dimensions) is not tuple or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or not item[0]
            or type(item[1]) is not int
            or item[1] < 0
            for item in self.axis_dimensions
        ):
            raise TypeError("axis_dimensions must contain (axis, dimension) pairs")
        axes = tuple(axis for axis, _ in self.axis_dimensions)
        dimensions = tuple(dimension for _, dimension in self.axis_dimensions)
        if axes != tuple(sorted(axes)) or len(axes) != len(set(axes)):
            raise ValueError("window tensor axes must be sorted and unique")
        if len(dimensions) != len(set(dimensions)):
            raise ValueError("window tensor dimensions must be unique")


@dataclass(frozen=True, slots=True)
class CompiledWindowField:
    """A tensor compiled once in full-domain coordinates."""

    tensor: torch.Tensor
    layout: WindowTensorLayout

    def gather(self, window: JointWindow) -> torch.Tensor:
        return gather_window_tensor(self.tensor, self.layout, window)


def _kind_indices(window: JointWindow, kind: str) -> dict[str, tuple[int, ...]]:
    try:
        return dict(next(indices.axes for indices in window.kind_indices if indices.kind == kind))
    except StopIteration:
        raise ValueError(f"joint window has no indices for tensor kind {kind!r}") from None


def gather_window_tensor(
    tensor: torch.Tensor,
    layout: WindowTensorLayout,
    window: JointWindow,
) -> torch.Tensor:
    """Gather one tensor through its declared per-kind media-axis maps."""

    if type(tensor) is not torch.Tensor:
        raise TypeError("window input must be an exact torch.Tensor")
    if type(layout) is not WindowTensorLayout:
        raise TypeError("layout must be an exact WindowTensorLayout")
    if type(window) is not JointWindow:
        raise TypeError("window must be an exact JointWindow")
    indices = _kind_indices(window, layout.kind)
    if set(indices) != {axis for axis, _ in layout.axis_dimensions}:
        raise ValueError("tensor layout axes do not match the compiled kind indices")
    gathered = tensor
    for axis, dimension in layout.axis_dimensions:
        if dimension >= gathered.ndim:
            raise ValueError(f"tensor kind {layout.kind!r} has no dimension {dimension}")
        gathered = gathered.index_select(
            dimension,
            torch.tensor(indices[axis], device=gathered.device),
        )
    return gathered


def _axis_positions(plan: CompositeWindowPlan, layout: WindowTensorLayout) -> dict[str, int]:
    plan_axes = tuple(axis.name for axis in plan.axes)
    layout_axes = tuple(axis for axis, _ in layout.axis_dimensions)
    if set(layout_axes) != set(plan_axes):
        raise ValueError("output layout must map every semantic plan axis")
    occurrence_axes = tuple(axis for layer in plan.layers for axis in layer.axes)
    return {axis: occurrence_axes.index(axis) for axis in layout_axes}


def merge_window_tensors(
    plan: CompositeWindowPlan,
    layout: WindowTensorLayout,
    outputs: tuple[torch.Tensor, ...],
    output_shape: tuple[int, ...],
) -> torch.Tensor:
    """Merge joint-window tensors in declared occurrence order."""

    if type(plan) is not CompositeWindowPlan:
        raise TypeError("plan must be an exact CompositeWindowPlan")
    if type(layout) is not WindowTensorLayout:
        raise TypeError("layout must be an exact WindowTensorLayout")
    if type(outputs) is not tuple or len(outputs) != len(plan.joint_windows):
        raise ValueError("outputs must contain one tensor per joint window")
    if type(output_shape) is not tuple or any(type(size) is not int or size < 1 for size in output_shape):
        raise TypeError("output_shape must be a tuple of positive exact ints")
    first = outputs[0]
    if type(first) is not torch.Tensor:
        raise TypeError("window outputs must be exact torch.Tensor values")
    accumulation_dtype = {
        AccumulationDType.FLOAT32: torch.float32,
        AccumulationDType.FLOAT64: torch.float64,
    }[plan.merge.accumulation_dtype]
    accumulator = torch.zeros(output_shape, dtype=accumulation_dtype, device=first.device)
    denominator = torch.zeros(output_shape, dtype=accumulation_dtype, device=first.device)
    axis_positions = _axis_positions(plan, layout)
    dimension_by_axis = dict(layout.axis_dimensions)

    for window, output in zip(plan.joint_windows, outputs, strict=True):
        expected_shape = list(output_shape)
        kind_indices = _kind_indices(window, layout.kind)
        for axis, dimension in layout.axis_dimensions:
            expected_shape[dimension] = len(kind_indices[axis])
        if (
            type(output) is not torch.Tensor
            or tuple(output.shape) != tuple(expected_shape)
            or output.dtype != first.dtype
            or output.device != first.device
        ):
            raise ValueError(f"joint window {window.index} returned an incompatible tensor")
        for occurrence in window.occurrences:
            source = [slice(None)] * len(output_shape)
            target = [slice(None)] * len(output_shape)
            for axis, dimension in dimension_by_axis.items():
                plan_position = axis_positions[axis]
                source[dimension] = occurrence.local_positions[plan_position]
                target[dimension] = occurrence.coordinate[plan_position]
            source_index = tuple(source)
            target_index = tuple(target)
            weight = occurrence.weight
            accumulator[target_index].add_(output[source_index].to(accumulation_dtype) * weight)
            denominator[target_index].add_(weight)
    return (accumulator / denominator).to(first.dtype)
