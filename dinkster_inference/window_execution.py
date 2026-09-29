"""Tensor gathering and merge for compiled window plans."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch

from .nested_tensor import NestedTensor
from . import patcher_extension, utils
from .window_plan import (
    AccumulationDType,
    CompositeWindowPlan,
    JointWindow,
    ProportionalRangeIndexMap,
    map_window_indices,
)

__all__ = [
    "CompiledWindowField",
    "WindowPlanExecutor",
    "WindowTensorLayout",
    "gather_window_value",
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


def _pack_primary_mask(mask: torch.Tensor, shapes: tuple[torch.Size, ...]) -> torch.Tensor:
    primary = mask.unsqueeze(1).expand(mask.shape[0], shapes[0][1], *mask.shape[1:])
    streams = [primary]
    streams.extend(
        torch.ones((mask.shape[0], *shape[1:]), dtype=mask.dtype, device=mask.device)
        for shape in shapes[1:]
    )
    return utils.pack_latents(streams)[0].squeeze(1)


def gather_window_value(
    value,
    window: JointWindow,
    packed_shapes: tuple[torch.Size, ...] | None = None,
):
    """Gather declared fields while preserving ordinary conditioning values."""

    if type(value) is CompiledWindowField:
        gathered = value.gather(window)
        if packed_shapes is not None:
            return _pack_primary_mask(gathered, packed_shapes)
        return gathered
    if type(value) is dict:
        return {
            key: gather_window_value(item, window, packed_shapes)
            for key, item in value.items()
        }
    if type(value) is list:
        return [gather_window_value(item, window, packed_shapes) for item in value]
    if type(value) is tuple:
        return tuple(gather_window_value(item, window, packed_shapes) for item in value)
    return value


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

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("window input must be a torch.Tensor")
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
    layout_axes = tuple(axis for axis, _ in layout.axis_dimensions)
    kind = next(kind for kind in plan.kinds if kind.name == layout.kind)
    mapped_axes = tuple(mapping.axis for mapping in kind.axis_maps)
    if set(layout_axes) != set(mapped_axes):
        raise ValueError("output layout must map every non-invariant kind axis")
    occurrence_axes = tuple(axis for layer in plan.layers for axis in layer.axes)
    return {axis: occurrence_axes.index(axis) for axis in layout_axes}


def _kind_local_positions(
    plan: CompositeWindowPlan,
    window: JointWindow,
    layout: WindowTensorLayout,
) -> dict[str, tuple[tuple[int, int], ...]]:
    kind = next(kind for kind in plan.kinds if kind.name == layout.kind)
    mappings = {mapping.axis: mapping for mapping in kind.axis_maps}
    primary_indices = dict(window.axis_indices)
    kind_indices = _kind_indices(window, layout.kind)
    axis_extents = {axis.name: axis.extent for axis in plan.axes}
    result = {}
    for axis, _ in layout.axis_dimensions:
        mapping = mappings[axis]
        entries = []
        seen = set()
        for primary_position, primary_index in enumerate(primary_indices[axis]):
            mapped = map_window_indices(
                mapping.profile,
                (primary_index,),
                primary_extent=axis_extents[axis],
                kind_extent=mapping.extent,
            )
            for coordinate in mapped:
                if type(mapping.profile) is ProportionalRangeIndexMap and coordinate in seen:
                    continue
                seen.add(coordinate)
                entries.append((coordinate, primary_position))
        if tuple(coordinate for coordinate, _ in entries) != kind_indices[axis]:
            raise ValueError("compiled kind indices cannot be traced to primary occurrences")
        result[axis] = tuple(entries)
    return result


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
    occurrence_positions = _axis_positions(plan, layout)
    dimension_by_axis = dict(layout.axis_dimensions)
    axis_by_dimension = {dimension: axis for axis, dimension in layout.axis_dimensions}

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
        local_entries = _kind_local_positions(plan, window, layout)
        projected_weights = {}
        occurrence_axes = tuple(dimension_by_axis)
        for occurrence in window.occurrences:
            key = tuple(
                occurrence.local_positions[occurrence_positions[axis]]
                for axis in occurrence_axes
            )
            projected_weights[key] = projected_weights.get(key, 0.0) + occurrence.weight
        local_shape = tuple(len(local_entries[axis]) for axis in occurrence_axes)
        weights = torch.tensor(
            [
                projected_weights.get(
                    tuple(
                        local_entries[axis][position][1]
                        for axis, position in zip(occurrence_axes, local_positions, strict=True)
                    ),
                    0.0,
                )
                for local_positions in itertools.product(*(range(size) for size in local_shape))
            ],
            dtype=accumulation_dtype,
            device=first.device,
        ).reshape(local_shape)
        dimension_order = sorted(
            range(len(occurrence_axes)),
            key=lambda index: dimension_by_axis[occurrence_axes[index]],
        )
        weights = weights.permute(dimension_order)
        weight_shape = [1] * len(output_shape)
        for axis in occurrence_axes:
            weight_shape[dimension_by_axis[axis]] = len(local_entries[axis])
        weights = weights.reshape(weight_shape)

        if all(len(indices) == len(set(indices)) for indices in kind_indices.values()):
            coordinates = []
            for dimension, size in enumerate(expected_shape):
                axis = axis_by_dimension.get(dimension)
                values = range(size) if axis is None else kind_indices[axis]
                coordinates.append(torch.tensor(tuple(values), device=first.device))
            target = torch.meshgrid(*coordinates, indexing="ij")
            accumulator[target] += output.to(accumulation_dtype) * weights
            denominator[target] += weights.expand(expected_shape)
            continue

        for local_positions in itertools.product(
            *(range(len(local_entries[axis])) for axis in dimension_by_axis)
        ):
            source = [slice(None)] * len(output_shape)
            target = [slice(None)] * len(output_shape)
            primary_positions = {}
            for (axis, dimension), local_position in zip(
                dimension_by_axis.items(), local_positions, strict=True
            ):
                coordinate, primary_position = local_entries[axis][local_position]
                source[dimension] = local_position
                target[dimension] = coordinate
                primary_positions[axis] = primary_position
            weight = projected_weights.get(
                tuple(primary_positions[axis] for axis in occurrence_axes),
                0.0,
            )
            source_index = tuple(source)
            target_index = tuple(target)
            accumulator[target_index].add_(output[source_index].to(accumulation_dtype) * weight)
            denominator[target_index].add_(weight)
    return (accumulator / denominator).to(first.dtype)


@dataclass(frozen=True, slots=True)
class WindowPlanExecutor:
    """Run one conditioning evaluation over every joint window and merge once."""

    plan: CompositeWindowPlan
    latent_layout: WindowTensorLayout | tuple[WindowTensorLayout, ...]

    def __post_init__(self) -> None:
        if type(self.plan) is not CompositeWindowPlan:
            raise TypeError("plan must be an exact CompositeWindowPlan")
        layouts = self.layouts
        if not layouts or any(type(layout) is not WindowTensorLayout for layout in layouts):
            raise TypeError("latent_layout must contain WindowTensorLayout values")

    @property
    def layouts(self) -> tuple[WindowTensorLayout, ...]:
        if type(self.latent_layout) is WindowTensorLayout:
            return (self.latent_layout,)
        if type(self.latent_layout) is tuple:
            return self.latent_layout
        return ()

    def compile_mask(self, tensor: torch.Tensor) -> CompiledWindowField:
        """Compile a channel-free full-domain mask from the primary latent declaration."""

        layout = self.layouts[0]
        if any(dimension < 2 for _, dimension in layout.axis_dimensions):
            raise ValueError("window masks require media axes after the latent channel dimension")
        mask_layout = WindowTensorLayout(
            layout.kind,
            tuple((axis, dimension - 1) for axis, dimension in layout.axis_dimensions),
        )
        if any(dimension >= tensor.ndim for _, dimension in mask_layout.axis_dimensions):
            raise ValueError("window mask does not contain every declared media axis")
        return CompiledWindowField(tensor, mask_layout)

    def _gather_latent(self, value, window: JointWindow):
        if type(value) is NestedTensor:
            tensors = value.unbind()
            if len(tensors) != len(self.layouts):
                raise ValueError("nested latent stream count does not match its declarations")
            return NestedTensor(
                tuple(
                    gather_window_tensor(tensor, layout, window)
                    for tensor, layout in zip(tensors, self.layouts, strict=True)
                )
            )
        if len(self.layouts) != 1:
            raise ValueError("ordinary latent tensors require exactly one declaration")
        return gather_window_tensor(value, self.layouts[0], window)

    def window_latent(self, index, template):
        """Gather the latent shape produced by one joint window."""
        return self._gather_latent(template, self.plan.joint_windows[index])

    def _merge_latent(self, outputs, template):
        if type(template) is NestedTensor:
            streams = tuple(output.unbind() for output in outputs)
            if any(len(output) != len(self.layouts) for output in streams):
                raise ValueError("window output stream count does not match its declarations")
            return NestedTensor(
                tuple(
                    merge_window_tensors(
                        self.plan,
                        layout,
                        tuple(output[index] for output in streams),
                        tuple(template.unbind()[index].shape),
                    )
                    for index, layout in enumerate(self.layouts)
                )
            )
        return merge_window_tensors(
            self.plan,
            self.layouts[0],
            tuple(outputs),
            tuple(template.shape),
        )

    @staticmethod
    def _patch_latent_shapes(conds, shapes):
        for conditioning in conds:
            if conditioning is None:
                continue
            for metadata in conditioning:
                model_conds = metadata.get("model_conds", {})
                latent_shapes = model_conds.get("latent_shapes")
                if latent_shapes is not None:
                    model_conds["latent_shapes"] = latent_shapes._copy_with(shapes)

    def evaluate_window(
        self, index, evaluate, model, conds, template, timestep, model_options, packed=False
    ):
        """Evaluate one joint window for a serial or distributed plan executor."""
        window = self.plan.joint_windows[index]
        sub_x = self.window_latent(index, template)
        sub_shapes = None
        if packed:
            sub_x, sub_shapes = utils.pack_latents(sub_x.unbind())
        sub_conds = gather_window_value(
            conds,
            window,
            tuple(sub_shapes) if sub_shapes is not None else None,
        )
        if packed:
            self._patch_latent_shapes(sub_conds, sub_shapes)
        sub_options = model_options.copy()
        transformer_options = model_options.get("transformer_options", {}).copy()
        transformer_options["window_plan"] = self.plan
        transformer_options["window"] = window
        sub_options["transformer_options"] = transformer_options
        outputs = evaluate(model, sub_conds, sub_x, timestep, sub_options)
        if packed:
            outputs = [
                NestedTensor(tuple(utils.unpack_latents(output, sub_shapes)))
                for output in outputs
            ]
        if len(outputs) != len(conds):
            raise ValueError("window evaluation returned the wrong conditioning count")
        return outputs

    def _evaluate_windows(self, evaluate, model, conds, template, timestep, model_options, packed):
        outputs = [[] for _ in conds]
        for index in range(len(self.plan.joint_windows)):
            sub_outputs = self.evaluate_window(
                index, evaluate, model, conds, template, timestep, model_options, packed
            )
            for condition_index, output in enumerate(sub_outputs):
                outputs[condition_index].append(output)
        return outputs

    def execute(self, evaluate, model, conds, x_in, timestep, model_options):
        packed = type(x_in) is not NestedTensor and len(self.layouts) > 1
        template = x_in
        if packed:
            if model.latent_shapes is None or len(model.latent_shapes) != len(self.layouts):
                raise ValueError("packed latent shapes do not match the window declarations")
            template = NestedTensor(tuple(utils.unpack_latents(x_in, model.latent_shapes)))
        executor = patcher_extension.WrapperExecutor.new_class_executor(
            self._evaluate_windows,
            self,
            patcher_extension.get_all_wrappers(
                patcher_extension.WrappersMP.WINDOW_EXECUTE,
                model_options,
                is_model_options=True,
            ),
        )
        outputs = executor.execute(
            evaluate, model, conds, template, timestep, model_options, packed
        )
        merged = [self._merge_latent(window_outputs, template) for window_outputs in outputs]
        if packed:
            return [utils.pack_latents(output.unbind())[0] for output in merged]
        return merged
