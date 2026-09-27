from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, Literal, Protocol, TypeAlias, TypeVar, runtime_checkable


TensorT = TypeVar("TensorT")
TensorT_co = TypeVar("TensorT_co", covariant=True)
StateT = TypeVar("StateT")
OptionValue: TypeAlias = float | int | str | bool | None


@runtime_checkable
class DivTensor(Protocol[TensorT_co]):
    def __truediv__(self, value: float) -> TensorT_co: ...


class Denoiser(Protocol[TensorT_co]):
    def __call__(self, value: TensorT_co, sigma: float) -> TensorT_co: ...


class NoiseSampler(Protocol[TensorT_co]):
    def __call__(self, sigma_from: float, sigma_to: float) -> TensorT_co: ...


class Parameterization(Enum):
    EPS = "eps"
    V_PREDICTION = "v_prediction"
    EDM = "edm"
    FLOW = "flow"
    IMAGE_TO_IMAGE_FLOW = "image_to_image_flow"
    X0 = "x0"


def is_flow_parameterization(parameterization: Parameterization) -> bool:
    return parameterization in (Parameterization.FLOW, Parameterization.IMAGE_TO_IMAGE_FLOW)


class NoiseKind(Enum):
    NONE = "none"
    RES4LYF_GAUSSIAN = "res4lyf_gaussian"


@dataclass(frozen=True)
class SolverStateEvent(Generic[StateT]):
    step: int
    total: int
    sigma: float
    phase: Literal["pre_update", "post_update"]
    current: StateT
    denoised: StateT | None = None


@dataclass(frozen=True)
class SubstepEvent:
    outer_step: int
    substep: int
    evaluation: int
    sigma: float


@dataclass(frozen=True)
class SamplerInfo:
    parameterization: Parameterization
    seed: int = 0
    sigma_min: float | None = None
    sigma_max: float | None = None
    on_state: Callable[[SolverStateEvent[object]], None] | None = field(default=None, repr=False, compare=False)
    on_substep: Callable[[SubstepEvent], None] | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class StepEvent:
    step: int
    total: int
    sigma: float


StepCallback: TypeAlias = Callable[[StepEvent], None]
SolverFn: TypeAlias = Callable[..., TensorT]
StepBeginSolverFn: TypeAlias = Callable[..., TensorT]


class OptionKind(Enum):
    FLOAT = "float"
    CHOICE = "choice"


@dataclass(frozen=True)
class OptionSpec:
    name: str
    kind: OptionKind
    default: OptionValue
    doc: str = ""
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or "." in self.name:
            raise ValueError(f"option name must be bare and nonempty, got {self.name!r}")
        self.check(self.default)

    def check(self, value: object) -> OptionValue:
        if self.kind is OptionKind.FLOAT:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"option {self.name!r} must be a finite float")
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError(f"option {self.name!r} must be a finite float")
            if self.minimum is not None and normalized < self.minimum:
                raise ValueError(f"option {self.name!r} must be at least {self.minimum}")
            if self.maximum is not None and normalized > self.maximum:
                raise ValueError(f"option {self.name!r} must be at most {self.maximum}")
            return normalized
        if type(value) is not str or value not in self.choices:
            raise ValueError(f"option {self.name!r} must be one of {', '.join(self.choices)}")
        return value


@dataclass(frozen=True)
class SamplerDescriptor(Generic[TensorT]):
    id: str
    display_name: str
    make: Callable[[Mapping[str, OptionValue]], SolverFn[TensorT]]
    options: tuple[OptionSpec, ...] = ()
    noise: NoiseKind = NoiseKind.NONE
    aliases: tuple[str, ...] = ()
    supports_step_begin: bool = False

    def build(self, **overrides: object) -> SolverFn[TensorT]:
        unknown = set(overrides).difference(option.name for option in self.options)
        if unknown:
            raise ValueError(f"unknown sampler options: {', '.join(sorted(unknown))}")
        values: dict[str, OptionValue] = {}
        for option in self.options:
            value = overrides.get(option.name, option.default)
            values[option.name] = option.check(value)
        return self.make(values)


def ordered_descriptors(descriptors: Sequence[SamplerDescriptor[TensorT]]) -> tuple[SamplerDescriptor[TensorT], ...]:
    names: set[str] = set()
    ordered: list[SamplerDescriptor[TensorT]] = []
    for descriptor in descriptors:
        for name in (descriptor.id, *descriptor.aliases):
            if name in names:
                raise ValueError(f"duplicate sampler identity {name!r}")
            names.add(name)
        ordered.append(descriptor)
    return tuple(ordered)
