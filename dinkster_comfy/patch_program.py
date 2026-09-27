from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch


def _tensor_descriptor(value: torch.Tensor) -> dict[str, object]:
    tensor = value.detach().cpu().contiguous()
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": hashlib.sha256(
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
    }


def _descriptor(value: object, active: set[int] | None = None) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("patch program values must be finite")
        return 0.0 if value == 0.0 else value
    if isinstance(value, torch.Tensor):
        return {"tensor": _tensor_descriptor(value)}

    if active is None:
        active = set()
    identity = id(value)
    if identity in active:
        raise ValueError("patch program values must not contain cycles")
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            return {
                "mapping": [
                    [_descriptor(key, active), _descriptor(item, active)]
                    for key, item in sorted(
                        value.items(), key=lambda pair: repr(pair[0])
                    )
                ]
            }
        if isinstance(value, (set, frozenset)):
            items = [_descriptor(item, active) for item in value]
            items.sort(key=lambda item: json.dumps(item, sort_keys=True))
            return {"set": items}
        if isinstance(value, (list, tuple)):
            return {"sequence": [_descriptor(item, active) for item in value]}
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                "class": f"{type(value).__module__}.{type(value).__qualname__}",
                "fields": {
                    item.name: _descriptor(getattr(value, item.name), active)
                    for item in dataclasses.fields(value)
                    if item.compare
                },
            }
        if callable(value):
            return {
                "callable": f"{value.__module__}.{value.__qualname__}",
            }
        attributes = getattr(value, "__dict__", None)
        if attributes is not None:
            return {
                "class": f"{type(value).__module__}.{type(value).__qualname__}",
                "attributes": _descriptor(attributes, active),
            }
    finally:
        active.remove(identity)
    raise TypeError(
        f"unsupported patch program value: {type(value).__module__}.{type(value).__qualname__}"
    )


def _digest(value: object) -> str:
    payload = json.dumps(
        _descriptor(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(b"dinkster.patch-program.v1\0" + payload).hexdigest()


@dataclass(frozen=True)
class PatchResource:
    identity: str
    value: object = field(compare=False, repr=False)

    @classmethod
    def bind(cls, value: object) -> PatchResource:
        return cls(identity=_digest(value), value=value)


@dataclass(frozen=True)
class WeightDeltaEntry:
    target: str
    patch: PatchResource
    strength_patch: float
    strength_model: float
    offset: object = None
    function: object = field(default=None, compare=False, repr=False)
    function_identity: str | None = None

    @classmethod
    def create(
        cls,
        *,
        target: str,
        patch: object,
        strength_patch: float,
        strength_model: float,
        offset: object = None,
        function: object = None,
    ) -> WeightDeltaEntry:
        strengths = {
            "strength_patch": float(strength_patch),
            "strength_model": float(strength_model),
        }
        for name, value in strengths.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        return cls(
            target=target,
            patch=PatchResource.bind(patch),
            strength_patch=0.0
            if strengths["strength_patch"] == 0.0
            else strengths["strength_patch"],
            strength_model=0.0
            if strengths["strength_model"] == 0.0
            else strengths["strength_model"],
            offset=offset,
            function=function,
            function_identity=None if function is None else _digest(function),
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "kind": "weight_delta",
            "target": self.target,
            "patch": self.patch.identity,
            "strength_patch": self.strength_patch,
            "strength_model": self.strength_model,
            "offset": _descriptor(self.offset),
            "function": self.function_identity,
        }

    def runtime_tuple(self) -> tuple[object, object, object, object, object]:
        return (
            self.strength_patch,
            self.patch.value,
            self.strength_model,
            self.offset,
            self.function,
        )


@dataclass(frozen=True)
class PatchProgram:
    entries: tuple[WeightDeltaEntry, ...] = ()

    @property
    def digest(self) -> str:
        payload = json.dumps(
            [entry.descriptor() for entry in self.entries],
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return hashlib.sha256(b"dinkster.patch-program.v1\0" + payload).hexdigest()

    def append_weight_delta(
        self,
        *,
        target: str,
        patch: object,
        strength_patch: float,
        strength_model: float,
        offset: object = None,
        function: object = None,
    ) -> PatchProgram:
        entry = WeightDeltaEntry.create(
            target=target,
            patch=patch,
            strength_patch=strength_patch,
            strength_model=strength_model,
            offset=offset,
            function=function,
        )
        return PatchProgram((*self.entries, entry))

    def weight_patches(
        self,
    ) -> dict[str, list[tuple[object, object, object, object, object]]]:
        patches: dict[str, list[tuple[object, object, object, object, object]]] = {}
        for entry in self.entries:
            patches.setdefault(entry.target, []).append(entry.runtime_tuple())
        return patches

    @classmethod
    def from_weight_patches(cls, patches: Mapping[str, list[tuple]]) -> PatchProgram:
        program = cls()
        for target, entries in patches.items():
            for entry in entries:
                if len(entry) != 5:
                    raise ValueError("weight patch entries must contain five values")
                strength_patch, patch, strength_model, offset, function = entry
                program = program.append_weight_delta(
                    target=target,
                    patch=patch,
                    strength_patch=strength_patch,
                    strength_model=strength_model,
                    offset=offset,
                    function=function,
                )
        return program
