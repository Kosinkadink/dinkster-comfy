from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable

import torch


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return 0.0 if value == 0.0 else value


@dataclass(frozen=True)
class GainKeyframe:
    keyframe_id: str
    anchor: float
    gain: float
    coordinate: str = "progress"
    minimum_realized_steps: int = 1

    def __post_init__(self):
        if not self.keyframe_id:
            raise ValueError("gain keyframe id must not be empty")
        if self.coordinate not in ("progress", "sigma", "step_index"):
            raise ValueError(f"unsupported gain coordinate: {self.coordinate}")
        object.__setattr__(self, "anchor", _finite(self.anchor, "gain anchor"))
        object.__setattr__(self, "gain", _finite(self.gain, "keyframe gain"))
        if self.minimum_realized_steps < 1:
            raise ValueError("minimum_realized_steps must be positive")


@dataclass(frozen=True)
class GainTimeline:
    constant: float | None = 1.0
    keyframes: tuple[GainKeyframe, ...] = ()
    table: tuple[float, ...] = ()
    interpolation: str = "hold.v1"
    endpoint_before: str = "clamp.v1"
    endpoint_after: str = "clamp.v1"
    minimum_policy: str = "best_effort"

    def __post_init__(self):
        declarations = sum(
            (self.constant is not None, bool(self.keyframes), bool(self.table))
        )
        if declarations != 1:
            raise ValueError("gain timeline requires exactly one declaration")
        if self.constant is not None:
            object.__setattr__(self, "constant", _finite(self.constant, "timeline gain"))
        if self.table:
            object.__setattr__(
                self,
                "table",
                tuple(_finite(value, "timeline table gain") for value in self.table),
            )
        if self.interpolation not in ("hold.v1", "linear.v1", "smoothstep.v1"):
            raise ValueError(f"unsupported gain interpolation: {self.interpolation}")
        for endpoint in (self.endpoint_before, self.endpoint_after):
            if endpoint not in ("clamp.v1", "zero.v1"):
                raise ValueError(f"unsupported gain endpoint: {endpoint}")
        if self.minimum_policy not in ("best_effort", "hard"):
            raise ValueError(f"unsupported keyframe minimum policy: {self.minimum_policy}")
        if self.keyframes:
            coordinates = {keyframe.coordinate for keyframe in self.keyframes}
            if len(coordinates) != 1:
                raise ValueError("gain keyframes must use one coordinate")
            anchors = [keyframe.anchor for keyframe in self.keyframes]
            reverse = self.keyframes[0].coordinate == "sigma"
            if anchors != sorted(anchors, reverse=reverse) or len(set(anchors)) != len(anchors):
                raise ValueError("gain keyframes must be unique and in execution order")

    @classmethod
    def constant_gain(cls, gain: float = 1.0) -> GainTimeline:
        return cls(constant=gain)

    @classmethod
    def from_table(cls, gains) -> GainTimeline:
        return cls(constant=None, table=tuple(gains))

    @classmethod
    def from_keyframes(
        cls,
        keyframes,
        *,
        interpolation: str = "hold.v1",
        endpoint_before: str = "clamp.v1",
        endpoint_after: str = "clamp.v1",
        minimum_policy: str = "best_effort",
    ) -> GainTimeline:
        return cls(
            constant=None,
            keyframes=tuple(keyframes),
            interpolation=interpolation,
            endpoint_before=endpoint_before,
            endpoint_after=endpoint_after,
            minimum_policy=minimum_policy,
        )

    @classmethod
    def active_range(cls, start: float, end: float) -> GainTimeline:
        start = _finite(start, "gain range start")
        end = _finite(end, "gain range end")
        if start < 0.0 or end > 1.0 or start >= end:
            raise ValueError("gain range must satisfy 0 <= start < end <= 1")
        return cls.from_keyframes(
            (
                GainKeyframe("range.start", start, 1.0),
                GainKeyframe("range.end", end, 1.0),
            ),
            endpoint_before="zero.v1",
            endpoint_after="zero.v1",
        )

    def realize(
        self,
        sigmas: torch.Tensor,
        percent_to_sigma: Callable[[float], float] | None = None,
    ) -> tuple[float, ...]:
        executed = tuple(float(value) for value in sigmas.detach().cpu().flatten()[:-1])
        if self.constant is not None:
            return (self.constant,) * len(executed)
        if self.table:
            if len(self.table) != len(executed):
                raise ValueError("gain table must match the executed sigma rows")
            return self.table
        if not self.keyframes:
            raise ValueError("gain timeline has no keyframes")

        coordinate = self.keyframes[0].coordinate
        if coordinate == "progress" and percent_to_sigma is None:
            raise ValueError("progress gain keyframes require percent_to_sigma")
        anchors = tuple(
            float(percent_to_sigma(keyframe.anchor))
            if coordinate == "progress"
            else keyframe.anchor
            for keyframe in self.keyframes
        )
        if coordinate == "step_index":
            positions = tuple(float(index) for index in range(len(executed)))
        else:
            positions = executed
        realized = list(self._realize_keyframes(positions, anchors))
        self._assign_keyframe_holds(realized, positions, anchors)
        return tuple(realized)

    def _assign_keyframe_holds(self, realized, positions, anchors):
        if not realized:
            if self.minimum_policy == "hard":
                raise ValueError("unsatisfied_keyframe_minimum")
            return
        cursor = 0
        for index, keyframe in enumerate(self.keyframes):
            requested = keyframe.minimum_realized_steps
            available = len(realized) - cursor
            count = min(requested, available)
            if count < requested and self.minimum_policy == "hard":
                raise ValueError("unsatisfied_keyframe_minimum")
            if count == 0:
                continue
            later = len(self.keyframes) - index - 1
            latest_start = max(cursor, len(realized) - count - min(later, len(realized) - count))
            nearest = min(
                range(cursor, len(realized)),
                key=lambda row: (abs(positions[row] - anchors[index]), row),
            )
            start = min(max(cursor, nearest - (count - 1) // 2), latest_start)
            for row in range(start, start + count):
                realized[row] = keyframe.gain
            cursor = start + count

    def _realize_keyframes(
        self, positions: tuple[float, ...], anchors: tuple[float, ...]
    ) -> tuple[float, ...]:
        descending = self.keyframes[0].coordinate in ("progress", "sigma")
        realized = []
        for position in positions:
            before = position > anchors[0] if descending else position < anchors[0]
            after = position < anchors[-1] if descending else position > anchors[-1]
            if before:
                realized.append(0.0 if self.endpoint_before == "zero.v1" else self.keyframes[0].gain)
                continue
            if after:
                realized.append(0.0 if self.endpoint_after == "zero.v1" else self.keyframes[-1].gain)
                continue
            right = len(anchors) - 1
            for index in range(1, len(anchors)):
                reached = position >= anchors[index] if not descending else position <= anchors[index]
                if not reached:
                    continue
                right = index
                break
            left = max(0, right - 1)
            if self.interpolation == "hold.v1":
                index = right if math.isclose(position, anchors[right]) else left
                realized.append(self.keyframes[index].gain)
                continue
            if left == right:
                realized.append(self.keyframes[right].gain)
                continue
            span = anchors[right] - anchors[left]
            fraction = (position - anchors[left]) / span
            if self.interpolation == "smoothstep.v1":
                fraction = fraction * fraction * (3.0 - 2.0 * fraction)
            realized.append(
                self.keyframes[left].gain
                + (self.keyframes[right].gain - self.keyframes[left].gain) * fraction
            )
        return tuple(realized)


@dataclass(frozen=True)
class ContributionGain:
    timeline: GainTimeline = field(default_factory=GainTimeline.constant_gain)
    global_gain: float = 1.0
    site_gains: tuple[tuple[str, float], ...] = ()
    lane_gains: tuple[tuple[str, float], ...] = ()
    effect_masks: tuple[torch.Tensor, ...] = field(default=(), compare=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "global_gain", _finite(self.global_gain, "global gain"))
        object.__setattr__(
            self,
            "site_gains",
            self._normalize_gains(self.site_gains, "site"),
        )
        object.__setattr__(
            self,
            "lane_gains",
            self._normalize_gains(self.lane_gains, "guidance lane"),
        )
        for mask in self.effect_masks:
            if not isinstance(mask, torch.Tensor):
                raise TypeError("effect masks must be tensors")
            if not torch.isfinite(mask).all() or bool((mask < 0).any()) or bool((mask > 1).any()):
                raise ValueError("effect mask values must be finite fractions")

    @staticmethod
    def _normalize_gains(gains, kind):
        normalized = tuple(
            sorted((str(key), _finite(value, f"{kind} gain")) for key, value in gains)
        )
        if len({key for key, _ in normalized}) != len(normalized):
            raise ValueError(f"duplicate {kind} gain")
        return normalized

    def with_timeline(self, timeline: GainTimeline) -> ContributionGain:
        return replace(self, timeline=timeline)

    def scaled(self, gain: float) -> ContributionGain:
        return replace(self, global_gain=self.global_gain * _finite(gain, "global gain"))

    def with_effect_masks(self, masks) -> ContributionGain:
        return replace(self, effect_masks=tuple(masks))

    def realize(
        self,
        sigmas: torch.Tensor,
        percent_to_sigma: Callable[[float], float] | None = None,
    ) -> RealizedGainTable:
        return RealizedGainTable(
            sigmas=tuple(float(value) for value in sigmas.detach().cpu().flatten()[:-1]),
            timeline_gains=self.timeline.realize(sigmas, percent_to_sigma),
            global_gain=self.global_gain,
            site_gains=self.site_gains,
            lane_gains=self.lane_gains,
            effect_masks=self.effect_masks,
        )


@dataclass(frozen=True)
class RealizedGainTable:
    sigmas: tuple[float, ...]
    timeline_gains: tuple[float, ...]
    global_gain: float
    site_gains: tuple[tuple[str, float], ...]
    lane_gains: tuple[tuple[str, float], ...]
    effect_masks: tuple[torch.Tensor, ...] = field(compare=False, repr=False)

    def step_index(self, sigma: torch.Tensor | float) -> int:
        value = float(sigma.flatten()[0]) if isinstance(sigma, torch.Tensor) else float(sigma)
        if not self.sigmas:
            raise ValueError("gain table has no executed rows")
        return min(range(len(self.sigmas)), key=lambda index: abs(self.sigmas[index] - value))

    def scalar_gain(
        self, sigma: torch.Tensor | float, *, site: str | None = None, lane: str | None = None
    ) -> float:
        site_gain = dict(self.site_gains).get(site, 1.0)
        lane_gain = dict(self.lane_gains).get(lane, 1.0)
        return (
            self.timeline_gains[self.step_index(sigma)]
            * self.global_gain
            * site_gain
            * lane_gain
        )

    def tensor_gain(
        self,
        sigma: torch.Tensor | float,
        output: torch.Tensor,
        *,
        site: str | None = None,
        lanes=(),
    ) -> float | torch.Tensor:
        if self.lane_gains and lanes:
            lane_names = tuple(
                "positive" if lane == 0 else "negative" for lane in lanes
            )
            values = [
                self.scalar_gain(sigma, site=site, lane=lane)
                for lane in lane_names
            ]
            repeats = output.shape[0] // len(values)
            gain = torch.tensor(
                values, device=output.device, dtype=output.dtype
            ).repeat_interleave(repeats)
            gain = gain.reshape((output.shape[0],) + (1,) * (output.ndim - 1))
        else:
            gain = self.scalar_gain(sigma, site=site)
        for mask in self.effect_masks:
            mask = mask.to(device=output.device, dtype=output.dtype)
            if output.ndim == 3 and mask.ndim >= 3:
                mask = mask.flatten(1).unsqueeze(-1)
            elif mask.ndim == output.ndim - 1:
                mask = mask.unsqueeze(1)
            if mask.shape[0] != output.shape[0]:
                mask = mask.repeat_interleave(output.shape[0] // mask.shape[0], dim=0)
            gain = gain * mask
        return gain
