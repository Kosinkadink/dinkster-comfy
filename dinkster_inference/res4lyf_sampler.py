from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from . import model_sampling
from .res4lyf_rk import (
    RES4LYF_DEIS_2M,
    RES4LYF_DEIS_2M_ODE,
    RES4LYF_DEIS_3M,
    RES4LYF_DEIS_3M_ODE,
    RES4LYF_RES_2M,
    RES4LYF_RES_2M_ODE,
    RES4LYF_RES_2S,
    RES4LYF_RES_2S_ODE,
    RES4LYF_RES_3M,
    RES4LYF_RES_3M_ODE,
    RES4LYF_RES_3S,
    RES4LYF_RES_3S_ODE,
    RES4LYF_RES_5S,
    RES4LYF_RES_5S_ODE,
    RES4LYF_RES_6S,
    RES4LYF_RES_6S_ODE,
    RES4LYF_RK_BETA,
)
from .sampler_assembly import (
    Parameterization,
    SamplerDescriptor,
    SamplerInfo,
    SolverStateEvent,
    ordered_descriptors,
)


RES4LYF_SAMPLERS = ordered_descriptors((
    RES4LYF_RES_2M,
    RES4LYF_RES_3M,
    RES4LYF_RES_2S,
    RES4LYF_RES_3S,
    RES4LYF_RES_5S,
    RES4LYF_RES_6S,
    RES4LYF_RES_2M_ODE,
    RES4LYF_RES_3M_ODE,
    RES4LYF_RES_2S_ODE,
    RES4LYF_RES_3S_ODE,
    RES4LYF_RES_5S_ODE,
    RES4LYF_RES_6S_ODE,
    RES4LYF_DEIS_2M,
    RES4LYF_DEIS_3M,
    RES4LYF_DEIS_2M_ODE,
    RES4LYF_DEIS_3M_ODE,
    RES4LYF_RK_BETA,
))
RES4LYF_SAMPLER_NAMES = tuple(
    name for descriptor in RES4LYF_SAMPLERS for name in (descriptor.id, *descriptor.aliases)
)


class _StandardizedGaussianStream:
    def __init__(self, like: torch.Tensor, seed: int) -> None:
        self._shape = tuple(like.shape)
        self._layout = like.layout
        self._device = like.device
        self._generator = torch.Generator(device=like.device)
        self._generator.manual_seed(seed)

    def __call__(self) -> torch.Tensor:
        draw = torch.randn(
            self._shape,
            dtype=torch.float64,
            layout=self._layout,
            device=self._device,
            generator=self._generator,
        )
        return (draw - draw.mean()) / draw.std()


class RES4LYFTwoStreamNoise:
    def __init__(self, like: torch.Tensor, seed: int) -> None:
        self.outer = _StandardizedGaussianStream(like, seed + 1)
        self.substep = _StandardizedGaussianStream(like, seed + 10001)

    @staticmethod
    def _swap_noise(draw: torch.Tensor) -> torch.Tensor:
        centered = draw - draw.mean(dim=(-2, -1), keepdim=True)
        return (centered / centered.std(dim=(-2, -1), keepdim=True)).to(torch.float32)

    def step_noise(self, sigma_from: float, sigma_to: float) -> torch.Tensor:
        return self._swap_noise(self.outer())

    def substep_noise(self, sigma_from: float, sigma_to: float) -> torch.Tensor:
        return self._swap_noise(self.substep())


def _parameterization(sampling: object) -> Parameterization:
    if isinstance(sampling, model_sampling.IMG_TO_IMG_FLOW):
        return Parameterization.IMAGE_TO_IMAGE_FLOW
    if isinstance(sampling, model_sampling.CONST):
        return Parameterization.FLOW
    if isinstance(sampling, model_sampling.EDM):
        return Parameterization.EDM
    if isinstance(sampling, model_sampling.V_PREDICTION):
        return Parameterization.V_PREDICTION
    if isinstance(sampling, model_sampling.X0):
        return Parameterization.X0
    if isinstance(sampling, model_sampling.EPS):
        return Parameterization.EPS
    raise TypeError(f"unsupported RES4LYF model sampling type {type(sampling).__name__}")


def _descriptor(name: str) -> SamplerDescriptor[Any]:
    for descriptor in RES4LYF_SAMPLERS:
        if name == descriptor.id or name in descriptor.aliases:
            return descriptor
    raise ValueError(f"unknown RES4LYF sampler {name!r}")


def sampler_function(name: str, options: Mapping[str, object] | None = None) -> Callable[..., torch.Tensor]:
    descriptor = _descriptor(name)
    solver = descriptor.build(**dict(options or {}))

    def sample(
        model: object,
        x: torch.Tensor,
        sigmas: torch.Tensor,
        extra_args: dict[str, object] | None = None,
        callback: Callable[[dict[str, object]], None] | None = None,
        disable: bool | None = None,
    ) -> torch.Tensor:
        del disable
        args = dict(extra_args or {})
        seed_value = args.get("seed")
        seed = 0 if seed_value is None else int(seed_value)
        sampling = model.inner_model.inner_model.model_sampling  # type: ignore[attr-defined]

        def denoise(value: torch.Tensor, sigma: float) -> torch.Tensor:
            sigma_batch = value.new_full((value.shape[0],), sigma)
            return model(value, sigma_batch, **args)  # type: ignore[operator]

        def on_state(event: SolverStateEvent[object]) -> None:
            if callback is None or event.phase != "post_update":
                return
            callback({
                "x": event.current,
                "i": event.step,
                "sigma": x.new_tensor(event.sigma),
                "sigma_hat": x.new_tensor(event.sigma),
                "denoised": event.denoised,
            })

        info = SamplerInfo(
            parameterization=_parameterization(sampling),
            seed=seed,
            sigma_min=float(sampling.sigma_min),
            sigma_max=float(sampling.sigma_max),
            on_state=on_state,
        )
        noise = RES4LYFTwoStreamNoise(x, seed)
        return solver(denoise, x, tuple(float(sigma) for sigma in sigmas), info, noise=noise)

    sample.__name__ = f"sample_{name}"
    return sample
