from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from dinkster_comfy import model_sampling, samplers
from dinkster_comfy.k_diffusion import sampling as k_diffusion_sampling
from dinkster_comfy.res4lyf_rk import _sde_step, prepare_rk_sigmas, resolve_rk_tableau
from dinkster_comfy.res4lyf_sampler import RES4LYFTwoStreamNoise, RES4LYF_SAMPLERS, sampler_function
from dinkster_comfy.sampler_assembly import Parameterization, SamplerInfo, SolverStateEvent, SubstepEvent


GOLDEN = json.loads((Path(__file__).parent / "goldens/res4lyf_rk_goldens.json").read_text())
TRAJECTORY_ATOL = 2e-5


def _tensor(values: list[float], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype).reshape(1, 1, 2, 2)


class _Model:
    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []
        self.sigmas: list[float] = []

    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        self.calls.append(x.detach().clone())
        self.sigmas.append(sigma)
        return x * (1.0 / (1.0 + sigma)) + x.square() * (0.05 / (1.0 + sigma))


class _RecordingStream:
    def __init__(self, inner: Any, tag: int, log: list[tuple[int, torch.Tensor]]) -> None:
        self._inner = inner
        self._tag = tag
        self._log = log

    def __call__(self) -> torch.Tensor:
        draw = self._inner()
        self._log.append((self._tag, draw.detach().clone()))
        return draw


class _ComfyModel:
    def __init__(self, sampling: object) -> None:
        sampling.sigma_min = 0.01
        sampling.sigma_max = 1.0
        self.inner_model = type("Inner", (), {"inner_model": type("Base", (), {"model_sampling": sampling})()})()
        self.sigma_batches: list[torch.Tensor] = []

    def __call__(self, value: torch.Tensor, sigma: torch.Tensor, **extra_args: object) -> torch.Tensor:
        assert extra_args["seed"] == 29
        self.sigma_batches.append(sigma)
        return value / (1.0 + sigma.reshape((-1,) + (1,) * (value.ndim - 1)))


def _case_setup(name: str) -> tuple[str, dict[str, object], str, Parameterization]:
    parameterization = Parameterization.FLOW if "_const_" in name else Parameterization.EPS
    if name.startswith("rk_beta_"):
        rk_type = name[len("rk_beta_") :].rsplit("_", 1)[0]
        return "res4lyf.rk_beta", {"rk_type": rk_type}, rk_type, parameterization
    rk_type = name[:7] if name.startswith("deis") else name[:6]
    sampler = f"res4lyf.{rk_type}_ode" if "_ode_" in name else f"res4lyf.{rk_type}"
    return sampler, {}, rk_type, parameterization


def _descriptor(name: str):
    return next(item for item in RES4LYF_SAMPLERS if name == item.id or name in item.aliases)


def _assert_values(actual: list[float], expected: list[float], atol: float = 1e-12) -> None:
    assert actual == pytest.approx(expected, abs=atol, rel=0.0)


@pytest.mark.parametrize("case_name", sorted(GOLDEN["cases"]))
def test_rk_engine_replays_recorded_reference(case_name: str) -> None:
    case: dict[str, Any] = GOLDEN["cases"][case_name]
    sampler_id, options, rk_type, parameterization = _case_setup(case_name)
    vp = parameterization is Parameterization.FLOW
    exponential = rk_type.startswith("res")
    sigma_min: float = case["sigma_min"]
    sigma_max: float = case["sigma_max"]
    seed: int = GOLDEN["_meta"]["seed"]

    prep = prepare_rk_sigmas(case["schedule_sigmas"], sigma_min)
    _assert_values(prep, case["prepared_sigmas"])
    for step, entry in enumerate(case["coeffs"]):
        sigma = prep[step]
        sigma_next = prep[step + 1]
        _, sigma_down, _ = _sde_step(sigma_next, 0.0, vp=vp, sigma_max=sigma_max)
        h = -math.log(sigma_down / sigma) if exponential else sigma_down - sigma
        a, b, c, multistep_stages = resolve_rk_tableau(
            rk_type,
            h,
            step=step,
            sigmas=prep,
            sigma=sigma,
            sigma_next=sigma_next,
            sigma_down=sigma_down,
        )
        for actual, expected in zip(a, entry["a"], strict=True):
            _assert_values(actual, expected)
        for actual, expected in zip(b, entry["b"], strict=True):
            _assert_values(actual, expected)
        _assert_values(c, entry["c"])
        assert multistep_stages == entry["multistep_stages"]

    model = _Model()
    initial = _tensor(GOLDEN["initial"])
    noise = RES4LYFTwoStreamNoise(initial, seed)
    draws: list[tuple[int, torch.Tensor]] = []
    noise.outer = _RecordingStream(noise.outer, 0, draws)
    noise.substep = _RecordingStream(noise.substep, 1, draws)
    states: list[SolverStateEvent[object]] = []
    substeps: list[SubstepEvent] = []
    result = _descriptor(sampler_id).build(**options)(
        model,
        initial,
        case["schedule_sigmas"],
        SamplerInfo(
            parameterization,
            seed=seed,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            on_state=states.append,
            on_substep=substeps.append,
        ),
        noise=noise,
    )

    _assert_values(model.sigmas, case["model_call_sigmas"], atol=1e-7)
    assert [event.evaluation for event in substeps] == list(range(len(substeps)))
    assert [event.sigma for event in substeps] == pytest.approx(model.sigmas, abs=5e-7, rel=0.0)
    assert [event.outer_step for event in substeps] == sorted(event.outer_step for event in substeps)
    for (tag, draw), expected in zip(draws, case["noise_draws"], strict=True):
        assert tag == expected["stream"]
        assert draw.dtype is torch.float64
        assert torch.allclose(draw, _tensor(expected["values"], torch.float64), atol=1e-12, rtol=0.0)

    atol = TRAJECTORY_ATOL if case["noise_draws"] else 0.0
    for actual, expected in zip(model.calls, case["model_calls"], strict=True):
        assert torch.allclose(actual, _tensor(expected), atol=atol, rtol=0.0)
    assert torch.allclose(result, _tensor(case["final"]), atol=atol, rtol=0.0)


def test_sampler_descriptors_cover_receipt_backed_res4lyf_names() -> None:
    expected = {
        "res_2m", "res_3m", "res_2s", "res_3s", "res_5s", "res_6s",
        "res_2m_ode", "res_3m_ode", "res_2s_ode", "res_3s_ode", "res_5s_ode", "res_6s_ode",
        "deis_2m", "deis_3m", "deis_2m_ode", "deis_3m_ode", "rk_beta",
    }
    assert {descriptor.aliases[0] for descriptor in RES4LYF_SAMPLERS} == expected
    assert {descriptor.id for descriptor in RES4LYF_SAMPLERS} == {f"res4lyf.{name}" for name in expected}


def test_sampler_options_refuse_unknown_or_out_of_range_values() -> None:
    descriptor = _descriptor("rk_beta")
    with pytest.raises(ValueError, match="unknown sampler options"):
        descriptor.build(unknown=True)
    with pytest.raises(ValueError, match="at most 0.99"):
        descriptor.build(eta=1.0)


def test_comfy_adapter_owns_seed_sigma_dtype_and_step_callbacks() -> None:
    initial = _tensor(GOLDEN["initial"])
    sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
    callback_events: list[dict[str, object]] = []

    model = _ComfyModel(model_sampling.EPS())
    sample = sampler_function("res_2m")
    result = sample(model, initial, sigmas, extra_args={"seed": 29}, callback=callback_events.append)
    repeated = sample(_ComfyModel(model_sampling.EPS()), initial, sigmas, extra_args={"seed": 29})

    assert torch.equal(result, repeated)
    assert model.sigma_batches
    assert all(batch.dtype is initial.dtype and batch.shape == (initial.shape[0],) for batch in model.sigma_batches)
    assert [event["i"] for event in callback_events] == [0, 1]
    assert all(event["sigma"].dtype is initial.dtype for event in callback_events)


def test_comfy_adapter_refuses_v_prediction_instead_of_treating_it_as_eps() -> None:
    sample = sampler_function("res_2m")
    with pytest.raises(ValueError, match="supports only the EPS and flow parameterizations"):
        sample(
            _ComfyModel(model_sampling.V_PREDICTION()),
            _tensor(GOLDEN["initial"]),
            torch.tensor([1.0, 0.0]),
            extra_args={"seed": 29},
        )


def test_ordinary_sampler_assembly_still_uses_stock_function() -> None:
    assembled = samplers.sampler_object("euler")
    assert assembled.sampler_function is k_diffusion_sampling.sample_euler


def test_receipt_backed_namespaced_sampler_assembles() -> None:
    assembled = samplers.sampler_object("res4lyf.res_2m")
    assert assembled.sampler_function.__name__ == "sample_res4lyf.res_2m"
