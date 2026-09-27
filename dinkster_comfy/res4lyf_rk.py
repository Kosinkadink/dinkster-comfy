"""RES4LYF beta RK engine samplers (sample_rk_beta @ 26036f64).

Ports the sixteen named res_*/deis_* sampler wrappers plus the general
rk_beta entry from RES4LYF beta/__init__.py at reference commit
26036f647ca15d3048a193daf99a40cecfc3820d, on their default option path
(guides, automation, frame weights, and the implicit/diag modes are not
ported). Solver math is pure Python float / ArithTensor arithmetic; the
seeded two-stream gaussian noise the engine consumes is produced by the
executing backend behind the :class:`RKNoiseSampler` protocol, declared
on the descriptors as ``NoiseKind.RES4LYF_GAUSSIAN``.

Deviations from the reference, all behavior-preserving on the default
path or out of parity scope:

- Seeding: the reference's seeded path (noise_seed -1 through
  beta/samplers.py) rewrites the outer stream seed to workflow seed + 1
  and derives the substep stream seed as outer + 10000. Dinkster follows
  that contract (outer = run seed + 1, substep = run seed + 10001). The
  direct-wrapper default of ``torch.initial_seed() + 1`` depends on
  process-global RNG state upstream and is unreproducible by
  construction, so it is out of parity scope.
- Phi coefficients use :mod:`decimal` (80 digits) instead of mpmath;
  the resulting float64 tableaus are bit-exact against reference-minted
  goldens.
- The eps_prev_/U/V tableau extensions are dropped (always None for the
  supported types), s_noise is fixed at its default 1.0, and the
  noise-boost / SYNC_SUBSTEP_MEAN_CW machinery is dropped (dead code at
  the default noise_boost 0.0, where h_new_orig == h_new bit-exact).
- The einsum stage accumulation becomes a left-to-right sum that skips
  exact-0.0 coefficients; any reduction-order difference lands in the
  trajectory value-close contract, never in the bit-exact seams.
- Degenerate inputs the reference reserves for its unsampling modes or
  crashes on (schedules starting at sigma 0, missing sigma bounds,
  wrong parameterization, missing noise streams, unknown rk_type or
  options) refuse loudly instead.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal, localcontext
from typing import Any, Protocol, cast, runtime_checkable

from .sampler_assembly import (
    Denoiser,
    DivTensor,
    NoiseKind,
    NoiseSampler,
    OptionKind,
    OptionSpec,
    OptionValue,
    Parameterization,
    SamplerDescriptor,
    SamplerInfo,
    SolverFn,
    SolverStateEvent,
    SubstepEvent,
    StepBeginSolverFn,
    StepCallback,
    StepEvent,
    TensorT,
    TensorT_co,
    is_flow_parameterization,
)

_RK_TYPES = (
    "res_2m",
    "res_3m",
    "res_2s",
    "res_3s",
    "res_5s",
    "res_6s",
    "deis_2m",
    "deis_3m",
)

# Prefixes the reference treats as exponential (logSNR-stepping) methods
# (get_rk_methods_beta @ 26036f64). Of the supported types, res_* are
# exponential and deis_* are linear.
_EXPONENTIAL_PREFIXES = ("res", "dpmpp", "ddim", "pec", "etdrk", "lawson")

# LINEAR_ANCHOR_X_0 / noise_anchor at its default 1.0. The epsilon blend
# keeps the reference's literal a + anchor*(b - a) form because it is not
# bit-equal to plain b in float arithmetic.
_NOISE_ANCHOR = 1.0


def _f32(value: float) -> float:
    """value rounded through IEEE binary32, as the reference does when
    materializing sigmas into float32 tensors."""
    return struct.unpack("!f", struct.pack("!f", value))[0]


# --------------------------------------------------------------------------
# Tableau coefficients (phi_functions / rk_coefficients_beta @ 26036f64)


def _phi_series(j: int, z: float) -> float:
    """phi_j(z) = (exp(z) - sum_{k<j} z^k/k!) / z^j at 80 decimal digits,
    rounded to float64 - bit-exact against the reference's mpmath values
    per the committed goldens."""
    with localcontext() as ctx:
        ctx.prec = 80
        zd = Decimal(z)
        s = Decimal(0)
        for k in range(j):
            s += zd**k / Decimal(math.factorial(k))
        return float((zd.exp() - s) / zd**j)


class _Phi:
    """Cached phi_j(-h*c_i) evaluator (class Phi @ 26036f64)."""

    def __init__(self, h: float, ci: Sequence[float]) -> None:
        self.h = h
        self.ci = list(ci)
        self.cache: dict[tuple[int, int], float] = {}

    def __call__(self, j: int, i: int = -1) -> float:
        key = (j, i)
        if key in self.cache:
            return self.cache[key]
        c = 1.0 if i < 0 else self.ci[i - 1]
        result = 0.0 if c == 0 else _phi_series(j, -self.h * c)
        self.cache[key] = result
        return result


def _gen_first_col_exp(
    a: list[list[float]], b: list[list[float]], ci: Sequence[float], phi: _Phi
) -> tuple[list[list[float]], list[list[float]]]:
    for i in range(len(ci)):
        a[i][0] = ci[i] * phi(1, i + 1) - sum(a[i])
    for i in range(len(b)):
        b[i][0] = phi(1) - sum(b[i])
    return a, b


_RK_COEFF: dict[str, tuple[list[list[float]], list[list[float]], list[float]]] = {
    "ralston_3s": ([[], [1 / 2], [0, 3 / 4]], [[2 / 9, 1 / 3, 4 / 9]], [0, 1 / 2, 3 / 4]),
    "ralston_2s": ([[], [2 / 3]], [[1 / 4, 3 / 4]], [0, 2 / 3]),
    "euler": ([[]], [[1]], [0]),
}


def _deis_coeff_list_rhoab(t_steps: Sequence[float], max_order: int) -> list[list[float]]:
    """DEIS rhoab coefficient lists (get_deis_coeff_list @ 26036f64,
    coeff_mode rhoab / integral form)."""

    def integral_2(a: float, b: float, start: float, end: float, c: float) -> float:
        coeff = (end**3 - start**3) / 3 - (end**2 - start**2) * (a + b) / 2 + (end - start) * a * b
        return coeff / ((c - a) * (c - b))

    out: list[list[float]] = []
    for i in range(len(t_steps) - 1):
        t_cur, t_next = t_steps[i], t_steps[i + 1]
        order = min(i + 1, max_order)
        if order == 1:
            out.append([])
        elif order == 2:
            prev1 = t_steps[i - 1]
            coeff_cur = ((t_next - prev1) ** 2 - (t_cur - prev1) ** 2) / (2 * (t_cur - prev1))
            coeff_prev1 = (t_next - t_cur) ** 2 / (2 * (prev1 - t_cur))
            out.append([coeff_cur, coeff_prev1])
        else:
            prev1, prev2 = t_steps[i - 1], t_steps[i - 2]
            coeff_cur = integral_2(prev1, prev2, t_cur, t_next, t_cur)
            coeff_prev1 = integral_2(t_cur, prev2, t_cur, t_next, prev1)
            coeff_prev2 = integral_2(t_cur, prev1, t_cur, t_next, prev2)
            out.append([coeff_cur, coeff_prev1, coeff_prev2])
    return out


def resolve_rk_tableau(
    rk_type: str,
    h: float,
    *,
    step: int,
    sigmas: Sequence[float],
    sigma: float,
    sigma_next: float,
    sigma_down: float,
    c2: float = 0.5,
    c3: float = 1.0,
) -> tuple[list[list[float]], list[list[float]], list[float], int]:
    """(a, b, ci, multistep_stages) for one step (get_rk_methods_beta
    @ 26036f64), for the supported res_*/deis_* types after their
    warm-up / step-size fallbacks. ``sigmas`` is the prepared schedule;
    ``h`` is the step's eta-free log/linear step size (it scales the phi
    arguments and the DEIS coefficient lists)."""

    multistep_stages = 0
    exponential = rk_type.startswith(_EXPONENTIAL_PREFIXES)

    if exponential:
        h_no_eta = -math.log(sigma_next / sigma)

        def h_prev(k: int) -> float:
            return -math.log(sigmas[step] / sigmas[step - k])

    else:
        h_no_eta = sigma_next - sigma

        def h_prev(k: int) -> float:
            return sigmas[step] - sigmas[step - k]

    if rk_type[:4] == "deis":
        order = int(rk_type[-2])
        if step < order + 1:
            rk_type = {2: "ralston_2s", 3: "ralston_3s"}[order]
        else:
            rk_type = "deis"
            multistep_stages = order - 1

    if rk_type[-2:] == "2m":
        rk_type = rk_type[:-2] + "2s"
        if h_no_eta < 1.0:
            if step >= 2:
                multistep_stages = 1
                c2 = -h_prev(1) / h_no_eta
        else:
            rk_type = "euler" if sigma < 0.1 else "res_2s"

    if rk_type[-2:] == "3m":
        rk_type = rk_type[:-2] + "3s"
        if h_no_eta < 1.0:
            if step >= 3:
                multistep_stages = 2
                c2 = -h_prev(1) / h_no_eta
                c3 = -h_prev(2) / h_no_eta
        else:
            rk_type = "euler" if sigma < 0.1 else "res_3s"

    a: list[list[float]]
    b: list[list[float]]
    ci: list[float]
    if rk_type in _RK_COEFF:
        a_fixed, b_fixed, ci_fixed = _RK_COEFF[rk_type]
        a = [list(row) + [0] * (len(ci_fixed) - len(row)) for row in a_fixed]
        b = [list(row) for row in b_fixed]
        ci = list(ci_fixed)
    elif rk_type == "deis":
        coeff_list = _deis_coeff_list_rhoab(sigmas, multistep_stages + 1)
        scaled = [[elem / h for elem in inner] for inner in coeff_list]
        n = multistep_stages + 1
        a = [[0.0] * n for _ in range(n)]
        b = [list(scaled[step])]
        ci = [0.0] * n
        for i in range(len(b[0])):
            b[0][i] *= (sigma_down - sigma) / (sigma_next - sigma)
    elif rk_type == "res_2s":
        ci = [0, c2]
        phi = _Phi(h, ci)
        a2_1 = c2 * phi(1, 2)
        b2 = phi(2) / c2
        b1 = phi(1) - b2
        a = [[0, 0], [a2_1, 0]]
        b = [[b1, b2]]
    elif rk_type == "res_3s":
        ci = [0, c2, c3]
        phi = _Phi(h, ci)
        gamma = (3 * (c3**3) - 2 * c3) / (c2 * (2 - 3 * c2))
        a3_2 = gamma * c2 * phi(2, 2) + (c3**2 / c2) * phi(2, 3)
        b3 = (1 / (gamma * c2 + c3)) * phi(2)
        b2 = gamma * b3
        a = [[0, 0, 0], [0, 0, 0], [0, a3_2, 0]]
        b = [[0, b2, b3]]
        a, b = _gen_first_col_exp(a, b, ci, phi)
    elif rk_type == "res_5s":
        ci = [0, 1 / 2, 1 / 2, 1, 1 / 2]
        phi = _Phi(h, ci)
        a3_2 = phi(2, 3)
        a4_2 = phi(2, 4)
        a5_2 = (1 / 2) * phi(2, 5) - phi(3, 4) + (1 / 4) * phi(2, 4) - (1 / 2) * phi(3, 5)
        a4_3 = a4_2
        a5_3 = a5_2
        a5_4 = (1 / 4) * phi(2, 5) - a5_2
        b4 = -phi(2) + 4 * phi(3)
        b5 = 4 * phi(2) - 8 * phi(3)
        a = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, a3_2, 0, 0, 0],
            [0, a4_2, a4_3, 0, 0],
            [0, a5_2, a5_3, a5_4, 0],
        ]
        b = [[0, 0, 0, b4, b5]]
        a, b = _gen_first_col_exp(a, b, ci, phi)
    elif rk_type == "res_6s":
        c1_, c2_, c3_, c4_, c5_, c6_ = 0, 1 / 2, 1 / 2, 1 / 3, 1 / 3, 5 / 6
        ci = [c1_, c2_, c3_, c4_, c5_, c6_]
        phi = _Phi(h, ci)
        a3_2 = (c3_**2 / c2_) * phi(2, 3)
        a4_2 = (c4_**2 / c2_) * phi(2, 4)
        a4_3 = (c4_**2 * phi(2, 4) - a4_2 * c2_) / c3_
        a5_2 = 0
        a5_3 = (-c4_ * c5_**2 * phi(2, 5) + 2 * c5_**3 * phi(3, 5)) / (c3_ * (c3_ - c4_))
        a5_4 = (-c3_ * c5_**2 * phi(2, 5) + 2 * c5_**3 * phi(3, 5)) / (c4_ * (c4_ - c3_))
        a6_2 = 0
        a6_3 = (-c4_ * c6_**2 * phi(2, 6) + 2 * c6_**3 * phi(3, 6)) / (c3_ * (c3_ - c4_))
        a6_4 = (-c3_ * c6_**2 * phi(2, 6) + 2 * c6_**3 * phi(3, 6)) / (c4_ * (c4_ - c3_))
        a6_5 = (c6_**2 * phi(2, 6) - a6_3 * c3_ - a6_4 * c4_) / c5_
        b5 = (-c6_ * phi(2) + 2 * phi(3)) / (c5_ * (c5_ - c6_))
        b6 = (-c5_ * phi(2) + 2 * phi(3)) / (c6_ * (c6_ - c5_))
        a = [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, a3_2, 0, 0, 0, 0],
            [0, a4_2, a4_3, 0, 0, 0],
            [0, a5_2, a5_3, a5_4, 0, 0],
            [0, a6_2, a6_3, a6_4, a6_5, 0],
        ]
        b = [[0, 0, 0, 0, b5, b6]]
        a, b = _gen_first_col_exp(a, b, ci, phi)
    else:
        raise ValueError(f"unknown rk_type {rk_type!r}; known: {', '.join(_RK_TYPES)}")

    ci = list(ci)
    ci.append(1)
    return a, b, ci, multistep_stages


# --------------------------------------------------------------------------
# Schedule preparation and SDE coefficients (rk_noise_sampler_beta
# prepare_sigmas / get_sde_coeff hard mode @ 26036f64)


def prepare_rk_sigmas(sigmas: Sequence[float], sigma_min: float) -> list[float]:
    """The reference's in-solver schedule preprocessing on a copy:
    consecutive duplicates removed, and a trailing zero preceded by the
    model's exact sigma_min (overwriting a smaller value, inserting when
    the gap exceeds 1e-4) so the final denoised transition fires."""

    if len(sigmas) < 2:
        raise ValueError("res4lyf samplers need a schedule of at least two sigmas")
    if sigmas[0] == 0.0:
        raise ValueError(
            "res4lyf samplers do not support schedules starting at sigma 0;"
            " the reference reserves that shape for its unsampling modes,"
            " which are not ported"
        )
    out = [float(sigmas[0])]
    for value in sigmas[1:]:
        if value != out[-1]:
            out.append(float(value))
    if out[-1] == 0.0:
        if out[-2] < sigma_min:
            out[-2] = sigma_min
        elif abs(out[-2] - sigma_min) > 1e-4:
            out.insert(-1, sigma_min)
    return out


def _sde_step(
    sigma_next: float, eta: float, *, vp: bool, sigma_max: float
) -> tuple[float, float, float]:
    """(sigma_up, sigma_down, alpha_ratio) for one step or substep
    toward ``sigma_next`` (get_sde_coeff hard mode @ 26036f64). ``vp``
    selects the variance-preserving branch the reference uses for flow
    and CONST models; EPS models take the variance-exploding branch.
    eta is bounded at 0.99 by the option schema, keeping the reference's
    NaN paths at eta >= 1 unreachable."""

    sigma_up = sigma_next * eta
    if vp:
        if sigma_up >= sigma_next:
            sigma_up = sigma_next * 0.9999 if eta >= 1.0 else sigma_next * eta
        signal = sigma_max - sigma_next
        residual = math.sqrt(sigma_next * sigma_next - sigma_up * sigma_up)
        alpha_ratio = signal + residual
        return sigma_up, residual / alpha_ratio, alpha_ratio
    return sigma_up, math.sqrt(sigma_next * sigma_next - sigma_up * sigma_up), 1.0


# --------------------------------------------------------------------------
# Solver


@runtime_checkable
class RKNoiseSampler(Protocol[TensorT_co]):
    """The two seeded gaussian streams the RK engine's noise swaps
    consume, produced by the executing backend for descriptors declaring
    ``NoiseKind.RES4LYF_GAUSSIAN``. Draws arrive fully normalized
    (global standardization at generation, channelwise zscore at the
    draw) as float32. The gaussian streams ignore the sigma bounds; the
    parameters exist for interface symmetry with NoiseSampler."""

    def step_noise(self, sigma_from: float, sigma_to: float) -> TensorT_co: ...

    def substep_noise(self, sigma_from: float, sigma_to: float) -> TensorT_co: ...


def _division_ops(value: TensorT, what: str) -> DivTensor[TensorT]:
    if not isinstance(value, DivTensor):
        raise TypeError(f"{what} requires tensor division")
    return cast("DivTensor[TensorT]", value)


def _zum(
    index: int,
    a: Sequence[Sequence[float]],
    b: Sequence[Sequence[float]],
    k: Sequence[TensorT],
    zero: TensorT,
) -> TensorT:
    """One tableau-row dot product with the stage buffer (zum @ 26036f64:
    row ``index`` of A, or of B past the last A row). Exact-0.0
    coefficients contribute nothing and are skipped."""
    rows = len(a)
    row = a[index] if index < rows else b[index - rows]
    total = zero
    for column, coeff in enumerate(row):
        if coeff == 0.0:
            continue
        total = total + k[column] * coeff
    return total


def _emit_post(
    on_step: StepCallback | None,
    info: SamplerInfo,
    i: int,
    total: int,
    sigma: float,
    current: TensorT,
    denoised: TensorT | None,
) -> None:
    """Per-step events after the noise swap, where the reference invokes
    its callback."""
    if info.on_state is not None:
        info.on_state(
            SolverStateEvent[object](
                step=i,
                total=total,
                sigma=sigma,
                phase="post_update",
                current=current,
                denoised=denoised,
            )
        )
    if on_step is not None:
        on_step(StepEvent(step=i, total=total, sigma=sigma))


def res4lyf_rk(
    *,
    rk_type: str = "res_2m",
    eta: float = 0.5,
    eta_substep: float = 0.5,
) -> StepBeginSolverFn[Any]:
    """The RES4LYF beta RK engine (sample_rk_beta @ 26036f64) on its
    default path: exponential/linear Runge-Kutta stepping with per-step
    tableau resolution, hard-mode SDE noise swaps at step and substep
    level, anchored epsilon reconstruction, overshoot rebound identities,
    and the fixed-point BONGMATH refinement inside its guard."""

    if rk_type not in _RK_TYPES:
        raise ValueError(f"unknown rk_type {rk_type!r}; known: {', '.join(_RK_TYPES)}")
    exponential = rk_type.startswith(_EXPONENTIAL_PREFIXES)
    name = f"res4lyf {rk_type}"

    if exponential:

        def h_fn(sigma_to: float, sigma_from: float) -> float:
            return -math.log(sigma_to / sigma_from)

    else:

        def h_fn(sigma_to: float, sigma_from: float) -> float:
            return sigma_to - sigma_from

    def solve(
        denoiser: Denoiser[TensorT],
        x: TensorT,
        sigmas: Sequence[float],
        info: SamplerInfo,
        *,
        noise: NoiseSampler[TensorT] | None = None,
        on_step: StepCallback | None = None,
        on_step_begin: Callable[[int], None] | None = None,
    ) -> TensorT:
        if len(sigmas) <= 1:
            return x
        if info.parameterization is not Parameterization.EPS and not is_flow_parameterization(
            info.parameterization
        ):
            raise ValueError(
                f"{name} supports only the EPS and flow parameterizations,"
                f" got {info.parameterization.value}"
            )
        vp = is_flow_parameterization(info.parameterization)
        if info.sigma_min is None or info.sigma_max is None:
            raise ValueError(
                f"{name} needs the model's sigma_min and sigma_max carried on"
                " SamplerInfo; the executing engine must populate them"
            )
        sigma_min = info.sigma_min
        sigma_max = info.sigma_max
        rk_noise: RKNoiseSampler[TensorT] | None = None
        if eta > 0.0 or eta_substep > 0.0:
            if not isinstance(noise, RKNoiseSampler):
                raise ValueError(
                    f"{name} with eta > 0 needs the two-stream RK noise sampler"
                    " (NoiseKind.RES4LYF_GAUSSIAN); construct one and pass it"
                    " as noise="
                )
            rk_noise = cast("RKNoiseSampler[TensorT]", noise)
        _division_ops(x, name)

        def div(value: TensorT, divisor: float) -> TensorT:
            # Every tensor here derives from x, whose division capability
            # the entry check above established.
            return cast("DivTensor[TensorT]", value) / divisor

        prep = prepare_rk_sigmas(sigmas, sigma_min)
        num_steps = len(prep) - 2 if prep[-1] == 0.0 else len(prep) - 1
        if num_steps <= 0:
            return x

        zero = x * 0.0
        x_buf: list[TensorT] = []
        eps_buf: list[TensorT] = []
        data_buf: list[TensorT] = []
        data_prev: list[TensorT] = [zero, zero, zero, zero]
        eps_final: TensorT | None = None
        x_cur = x
        evaluation = 0

        for step in range(num_steps):
            sigma = prep[step]
            sigma_next = prep[step + 1]
            if on_step_begin is not None:
                on_step_begin(step)
            _, sd_noeta, _ = _sde_step(sigma_next, 0.0, vp=vp, sigma_max=sigma_max)
            h = h_fn(sd_noeta, sigma)
            su_eta, sd_eta, alpha_eta = _sde_step(sigma_next, eta, vp=vp, sigma_max=sigma_max)
            a, b, ci, ms = resolve_rk_tableau(
                rk_type,
                h,
                step=step,
                sigmas=prep,
                sigma=sigma,
                sigma_next=sigma_next,
                sigma_down=sd_noeta,
            )
            rows = len(a)
            if exponential:
                t = -math.log(sigma)
                s_ = [math.exp(-(t + h * c)) for c in ci]
            else:
                s_ = [sigma + h * c for c in ci]

            while len(x_buf) < rows + 2:
                x_buf.append(zero)
                eps_buf.append(zero)
                data_buf.append(zero)
            x_buf[0] = x_cur
            x_0 = x_cur

            if ms > 0:
                for m in range(min(len(data_prev), len(eps_buf))):
                    if exponential:
                        eps_buf[m] = data_prev[m] - x_0
                    else:
                        eps_buf[m] = div(x_0 - data_prev[m], sigma)

            for row in range(rows - ms):
                sub_sigma = s_[row]
                if sub_sigma == 0.0:
                    break
                sub_sigma_next = s_[row + 1 + ms]
                sub_su_eta = 0.0
                sub_sd_noeta = sub_sigma_next
                sub_sd_eta = sub_sigma_next
                sub_alpha_eta = 1.0
                if row < rows - 1 - ms and sub_sigma_next > 0.0:
                    _, sub_sd_noeta, _ = _sde_step(sub_sigma_next, 0.0, vp=vp, sigma_max=sigma_max)
                    sub_su_eta, sub_sd_eta, sub_alpha_eta = _sde_step(
                        sub_sigma_next, eta_substep, vp=vp, sigma_max=sigma_max
                    )
                denom = h_fn(sub_sigma_next, sigma)
                h_new = h * h_fn(sub_sd_noeta, sigma) / denom if denom != 0.0 else h

                if info.on_substep is not None:
                    info.on_substep(SubstepEvent(step, row, evaluation, sub_sigma))
                evaluation += 1
                denoised = denoiser(x_buf[row], _f32(sub_sigma))
                eps_anchored = div(x_0 - denoised, sigma)
                eps_unmoored = div(x_buf[row] - denoised, sub_sigma)
                blended = eps_unmoored + (eps_anchored - eps_unmoored) * _NOISE_ANCHOR
                if exponential:
                    denoised2 = x_0 - blended * sigma
                    eps_buf[row] = denoised2 - x_0
                    data_buf[row] = denoised2
                else:
                    eps_buf[row] = blended
                    data_buf[row] = denoised

                x_row = x_0 + _zum(row + 1 + ms, a, b, eps_buf, zero) * h_new
                if sigma - sub_sd_noeta > 0.0:
                    sub_eps = div(x_0 - x_row, sigma - sub_sd_noeta)
                    x_row = (x_0 - sub_eps * sigma) + sub_eps * sub_sigma_next
                if sub_su_eta != 0.0 and sub_sigma_next != 0.0:
                    bs = sub_sigma
                    bsn = sub_sigma_next
                    if bs == bsn:
                        bsn = bsn * 0.999
                    eps_next = div(x_0 - x_row, sigma - sub_sigma_next)
                    denoised_next = x_0 - eps_next * sigma
                    if bsn > bs:
                        bs, bsn = bsn, bs
                    assert rk_noise is not None  # eta_substep > 0 was narrowed at entry
                    noise_draw = rk_noise.substep_noise(bs, bsn)
                    x_row = (
                        denoised_next + eps_next * sub_sd_eta
                    ) * sub_alpha_eta + noise_draw * sub_su_eta
                x_buf[row + 1] = x_row

                if (
                    sub_sigma > sigma_min
                    and h < sigma_max / 2
                    and sigma > 0.03
                    and row < rows - 1
                    and ms == 0
                ):
                    # BONGMATH fixed-point refinement (bong_iter
                    # @ 26036f64, default strength 1.0). The x_0
                    # rebinding deliberately persists into later rows
                    # and the epilogue, as in the reference.
                    for _ in range(100):
                        x_0 = x_buf[row + 1] - _zum(row + 1, a, b, eps_buf, zero) * h
                        for rr in range(row + 1):
                            x_buf[rr] = x_0 + _zum(rr, a, b, eps_buf, zero) * h
                        for rr in range(row + 1):
                            e_anchored = div(x_0 - data_buf[rr], sigma)
                            e_unmoored = div(x_buf[rr] - data_buf[rr], s_[rr])
                            e_blend = e_unmoored + (e_anchored - e_unmoored) * _NOISE_ANCHOR
                            if exponential:
                                den2 = x_0 - e_blend * sigma
                                eps_buf[rr] = den2 - x_0
                            else:
                                eps_buf[rr] = e_blend

            x_next = x_buf[rows - ms]
            # rebound_overshoot_step @ 26036f64, unconditional: an exact
            # identity when overshoot is 0 (sd_noeta lands at sigma_next
            # up to float rounding), executed anyway for parity.
            reb_eps = div(x_0 - x_next, sigma - sd_noeta)
            x_next = (x_0 - reb_eps * sigma) + reb_eps * sigma_next
            eps_final = div(x_0 - x_next, sigma - sigma_next)
            denoised_final = x_0 - eps_final * sigma
            if eta == 0.0 or su_eta == 0.0 or sigma_next == 0.0:
                x_cur = x_next
            else:
                # swap_noise_step @ 26036f64; its eps/denoised recompute
                # is bit-identical to eps_final/denoised_final above.
                bs = sigma
                bsn = sigma_next
                if bs == bsn:
                    bsn = bsn * 0.999
                if bsn > bs:
                    bs, bsn = bsn, bs
                assert rk_noise is not None  # eta > 0 was narrowed at entry
                noise_draw = rk_noise.step_noise(bs, bsn)
                x_cur = (denoised_final + eps_final * sd_eta) * alpha_eta + noise_draw * su_eta
            _emit_post(on_step, info, step, num_steps, sigma, x_cur, denoised_final)
            data_prev[0] = data_buf[0]
            for m in range(3):
                data_prev[3 - m] = data_prev[2 - m]

        if prep[-1] == 0.0 and prep[-2] == sigma_min and eps_final is not None:
            # The final denoised transition: the reference's last loop
            # iteration breaks before its model call and applies the
            # pre-swap eps_final at sigma_min instead.
            x_cur = x_cur - eps_final * sigma_min
        return x_cur

    return solve


# --------------------------------------------------------------------------
# Descriptors (wrapper names, aliases, and order per beta/__init__.py
# @ 26036f64; rk_beta exposes the engine's type/eta options directly)


def _float_option(opts: Mapping[str, OptionValue], option: str) -> float:
    value = opts[option]
    assert isinstance(value, float), f"option {option!r} resolved to non-float {value!r}"
    return value


def _make_res_2m(opts: Mapping[str, OptionValue]) -> SolverFn[Any]:
    del opts
    return res4lyf_rk(rk_type="res_2m")


def _make_res_3m(opts: Mapping[str, OptionValue]) -> SolverFn[Any]:
    del opts
    return res4lyf_rk(rk_type="res_3m")


def _make_res_2s(opts: Mapping[str, OptionValue]) -> SolverFn[Any]:
    del opts
    return res4lyf_rk(rk_type="res_2s")


def _make_res_3s(opts: Mapping[str, OptionValue]) -> SolverFn[Any]:
    del opts
    return res4lyf_rk(rk_type="res_3s")


def _make_res_5s(opts: Mapping[str, OptionValue]) -> SolverFn[Any]:
    del opts
    return res4lyf_rk(rk_type="res_5s")


def _make_res_6s(opts: Mapping[str, OptionValue]) -> SolverFn[Any]:
    del opts
    return res4lyf_rk(rk_type="res_6s")


def _make_res_2m_ode(opts: Mapping[str, OptionValue]) -> SolverFn[Any]:
    del opts
    return res4lyf_rk(rk_type="res_2m", eta=0.0, eta_substep=0.0)


def _make_res_3m_ode(opts: Mapping[str, OptionValue]) -> SolverFn[Any]:
    del opts
    return res4lyf_rk(rk_type="res_3m", eta=0.0, eta_substep=0.0)


def _make_res_2s_ode(opts: Mapping[str, OptionValue]) -> SolverFn[Any]:
    del opts
    return res4lyf_rk(rk_type="res_2s", eta=0.0, eta_substep=0.0)


def _make_res_3s_ode(opts: Mapping[str, OptionValue]) -> SolverFn[Any]:
    del opts
    return res4lyf_rk(rk_type="res_3s", eta=0.0, eta_substep=0.0)


def _make_res_5s_ode(opts: Mapping[str, OptionValue]) -> SolverFn[Any]:
    del opts
    return res4lyf_rk(rk_type="res_5s", eta=0.0, eta_substep=0.0)


def _make_res_6s_ode(opts: Mapping[str, OptionValue]) -> SolverFn[Any]:
    del opts
    return res4lyf_rk(rk_type="res_6s", eta=0.0, eta_substep=0.0)


def _make_deis_2m(opts: Mapping[str, OptionValue]) -> SolverFn[Any]:
    del opts
    return res4lyf_rk(rk_type="deis_2m")


def _make_deis_3m(opts: Mapping[str, OptionValue]) -> SolverFn[Any]:
    del opts
    return res4lyf_rk(rk_type="deis_3m")


def _make_deis_2m_ode(opts: Mapping[str, OptionValue]) -> SolverFn[Any]:
    del opts
    return res4lyf_rk(rk_type="deis_2m", eta=0.0, eta_substep=0.0)


def _make_deis_3m_ode(opts: Mapping[str, OptionValue]) -> SolverFn[Any]:
    del opts
    return res4lyf_rk(rk_type="deis_3m", eta=0.0, eta_substep=0.0)


def _make_rk_beta(opts: Mapping[str, OptionValue]) -> SolverFn[Any]:
    rk_type = opts["rk_type"]
    assert isinstance(rk_type, str)
    return res4lyf_rk(
        rk_type=rk_type,
        eta=_float_option(opts, "eta"),
        eta_substep=_float_option(opts, "eta_substep"),
    )


_RK_BETA_OPTIONS = (
    OptionSpec(
        "rk_type",
        OptionKind.CHOICE,
        "res_2m",
        choices=_RK_TYPES,
        doc="Runge-Kutta method family",
    ),
    OptionSpec(
        "eta",
        OptionKind.FLOAT,
        0.5,
        minimum=0.0,
        maximum=0.99,
        doc="step SDE noise fraction (the reference NaNs at eta >= 1 on EPS models)",
    ),
    OptionSpec(
        "eta_substep",
        OptionKind.FLOAT,
        0.5,
        minimum=0.0,
        maximum=0.99,
        doc="substep SDE noise fraction",
    ),
)

RES4LYF_RES_2M: SamplerDescriptor[Any] = SamplerDescriptor(
    id="res4lyf.res_2m",
    display_name="RES 2M",
    make=_make_res_2m,
    noise=NoiseKind.RES4LYF_GAUSSIAN,
    aliases=("res_2m",),
    supports_step_begin=True,
)

RES4LYF_RES_3M: SamplerDescriptor[Any] = SamplerDescriptor(
    id="res4lyf.res_3m",
    display_name="RES 3M",
    make=_make_res_3m,
    noise=NoiseKind.RES4LYF_GAUSSIAN,
    aliases=("res_3m",),
    supports_step_begin=True,
)

RES4LYF_RES_2S: SamplerDescriptor[Any] = SamplerDescriptor(
    id="res4lyf.res_2s",
    display_name="RES 2S",
    make=_make_res_2s,
    noise=NoiseKind.RES4LYF_GAUSSIAN,
    aliases=("res_2s",),
    supports_step_begin=True,
)

RES4LYF_RES_3S: SamplerDescriptor[Any] = SamplerDescriptor(
    id="res4lyf.res_3s",
    display_name="RES 3S",
    make=_make_res_3s,
    noise=NoiseKind.RES4LYF_GAUSSIAN,
    aliases=("res_3s",),
    supports_step_begin=True,
)

RES4LYF_RES_5S: SamplerDescriptor[Any] = SamplerDescriptor(
    id="res4lyf.res_5s",
    display_name="RES 5S",
    make=_make_res_5s,
    noise=NoiseKind.RES4LYF_GAUSSIAN,
    aliases=("res_5s",),
    supports_step_begin=True,
)

RES4LYF_RES_6S: SamplerDescriptor[Any] = SamplerDescriptor(
    id="res4lyf.res_6s",
    display_name="RES 6S",
    make=_make_res_6s,
    noise=NoiseKind.RES4LYF_GAUSSIAN,
    aliases=("res_6s",),
    supports_step_begin=True,
)

RES4LYF_RES_2M_ODE: SamplerDescriptor[Any] = SamplerDescriptor(
    id="res4lyf.res_2m_ode",
    display_name="RES 2M ODE",
    make=_make_res_2m_ode,
    # "rk" is the retired legacy-module sampler name (legacy/__init__.py
    # @ 26036f64). Workflows reach it only as a bare name, so it always ran
    # sample_rk's library defaults: rk_type "res_2m", eta 0.0 - a
    # deterministic res_2m trajectory. This descriptor is the beta engine's
    # deterministic res_2m; outputs differ from the retired legacy engine.
    # The sibling legacy name "legacy_rk" is intentionally unmapped: its
    # engine's semantics (terminal buehler swap, per-step c2/c3 feedback)
    # have no beta equivalent, so it refuses as an unknown sampler.
    aliases=("res_2m_ode", "rk"),
    supports_step_begin=True,
)

RES4LYF_RES_3M_ODE: SamplerDescriptor[Any] = SamplerDescriptor(
    id="res4lyf.res_3m_ode",
    display_name="RES 3M ODE",
    make=_make_res_3m_ode,
    aliases=("res_3m_ode",),
    supports_step_begin=True,
)

RES4LYF_RES_2S_ODE: SamplerDescriptor[Any] = SamplerDescriptor(
    id="res4lyf.res_2s_ode",
    display_name="RES 2S ODE",
    make=_make_res_2s_ode,
    aliases=("res_2s_ode",),
    supports_step_begin=True,
)

RES4LYF_RES_3S_ODE: SamplerDescriptor[Any] = SamplerDescriptor(
    id="res4lyf.res_3s_ode",
    display_name="RES 3S ODE",
    make=_make_res_3s_ode,
    aliases=("res_3s_ode",),
    supports_step_begin=True,
)

RES4LYF_RES_5S_ODE: SamplerDescriptor[Any] = SamplerDescriptor(
    id="res4lyf.res_5s_ode",
    display_name="RES 5S ODE",
    make=_make_res_5s_ode,
    aliases=("res_5s_ode",),
    supports_step_begin=True,
)

RES4LYF_RES_6S_ODE: SamplerDescriptor[Any] = SamplerDescriptor(
    id="res4lyf.res_6s_ode",
    display_name="RES 6S ODE",
    make=_make_res_6s_ode,
    aliases=("res_6s_ode",),
    supports_step_begin=True,
)

RES4LYF_DEIS_2M: SamplerDescriptor[Any] = SamplerDescriptor(
    id="res4lyf.deis_2m",
    display_name="DEIS 2M",
    make=_make_deis_2m,
    noise=NoiseKind.RES4LYF_GAUSSIAN,
    aliases=("deis_2m",),
    supports_step_begin=True,
)

RES4LYF_DEIS_3M: SamplerDescriptor[Any] = SamplerDescriptor(
    id="res4lyf.deis_3m",
    display_name="DEIS 3M",
    make=_make_deis_3m,
    noise=NoiseKind.RES4LYF_GAUSSIAN,
    aliases=("deis_3m",),
    supports_step_begin=True,
)

RES4LYF_DEIS_2M_ODE: SamplerDescriptor[Any] = SamplerDescriptor(
    id="res4lyf.deis_2m_ode",
    display_name="DEIS 2M ODE",
    make=_make_deis_2m_ode,
    aliases=("deis_2m_ode",),
    supports_step_begin=True,
)

RES4LYF_DEIS_3M_ODE: SamplerDescriptor[Any] = SamplerDescriptor(
    id="res4lyf.deis_3m_ode",
    display_name="DEIS 3M ODE",
    make=_make_deis_3m_ode,
    aliases=("deis_3m_ode",),
    supports_step_begin=True,
)

RES4LYF_RK_BETA: SamplerDescriptor[Any] = SamplerDescriptor(
    id="res4lyf.rk_beta",
    display_name="RK beta",
    make=_make_rk_beta,
    options=_RK_BETA_OPTIONS,
    noise=NoiseKind.RES4LYF_GAUSSIAN,
    aliases=("rk_beta",),
    supports_step_begin=True,
)
