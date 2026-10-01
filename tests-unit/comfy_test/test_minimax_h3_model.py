import pytest
import torch
from torch import nn

from dinkster_inference.ldm.minimax.model import (
    MiniMaxH3Model,
    _mod_gate,
    _mod_scale_shift,
    time_shift_sigma,
)
from dinkster_inference.model_sampling import CONST


def make_model(video_output, audio_output):
    model = MiniMaxH3Model.__new__(MiniMaxH3Model)
    nn.Module.__init__(model)
    model.sigma_shift_video = 12.0
    model.sigma_shift_audio = 3.0
    model._forward = lambda *args, **kwargs: [
        video_output.clone(),
        audio_output.clone(),
    ]
    return model


def test_forward_scales_velocity_to_mask_timestep():
    video_output = torch.full((1, 2, 1, 2, 2), 2.0)
    audio_output = torch.full((1, 2, 2, 3), 3.0)
    video_mask = torch.tensor([[[[[1.0, 0.75], [0.5, 0.25]]]]])
    audio_mask = torch.tensor([[[[1.0, 0.5, 0.25], [0.75, 0.5, 0.0]]]])
    sigma = torch.tensor([0.5])
    clean = torch.arange(video_output.numel(), dtype=torch.float32).reshape_as(
        video_output
    )
    model_input = clean + sigma.reshape(1, 1, 1, 1, 1) * video_mask * video_output
    model = make_model(video_output, audio_output)

    out = model(
        [model_input, torch.zeros_like(audio_output)],
        sigma * 1000.0,
        torch.empty(1, 1, 1),
        minimax_payload={"audio_scale": 1.0},
        denoise_mask=video_mask,
        audio_denoise_mask=audio_mask,
    )

    torch.testing.assert_close(out[0], video_output * video_mask)
    torch.testing.assert_close(out[1], audio_output * audio_mask)
    denoised = CONST.calculate_denoised(None, sigma, out[0], model_input)
    torch.testing.assert_close(denoised, clean)


def test_forward_scales_audio_velocity_before_carry_conversion():
    video_output = torch.ones((1, 1, 1, 1, 1))
    audio_output = torch.full((1, 1, 2, 2), 3.0)
    audio_src = torch.full_like(audio_output, 2.0)
    audio_mask = torch.tensor([[[[0.75, 0.5], [0.25, 0.0]]]])
    model = make_model(video_output, audio_output)
    sigma_v = torch.tensor(0.5)
    sigma_a = time_shift_sigma(sigma_v, 12.0, 3.0)
    carry = sigma_a / sigma_v

    out = model(
        [torch.zeros_like(video_output), audio_src],
        sigma_v.reshape(1) * 1000.0,
        torch.empty(1, 1, 1),
        minimax_payload={"audio_scale": 4.0},
        audio_denoise_mask=audio_mask,
    )

    expected = (
        -3.0 * audio_src * carry + (1.0 + 3.0 * sigma_a) * audio_output * audio_mask
    )
    torch.testing.assert_close(out[1], expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_indexed_bf16_modulation_is_bit_exact_to_segment_operations():
    generator = torch.Generator(device="cuda").manual_seed(484)
    rows, hidden = 19, 96
    indices = torch.tensor(
        [2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 0, 2, 1, 0, 2, 1, 1, 0, 2],
        device="cuda",
    )
    segments = [(0, rows, indices)]
    modulation = torch.randn(
        (3, hidden * 3), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    shift, scale, gate = modulation.chunk(3, dim=1)
    source = torch.randn(
        (rows, hidden), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    other = torch.randn(
        (rows, hidden), device="cuda", dtype=torch.bfloat16, generator=generator
    )

    expected_scale_shift = _mod_scale_shift(source.clone(), shift, scale, segments)
    actual_scale_shift = _mod_scale_shift(
        source.clone(), shift, scale, segments, indices
    )
    expected_gate = _mod_gate(source.clone(), gate, other, segments)
    actual_gate = _mod_gate(source.clone(), gate, other, segments, indices)

    assert torch.equal(actual_scale_shift, expected_scale_shift)
    assert torch.equal(actual_gate, expected_gate)
