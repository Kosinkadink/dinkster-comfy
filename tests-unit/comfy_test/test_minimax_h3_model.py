import pytest
import torch
from torch import nn

from dinkster_inference.ldm.minimax.model import (
    MiniMaxH3Model,
    _cache_dit_key,
    _run_cache_dit_blocks,
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


def cache_config(**overrides):
    config = {
        "model_identity": "model:patches",
        "policy": "quality",
        "Fn_compute_blocks": 1,
        "max_warmup_steps": 1,
        "residual_diff_threshold": 0.04,
        "max_continuous_cached_steps": 1,
    }
    config.update(overrides)
    return config


def cache_runtime():
    return {
        "key": None,
        "key_fields": None,
        "state": None,
        "hits": 0,
        "misses": 0,
        "invalidations": 0,
        "events": [],
    }


def run_blocks(value, config, runtime, key, step, calls):
    increments = (1.0, 2.0, 3.0)

    def run_block(index, hidden):
        calls.append(index)
        return hidden.add(increments[index])

    return _run_cache_dit_blocks(
        value,
        run_block,
        len(increments),
        config,
        runtime,
        key,
        {"key": key},
        step,
    )


def test_cache_dit_reuses_joint_middle_residual_after_warmup():
    config = cache_config()
    runtime = cache_runtime()
    calls = []

    first = run_blocks(torch.zeros(2), config, runtime, "same", 0, calls)
    second = run_blocks(torch.zeros(2), config, runtime, "same", 1, calls)

    torch.testing.assert_close(first, torch.full((2,), 6.0))
    torch.testing.assert_close(second, first)
    assert calls == [0, 1, 2, 0]
    assert runtime["hits"] == 1
    assert runtime["misses"] == 1
    assert runtime["events"][1]["computed_blocks"] == [0]
    assert runtime["events"][1]["skipped_blocks"] == [1, 2]


def test_cache_dit_forces_full_compute_after_consecutive_hit_limit():
    config = cache_config()
    runtime = cache_runtime()
    calls = []

    for step in range(3):
        run_blocks(torch.zeros(1), config, runtime, "same", step, calls)

    assert calls == [0, 1, 2, 0, 0, 1, 2]
    assert [event["cache_hit"] for event in runtime["events"]] == [False, True, False]


def test_cache_dit_key_change_invalidates_audio_and_video_together():
    config = cache_config()
    runtime = cache_runtime()
    calls = []

    run_blocks(torch.zeros(1), config, runtime, "first", 0, calls)
    run_blocks(torch.zeros(1), config, runtime, "first", 1, calls)
    run_blocks(torch.zeros(1), config, runtime, "changed-audio-geometry", 2, calls)

    assert calls[-3:] == [0, 1, 2]
    assert runtime["invalidations"] == 1
    assert runtime["events"][-1]["cache_hit"] is False


def test_cache_dit_key_covers_exact_schedule_conditioning_layout_and_block_span():
    class Layout:
        signature = (3, 4, 6, 8, 10)
        segments = ((0, 3, "text"), (3, 23, "audio"), (23, 119, "video"))

    config = cache_config()
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    context = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)

    key, fields = _cache_dit_key(config, sigmas, context, Layout(), 4)
    changed_sigmas, _ = _cache_dit_key(
        config, torch.tensor([1.0, 0.49, 0.0]), context, Layout(), 4
    )
    changed_context, _ = _cache_dit_key(config, sigmas, context + 1, Layout(), 4)

    assert key != changed_sigmas
    assert key != changed_context
    assert fields["model"] == "model:patches"
    assert fields["layout"] == (Layout.signature, Layout.segments)
    assert fields["blocks"] == (1, 2, 3)
    assert fields["segments"] == (Layout.segments[1], Layout.segments[2])


def test_cache_dit_rejects_malformed_internal_policy():
    config = cache_config(residual_diff_threshold=1.0)

    with pytest.raises(ValueError, match="threshold"):
        _cache_dit_key(
            config,
            torch.tensor([1.0, 0.0]),
            torch.zeros(1, 1, 1),
            type(
                "Layout",
                (),
                {
                    "signature": (1, 1, 1, 1, 1),
                    "segments": ((0, 1, "audio"), (1, 2, "video")),
                },
            )(),
            2,
        )


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
