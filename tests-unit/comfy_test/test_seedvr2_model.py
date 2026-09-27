"""SeedVR2 model, latent-format, and VAE graph regression tests."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dinkster_comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import dinkster_comfy  # noqa: E402
import dinkster_comfy.latent_formats  # noqa: E402
import dinkster_comfy.ldm.seedvr.model as seedvr_model  # noqa: E402
import dinkster_comfy.ldm.seedvr.vae as seedvr_vae_mod  # noqa: E402
import dinkster_comfy.ops as comfy_ops  # noqa: E402
import dinkster_comfy.sample  # noqa: E402
from dinkster_comfy.ldm.seedvr.model import NaDiT  # noqa: E402


_LATENT_CHANNELS = seedvr_vae_mod.SEEDVR2_LATENT_CHANNELS


def _make_standin(positive_conditioning):
    class _StandIn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "positive_conditioning", positive_conditioning
            )

        _resolve_text_conditioning = NaDiT._resolve_text_conditioning

    return _StandIn()


class _StubModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


def _capture_last_layer_flags(monkeypatch, vid_dim: int, txt_in_dim: int) -> list[bool]:
    flags = []

    class _Block(_StubModule):
        def __init__(self, *args, **kwargs):
            flags.append(kwargs["is_last_layer"])
            super().__init__()

    monkeypatch.setattr(seedvr_model, "NaPatchIn", _StubModule)
    monkeypatch.setattr(seedvr_model, "NaPatchOut", _StubModule)
    monkeypatch.setattr(seedvr_model, "TimeEmbedding", _StubModule)
    monkeypatch.setattr(seedvr_model, "NaMMSRTransformerBlock", _Block)

    seedvr_model.NaDiT(
        norm_eps=1e-5,
        num_layers=4,
        mlp_type="normal",
        vid_dim=vid_dim,
        txt_in_dim=txt_in_dim,
        heads=24,
        mm_layers=3,
        operations=comfy_ops.disable_weight_init,
    )

    return flags


class _Model:
    def __init__(self, latent_format):
        self._latent_format = latent_format

    def get_model_object(self, name):
        assert name == "latent_format"
        return self._latent_format


def test_seedvr2_wrapper_public_encode_returns_tensor(monkeypatch):
    raw_latent = torch.full((1, _LATENT_CHANNELS, 1, 4, 5), 2.0)
    seen_shapes = []

    def base_encode(self, x):
        seen_shapes.append(tuple(x.shape))
        return raw_latent.to(device=x.device, dtype=x.dtype)

    monkeypatch.setattr(seedvr_vae_mod.VideoAutoencoderKL, "encode", base_encode)

    vae = seedvr_vae_mod.VideoAutoencoderKLWrapper.__new__(seedvr_vae_mod.VideoAutoencoderKLWrapper)
    nn.Module.__init__(vae)
    vae._dummy = nn.Parameter(torch.zeros((), dtype=torch.float32))

    latent = vae.encode(torch.zeros(1, 3, 32, 40))

    assert type(latent) is torch.Tensor
    assert tuple(latent.shape) == (1, _LATENT_CHANNELS, 4, 5)
    assert seen_shapes == [(1, 3, 1, 32, 40)]


def test_seedvr2_wrapper_private_encode_helper_keeps_raw_latent(monkeypatch):
    raw_latent = torch.full((1, _LATENT_CHANNELS, 1, 4, 5), 3.0)

    def base_encode(self, x):
        return raw_latent.to(device=x.device, dtype=x.dtype)

    monkeypatch.setattr(seedvr_vae_mod.VideoAutoencoderKL, "encode", base_encode)

    vae = seedvr_vae_mod.VideoAutoencoderKLWrapper.__new__(seedvr_vae_mod.VideoAutoencoderKLWrapper)
    nn.Module.__init__(vae)
    vae._dummy = nn.Parameter(torch.zeros((), dtype=torch.float32))

    latent, raw = vae._encode_with_raw_latent(torch.zeros(1, 3, 32, 40))

    assert tuple(latent.shape) == (1, _LATENT_CHANNELS, 4, 5)
    assert tuple(raw.shape) == (1, _LATENT_CHANNELS, 1, 4, 5)
    assert torch.equal(raw, raw_latent)


def test_missing_context_falls_back_to_positive_buffer():
    pos_buffer = torch.full((58, 5120), 7.0)
    standin = _make_standin(pos_buffer)
    txt, txt_shape = standin._resolve_text_conditioning(None)
    assert txt.shape == (58, 5120)
    assert (txt == 7.0).all(), (
        "fallback path must use the positive_conditioning buffer "
        "verbatim, not a zero tensor"
    )
    assert txt_shape.shape == (1, 1)
    assert txt_shape[0, 0].item() == 58


def test_seedvr2_7b_keeps_final_block_text_path(monkeypatch):
    assert _capture_last_layer_flags(monkeypatch, vid_dim=3072, txt_in_dim=3072) == [
        False,
        False,
        False,
        False,
    ]


def test_seedvr2_7b_rope3d_matches_wrapper_oracle():
    rope = seedvr_model.get_na_rope("rope3d", dim=64)
    generator = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(4, 2, 128, generator=generator)
    k = torch.randn(4, 2, 128, generator=generator)
    shape = torch.tensor([[1, 2, 2]], dtype=torch.long)
    freqs = rope.get_axial_freqs(1, 2, 2).reshape(4, -1)

    expected_q = seedvr_model._apply_seedvr2_rotary_emb(
        freqs,
        q.permute(1, 0, 2).float(),
    ).to(q.dtype).permute(1, 0, 2)
    expected_k = seedvr_model._apply_seedvr2_rotary_emb(
        freqs,
        k.permute(1, 0, 2).float(),
    ).to(k.dtype).permute(1, 0, 2)

    actual_q, actual_k = rope(q.clone(), k.clone(), shape, seedvr_model.Cache(disable=True))

    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)


def test_seedvr2_forward_requires_conditioning_latents():
    model = NaDiT.__new__(NaDiT)
    x = torch.zeros(1, _LATENT_CHANNELS, 1, 4, 5)

    with pytest.raises(ValueError, match="requires conditioning latents"):
        NaDiT.forward(model, x, timestep=torch.tensor([1.0]), context=None)


def test_seedvr2_latent_format_uses_native_video_latent_shape():
    latent_format = dinkster_comfy.latent_formats.SeedVR2()
    latent_image = torch.zeros(1, 1, 4, 5)

    fixed = dinkster_comfy.sample.fix_empty_latent_channels(_Model(latent_format), latent_image)

    assert latent_format.latent_channels == _LATENT_CHANNELS
    assert latent_format.latent_dimensions == 3
    assert fixed.shape == (1, _LATENT_CHANNELS, 1, 4, 5)


def test_seedvr2_model_requires_native_5d_latent():
    latent = torch.zeros(1, _LATENT_CHANNELS, 2, 4, 5)
    assert NaDiT._check_seedvr2_video_latent(latent, _LATENT_CHANNELS, "latent") is latent

    with pytest.raises(ValueError, match="5-D native latent"):
        NaDiT._check_seedvr2_video_latent(torch.zeros(1, _LATENT_CHANNELS * 2, 4, 5), _LATENT_CHANNELS, "latent")
