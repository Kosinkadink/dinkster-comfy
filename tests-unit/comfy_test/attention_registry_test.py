import logging

import pytest
import torch

from dinkster_inference.cli_args import args

args.cpu = True

from dinkster_inference.ldm.modules import attention


def test_attention_registries_are_independently_owned():
    first = attention.create_attention_function_registry()
    second = attention.create_attention_function_registry()
    first_kernel = lambda: "first"
    second_kernel = lambda: "second"

    attention.register_attention_function("worker", first_kernel, registry=first)
    attention.register_attention_function("worker", second_kernel, registry=second)

    assert first is not second
    assert attention.get_attention_function("worker", registry=first) is first_kernel
    assert attention.get_attention_function("worker", registry=second) is second_kernel
    with pytest.raises(KeyError, match="Attention function worker not found"):
        attention.get_attention_function("worker")


def test_explicit_registry_does_not_fall_back_to_default(monkeypatch):
    default_kernel = lambda: "default"
    monkeypatch.setitem(
        attention.REGISTERED_ATTENTION_FUNCTIONS, "plugin", default_kernel
    )

    registry = attention.create_attention_function_registry()

    assert attention.get_attention_function("plugin") is default_kernel
    assert attention.get_attention_function("plugin", None, registry=registry) is None
    assert "plugin" not in registry


def test_registry_preserves_duplicate_and_optimized_behavior(caplog):
    registry = attention.create_attention_function_registry()
    original = lambda: "original"
    replacement = lambda: "replacement"
    attention.register_attention_function("worker", original, registry=registry)

    with caplog.at_level(logging.WARNING):
        attention.register_attention_function("worker", replacement, registry=registry)

    assert attention.get_attention_function("worker", registry=registry) is original
    assert "already registered" in caplog.text
    registry["optimized"] = replacement
    assert (
        attention.get_attention_function("optimized", registry=registry)
        is attention.optimized_attention
    )


def test_registry_selection_dispatches_through_attention_override():
    registry = attention.create_attention_function_registry()

    def worker_kernel(q, k, v, heads, **kwargs):
        return (q, k, v, heads, kwargs["marker"])

    attention.register_attention_function("worker", worker_kernel, registry=registry)
    selected = attention.get_attention_function("worker", registry=registry)

    def override(_, *args, **kwargs):
        return selected(*args, **kwargs)

    @attention.wrap_attn
    def default_kernel(*args, **kwargs):
        raise AssertionError((args, kwargs))

    result = default_kernel(
        "q",
        "k",
        "v",
        8,
        marker="worker",
        transformer_options={"optimized_attention_override": override},
    )

    assert result == ("q", "k", "v", 8, "worker")


def test_flash4_sm120_dense_preserves_attention_layout_and_scale(monkeypatch):
    calls = []

    def flash4(q, k, v, **kwargs):
        calls.append((q.shape, k.shape, v.shape, kwargs))
        return q + 1

    monkeypatch.setattr(attention, "flash_attn4_func", flash4, raising=False)
    monkeypatch.setattr(attention, "_flash4_sm120_dense_supported", lambda *args: True)
    q = torch.arange(24.0).reshape(1, 2, 3, 4)

    result = attention.attention_flash4_sm120_dense(
        q,
        q + 100,
        q + 200,
        2,
        skip_reshape=True,
        skip_output_reshape=True,
        scale=0.375,
    )

    assert calls == [
        (
            torch.Size((1, 3, 2, 4)),
            torch.Size((1, 3, 2, 4)),
            torch.Size((1, 3, 2, 4)),
            {"softmax_scale": 0.375, "causal": False, "num_splits": 1},
        )
    ]
    assert torch.equal(result, q + 1)


def test_flash4_sm120_dense_uses_sdpa_for_unsupported_calls(monkeypatch):
    calls = []

    def sdpa(q, k, v, heads, **kwargs):
        calls.append((q, k, v, heads, kwargs))
        return q + 7

    monkeypatch.setattr(attention, "attention_pytorch", sdpa)
    q = torch.zeros((1, 2, 3, 4))

    result = attention.attention_flash4_sm120_dense(
        q, q + 1, q + 2, 2, mask=torch.ones((3, 3)), skip_reshape=True, scale=0.5
    )

    assert torch.equal(result, q + 7)
    assert calls[0][3] == 2
    assert calls[0][4]["mask"].shape == (3, 3)
    assert calls[0][4]["scale"] == 0.5


def test_flash4_sm120_dense_does_not_hide_eligible_kernel_failures(monkeypatch):
    monkeypatch.setattr(attention, "_flash4_sm120_dense_supported", lambda *args: True)

    def fail(*args, **kwargs):
        raise RuntimeError("FA4 failed")

    monkeypatch.setattr(attention, "flash_attn4_func", fail, raising=False)
    q = torch.zeros((1, 2, 3, 4))

    with pytest.raises(RuntimeError, match="FA4 failed"):
        attention.attention_flash4_sm120_dense(
            q, q, q, 2, skip_reshape=True, skip_output_reshape=True
        )


def test_sol_attention_registry_adapter_preserves_comfy_attention_layout(monkeypatch):
    calls = []

    def sol_attn(q, k, v, **kwargs):
        calls.append((q.shape, k.shape, v.shape, kwargs))
        return q + 1

    monkeypatch.setattr(attention.comfy_kitchen, "sol_attn", sol_attn)
    selected = attention.get_attention_function(
        "comfy_kitchen_sol", registry=attention.create_attention_function_registry()
    )
    q = torch.arange(24.0).reshape(1, 2, 3, 4)

    result = selected(
        q,
        q + 100,
        q + 200,
        2,
        skip_reshape=True,
        skip_output_reshape=True,
        sol_options={"tau": 1.3, "sink_blocks": [0, 2]},
    )

    assert calls == [
        (
            torch.Size((1, 3, 2, 4)),
            torch.Size((1, 3, 2, 4)),
            torch.Size((1, 3, 2, 4)),
            {"tau": 1.3, "sink_blocks": [0, 2]},
        )
    ]
    assert torch.equal(result, q + 1)


def test_chunked_sol_attention_projects_bounded_h3_slices(monkeypatch):
    chunks = []
    calls = []

    class Norm:
        weight = torch.ones(8)
        eps = 1e-6

    def qkv_proj(value):
        chunks.append(value.shape[0])
        return value.repeat(1, 3)

    def sol_attn_chunked(producer, rows, heads, rope, weights, **kwargs):
        projected = list(producer())
        calls.append((rows, heads, rope, weights, kwargs, projected))
        return torch.ones((rows, heads, 8)), "kmean", "vscale"

    monkeypatch.setattr(attention.comfy_kitchen, "sol_attn_chunked", sol_attn_chunked)
    selected = attention.get_attention_function(
        "comfy_kitchen_sol_chunked",
        registry=attention.create_attention_function_registry(),
    )
    x = torch.zeros((4097, 8))
    rope = object()

    output, key_mean, value_scale = selected(
        x,
        qkv_proj,
        lambda value: value + 2,
        Norm(),
        Norm(),
        4,
        rope,
        tau=1.3,
        sink_blocks=[0, 2],
    )

    assert chunks == [4096, 1]
    assert calls[0][:3] == (4097, 4, rope)
    assert calls[0][4] == {
        "kmean": None,
        "vscale": None,
        "rope_eps": 1e-6,
        "tau": 1.3,
        "sink_blocks": [0, 2],
    }
    assert [value.shape for value in calls[0][5]] == [(4096, 24), (1, 24)]
    assert output.shape == (4097, 32)
    assert key_mean == "kmean"
    assert value_scale == "vscale"
