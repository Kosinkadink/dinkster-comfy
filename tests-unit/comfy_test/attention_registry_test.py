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
