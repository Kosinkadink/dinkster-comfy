from types import SimpleNamespace

import torch

import dinkster_comfy.ops as ops


def test_cuda_sdpa_initialization_reapplies_declared_order_once(monkeypatch):
    calls = []
    tensor = SimpleNamespace()

    def choose(*args, **kwargs):
        calls.append(("choose", args, kwargs))
        return 3

    def get_order():
        calls.append(("get",))
        return [3, 1, 2, 0, 4]

    def set_order(order):
        calls.append(("set", order))

    monkeypatch.setattr(ops, "_sdpa_cuda_priority_initialized", False)
    monkeypatch.setattr(ops, "SDPA_BACKEND_PRIORITY", [1, 3, 2, 0])
    monkeypatch.setattr(torch, "_fused_sdp_choice", choose)
    monkeypatch.setattr(torch._C, "_get_sdp_priority_order", get_order)
    monkeypatch.setattr(torch._C, "_set_sdp_priority_order", set_order)

    for _ in range(2):
        ops._initialize_cuda_sdpa_priority(
            tensor,
            tensor,
            tensor,
            None,
            0.0,
            True,
            0.25,
            True,
        )

    assert calls == [
        (
            "choose",
            (tensor, tensor, tensor),
            {
                "attn_mask": None,
                "dropout_p": 0.0,
                "is_causal": True,
                "scale": 0.25,
                "enable_gqa": True,
            },
        ),
        ("get",),
        ("set", [1, 3, 2, 0, 4]),
    ]
