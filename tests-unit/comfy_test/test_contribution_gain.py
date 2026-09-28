import pytest
import torch

import dinkster_inference.float as float_module
import dinkster_inference.lora as lora_module
import dinkster_inference.model_patcher as model_patcher_module
import dinkster_inference.ops as ops_module
from dinkster_inference.contribution_gain import ContributionGain, GainKeyframe, GainTimeline
from dinkster_inference.controlnet import ControlBase
from dinkster_inference.hooks import (
    EnumWeightTarget,
    Hook,
    HookGroup,
    HookKeyframe,
    HookKeyframeGroup,
    TransformerOptionsHook,
    WeightHook,
    create_target_dict,
    load_hook_lora_for_models,
)
from dinkster_inference.model_patcher import HookWeightPatch, ModelPatcher
from dinkster_inference.samplers import realize_contribution_gains
from dinkster_inference.weight_adapter.base import WeightAdapterBase


def test_constant_gain_realizes_once_for_each_executed_sigma():
    gain = ContributionGain(global_gain=2.0)

    table = gain.realize(torch.tensor([10.0, 5.0, 1.0, 0.0]))

    assert table.sigmas == (10.0, 5.0, 1.0)
    assert table.timeline_gains == (1.0, 1.0, 1.0)
    assert table.scalar_gain(5.0) == 2.0


def test_table_must_match_executed_timeline():
    gain = ContributionGain(timeline=GainTimeline.from_table([1.0, 0.5]))

    with pytest.raises(ValueError, match="match the executed sigma rows"):
        gain.realize(torch.tensor([3.0, 2.0, 1.0, 0.0]))


def test_hold_keyframes_use_progress_anchors_and_zero_endpoints():
    gain = ContributionGain(
        timeline=GainTimeline.from_keyframes(
            [
                GainKeyframe("start", 0.25, 0.5),
                GainKeyframe("end", 0.75, 1.0),
            ],
            endpoint_before="zero.v1",
            endpoint_after="zero.v1",
        ),
        global_gain=0.8,
        site_gains=(("down.0", 0.5),),
        lane_gains=(("positive", 0.25),),
    )

    table = gain.realize(
        torch.tensor([10.0, 7.5, 5.0, 2.5, 0.0]),
        percent_to_sigma=lambda percent: 10.0 * (1.0 - percent),
    )

    assert table.timeline_gains == (0.0, 0.5, 0.5, 1.0)
    assert table.scalar_gain(5.0, site="down.0", lane="positive") == pytest.approx(0.05)


def test_effect_masks_reject_values_outside_fraction_domain():
    with pytest.raises(ValueError, match="finite fractions"):
        ContributionGain(effect_masks=(torch.tensor([0.0, 1.1]),))


def test_hook_strength_reads_the_realized_table_without_runtime_counters():
    keyframes = HookKeyframeGroup()
    keyframes.add(HookKeyframe(1.0, start_percent=0.0))
    keyframes.add(HookKeyframe(0.5, start_percent=0.5))
    hook = Hook(hook_keyframe=keyframes)
    model = type(
        "Model",
        (),
        {
            "model_sampling": type(
                "Sampling", (), {"percent_to_sigma": lambda self, value: 10.0 * (1.0 - value)}
            )()
        },
    )()

    hook.realize_gain(torch.tensor([10.0, 7.0, 4.0, 1.0, 0.0]), model)

    assert hook.strength == 1.0
    assert hook.prepare_current_gain(torch.tensor([4.0]))
    assert hook.strength == 0.5
    assert hook.prepare_current_gain(torch.tensor([7.0]))
    assert hook.strength == 1.0


def test_conditioning_legacy_strength_and_range_compile_to_one_gain_table():
    model = type(
        "Model",
        (),
        {
            "model_sampling": type(
                "Sampling", (), {"percent_to_sigma": lambda self, value: 10.0 * (1.0 - value)}
            )()
        },
    )()
    metadata = {"strength": 0.5, "start_percent": 0.25, "end_percent": 0.75}

    realize_contribution_gains(
        model,
        {"positive": [metadata]},
        torch.tensor([10.0, 7.5, 5.0, 2.5, 0.0]),
    )

    assert "strength" not in metadata
    assert "start_percent" not in metadata
    assert "end_percent" not in metadata
    table = metadata["realized_contribution_gain"]
    assert table.timeline_gains == (0.0, 1.0, 1.0, 1.0)
    assert table.scalar_gain(5.0, lane="positive") == 0.5


def test_control_residual_uses_realized_global_site_lane_and_mask_gain():
    model = type(
        "Model",
        (),
        {
            "model_sampling": type(
                "Sampling", (), {"percent_to_sigma": lambda self, value: 10.0 * (1.0 - value)}
            )()
        },
    )()
    control = ControlBase()
    control.contribution_gain = ContributionGain(
        global_gain=0.5,
        site_gains=(("middle.0", 0.5),),
        lane_gains=(("positive", 0.5),),
        effect_masks=(torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),),
    )
    control.realize_gain(torch.tensor([10.0, 5.0, 0.0]), model)

    merged = control.control_merge(
        {"middle": [torch.ones(1, 1, 2, 2)]},
        None,
        None,
        torch.tensor([10.0]),
        {"cond_or_uncond": [0]},
    )

    assert torch.equal(
        merged["middle"][0],
        torch.tensor([[[[0.125, 0.0], [0.125, 0.0]]]]),
    )


def test_float_weight_overlay_matches_materialized_hook_without_mutating_base_weight():
    patcher = ModelPatcher(
        torch.nn.Linear(2, 2, bias=False),
        torch.device("cpu"),
        torch.device("cpu"),
    )
    hook = WeightHook()
    diff = torch.tensor([[0.10003, -0.20007], [0.30011, -0.40013]])
    patcher.hook_patches[hook.hook_ref] = {
        "weight": [
            (1.0, ("diff", (diff,)), 1.0, None, None)
        ]
    }
    overlay = HookWeightPatch(patcher, "weight")
    weight = torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float16)
    original = weight.clone()

    assert torch.equal(overlay(weight), weight)

    active = HookGroup()
    active.add(hook)
    patcher.current_hooks = active

    hook.current_gain = 0.0
    assert not overlay.active()
    assert torch.equal(overlay(weight), original)

    hook.current_gain = 1.0
    assert overlay.active()
    materialized = float_module.stochastic_rounding(
        original.float() + diff,
        original.dtype,
    )
    assert torch.equal(overlay(weight), materialized)
    assert torch.equal(weight, original)


def test_inactive_float_weight_overlay_preserves_resident_weight() -> None:
    model = ops_module.disable_weight_init.Linear(2, 2, bias=False)
    model.weight.data.copy_(torch.tensor([[1.0, -2.0], [3.0, -4.0]]))
    patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    hook = WeightHook()
    patcher.hook_patches[hook.hook_ref] = {
        "weight": [(1.0, ("diff", (torch.ones(2, 2),)), 1.0, None, None)]
    }
    active = HookGroup()
    active.add(hook)
    patcher.current_hooks = active
    hook.current_gain = 0.0
    model.weight_function = [HookWeightPatch(patcher, "weight")]

    weight, _ = ops_module.cast_bias_weight(model, dtype=model.weight.dtype, device=model.weight.device)

    assert weight.data_ptr() == model.weight.data_ptr()


def test_float_weight_overlay_is_not_baked_before_model_load():
    model = torch.nn.Linear(2, 2, bias=False)
    model.weight.data.zero_()
    model.weight_function = []
    patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    hook = WeightHook()
    patcher.add_hook_patches(
        hook,
        {"weight": ("diff", (torch.ones(2, 2),))},
    )
    active = HookGroup()
    active.add(hook)

    patcher.patch_hooks(active)

    assert patcher.hook_weight_function_keys == {"weight"}
    assert torch.equal(model.weight, torch.zeros(2, 2))


def test_loaded_lora_hook_carries_reusable_patches(monkeypatch):
    model = torch.nn.Linear(2, 2, bias=False)
    patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    patch = ("diff", (torch.ones(2, 2),))
    load_calls = []
    monkeypatch.setattr(
        lora_module,
        "model_lora_keys_unet",
        lambda model, key_map: key_map,
    )

    def load_lora(lora, key_map, log_missing=True):
        load_calls.append(lora)
        return {"weight": patch}

    monkeypatch.setattr(lora_module, "load_lora", load_lora)

    loaded, _, hooks = load_hook_lora_for_models(
        patcher,
        None,
        {"source": torch.ones(1)},
        strength_model=1.0,
        strength_clip=0.0,
    )
    hook = hooks.hooks[0]
    loaded.hook_patches.clear()

    registered = HookGroup()
    hook.add_hook_patches(
        loaded,
        {},
        create_target_dict(EnumWeightTarget.Model),
        registered,
    )

    assert load_calls == [{"source": torch.ones(1)}]
    assert loaded.hook_patches[hook.hook_ref]["weight"][0][1] == patch
    assert registered.contains(hook)


def test_hook_patching_does_not_reload_unrelated_virtual_state_keys():
    class VirtualStateLinear(torch.nn.Linear):
        def state_dict(self, *args, **kwargs):
            state = super().state_dict(*args, **kwargs)
            state["weight_scale"] = torch.tensor(0.5)
            return state

    model = VirtualStateLinear(2, 2, bias=False)
    model.weight.data.zero_()
    patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    hook = WeightHook()
    patcher.add_hook_patches(
        hook,
        {"weight": ("diff", (torch.ones(2, 2),))},
    )
    active = HookGroup()
    active.add(hook)

    patcher.patch_hooks(active)

    assert torch.equal(model.weight, torch.ones(2, 2))


class _BypassAdapter(WeightAdapterBase):
    def __init__(self):
        self.weights = (torch.ones(1),)

    def h(self, x, base_out):
        return torch.ones_like(base_out) * self.multiplier


def test_quantized_scheduled_adapter_uses_bypass_without_changing_base_weight(monkeypatch):
    monkeypatch.setattr(model_patcher_module, "QuantizedTensor", torch.Tensor)
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
    model[0].weight.data.zero_()
    patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    hook = WeightHook()

    applied = patcher.add_hook_patches(
        hook,
        {"0.weight": _BypassAdapter()},
        strength_patch=2.0,
    )

    assert applied == ["0.weight"]
    assert patcher.hook_patches[hook.hook_ref] == {}
    assert patcher.get_module_insertions()[0].recipe == "dinkster.scheduled_bypass"

    patcher.inject_model()
    inputs = torch.ones(1, 2)
    assert torch.equal(model(inputs), torch.zeros(1, 2))

    active = HookGroup()
    hook.current_gain = 0.25
    active.add(hook)
    patcher.current_hooks = active
    assert torch.equal(model(inputs), torch.full((1, 2), 0.5))

    patcher.eject_model()
    assert torch.equal(model(inputs), torch.zeros(1, 2))


def test_guidance_row_gain_routes_float_adapter_through_bypass():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
    model[0].weight.data.zero_()
    patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    hook = WeightHook()
    hook.contribution_gain = ContributionGain(
        lane_gains=(("positive", 2.0), ("negative", 0.5))
    )
    hook.realized_gain = hook.contribution_gain.realize(torch.tensor([1.0, 0.0]))
    hook.current_timestep = 1.0
    patcher.add_hook_patches(hook, {"0.weight": _BypassAdapter()})
    active = HookGroup()
    active.add(hook)
    patcher.current_hooks = active
    patcher.current_transformer_options = {"cond_or_uncond": [0, 1]}
    patcher.inject_model()

    output = model(torch.ones(2, 2))

    assert torch.equal(output[0], torch.full((2,), 2.0))
    assert torch.equal(output[1], torch.full((2,), 0.5))


def test_static_bypass_adapter_program_preserves_trainable_resource():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
    model[0].weight.data.zero_()
    patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    adapter = _BypassAdapter()
    patcher.set_bypass_adapters("training", {"0.weight": (adapter, 0.5)})

    patcher.inject_model()
    assert torch.equal(model(torch.ones(1, 2)), torch.full((1, 2), 0.5))
    patcher.eject_model()

    adapter.weights[0].fill_(2.0)
    patcher.inject_model()
    assert torch.equal(model(torch.ones(1, 2)), torch.full((1, 2), 0.5))

    patcher.eject_model()
    patcher.remove_bypass_adapters("training")
    assert patcher.get_module_insertions("bypass:training") == ()


def test_program_identity_distinguishes_scheduled_bypass_gain_declarations(monkeypatch):
    monkeypatch.setattr(model_patcher_module, "QuantizedTensor", torch.Tensor)
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
    first = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    second = first.clone()
    first_hook = WeightHook()
    second_hook = WeightHook()
    second_hook.hook_keyframe.add(HookKeyframe(0.5, 0.0))

    first.add_hook_patches(first_hook, {"0.weight": _BypassAdapter()})
    second.add_hook_patches(second_hook, {"0.weight": _BypassAdapter()})

    assert first.patch_program.digest != second.patch_program.digest
    assert not first.clone_has_same_weights(second)


def test_attention_term_uses_realized_site_gain():
    def add_one(query, key, value, extra_options):
        return query + 1, key + 1, value + 1

    hook = TransformerOptionsHook(
        {"patches": {"attn1_patch": [add_one]}}
    )
    hook.realized_gain = ContributionGain(
        site_gains=(("attn1_patch", 0.25),)
    ).realize(torch.tensor([1.0, 0.0]))
    hook.current_timestep = 1.0
    options = {}

    hook.on_apply_hooks(None, options)
    patch = options["patches"]["attn1_patch"][0]
    inputs = (torch.zeros(1), torch.zeros(1), torch.zeros(1))

    assert patch(*inputs, {}) == tuple(torch.full((1,), 0.25) for _ in range(3))


def test_tensor_gain_applies_guidance_rows_and_ordered_masks():
    table = ContributionGain(
        lane_gains=(("positive", 2.0), ("negative", 0.5)),
        effect_masks=(torch.full((1, 1, 1), 0.5), torch.full((1, 1, 1), 0.25)),
    ).realize(torch.tensor([1.0, 0.0]))
    output = torch.ones(2, 1, 1, 1)

    gain = table.tensor_gain(1.0, output, lanes=(0, 1))

    assert torch.equal(
        gain,
        torch.tensor([0.25, 0.0625]).reshape(2, 1, 1, 1),
    )
