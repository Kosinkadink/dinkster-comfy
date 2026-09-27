import pytest
import torch

from dinkster_comfy.contribution_gain import ContributionGain, GainKeyframe, GainTimeline
from dinkster_comfy.controlnet import ControlBase
from dinkster_comfy.hooks import Hook, HookKeyframe, HookKeyframeGroup
from dinkster_comfy.samplers import realize_contribution_gains


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
