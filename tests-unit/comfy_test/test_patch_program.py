from __future__ import annotations

import functools
from typing import Any, cast

import pytest
import torch

from dinkster_comfy.model_patcher import ModelPatcher
from dinkster_comfy.patch_program import PatchProgram


def _program(value: torch.Tensor, *, target: str = "block.weight") -> PatchProgram:
    return PatchProgram().append_weight_delta(
        target=target,
        patch=("diff", (value,)),
        strength_patch=0.75,
        strength_model=1.0,
    )


def test_equal_programs_have_content_identity() -> None:
    first = _program(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    second = _program(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    changed = _program(torch.tensor([[1.0, 2.0], [3.0, 5.0]]))

    assert first.digest == second.digest
    assert first.digest != changed.digest


def test_equivalent_numeric_strengths_have_the_same_identity() -> None:
    patch = ("diff", (torch.tensor([1.0]),))
    integer_strength = PatchProgram().append_weight_delta(
        target="block.weight",
        patch=patch,
        strength_patch=1,
        strength_model=-0.0,
    )
    float_strength = PatchProgram().append_weight_delta(
        target="block.weight",
        patch=patch,
        strength_patch=1.0,
        strength_model=0.0,
    )

    assert integer_strength.digest == float_strength.digest


def test_append_derives_without_changing_the_base_program() -> None:
    base = _program(torch.tensor([1.0]), target="first.weight")
    derived = base.append_weight_delta(
        target="second.weight",
        patch=("diff", (torch.tensor([2.0]),)),
        strength_patch=1.0,
        strength_model=1.0,
    )

    assert list(base.weight_patches()) == ["first.weight"]
    assert list(derived.weight_patches()) == ["first.weight", "second.weight"]
    assert base.digest != derived.digest


def test_program_order_is_part_of_identity_and_runtime_order() -> None:
    first_then_second = _program(torch.tensor([1.0])).append_weight_delta(
        target="block.weight",
        patch=("diff", (torch.tensor([2.0]),)),
        strength_patch=1.0,
        strength_model=1.0,
    )
    second_then_first = (
        PatchProgram()
        .append_weight_delta(
            target="block.weight",
            patch=("diff", (torch.tensor([2.0]),)),
            strength_patch=1.0,
            strength_model=1.0,
        )
        .append_weight_delta(
            target="block.weight",
            patch=("diff", (torch.tensor([1.0]),)),
            strength_patch=0.75,
            strength_model=1.0,
        )
    )

    assert first_then_second.digest != second_then_first.digest
    assert first_then_second.weight_patches()["block.weight"][0][0] == 0.75
    assert first_then_second.weight_patches()["block.weight"][1][0] == 1.0


def test_compiled_runtime_mapping_cannot_mutate_the_program() -> None:
    program = _program(torch.tensor([1.0]))
    compiled = program.weight_patches()
    compiled["block.weight"].clear()

    assert len(program.weight_patches()["block.weight"]) == 1


def test_partial_patch_function_has_content_identity() -> None:
    identity = functools.partial(torch.mul, other=1.0)
    doubled = functools.partial(torch.mul, other=2.0)

    first = PatchProgram().append_weight_delta(
        target="block.weight",
        patch=("diff", (torch.tensor([1.0]),)),
        strength_patch=1.0,
        strength_model=1.0,
        function=identity,
    )
    second = PatchProgram().append_weight_delta(
        target="block.weight",
        patch=("diff", (torch.tensor([1.0]),)),
        strength_patch=1.0,
        strength_model=1.0,
        function=doubled,
    )

    assert first.digest != second.digest


def test_model_patcher_clone_derives_without_changing_parent() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    parent = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    child = parent.clone()

    child.add_patches(
        {"weight": ("diff", (torch.ones_like(model.weight),))},
        strength_patch=0.5,
    )

    assert parent.patch_program.entries == ()
    assert len(child.patch_program.entries) == 1
    assert parent.patches == {}
    assert child.patches["weight"][0][0] == 0.5
    assert parent.patches_uuid != child.patches_uuid

    immutable_patches = cast(Any, child.patches)
    with pytest.raises(TypeError):
        immutable_patches["weight"] = child.patches["weight"]


def test_model_patcher_refuses_a_mutated_bound_resource() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    delta = torch.ones_like(model.weight)
    patcher.add_patches({"weight": ("diff", (delta,))})

    delta[0, 0] = 2.0

    with pytest.raises(RuntimeError, match="patch resource for 'weight' changed"):
        patcher.patch_weight_to_device("weight", return_weight=True)


def test_model_patcher_revalidates_before_each_materialization() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    delta = torch.ones_like(model.weight)
    patcher.add_patches({"weight": ("diff", (delta,))})
    patcher.patch_weight_to_device("weight", return_weight=True)

    delta[0, 0] = 2.0

    with pytest.raises(RuntimeError, match="patch resource for 'weight' changed"):
        patcher.patch_weight_to_device("weight", return_weight=True)


@pytest.mark.parametrize("strength", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_strength_is_refused(strength: float) -> None:
    with pytest.raises(ValueError, match="strength_patch must be finite"):
        PatchProgram().append_weight_delta(
            target="block.weight",
            patch=("diff", (torch.tensor([1.0]),)),
            strength_patch=strength,
            strength_model=1.0,
        )
