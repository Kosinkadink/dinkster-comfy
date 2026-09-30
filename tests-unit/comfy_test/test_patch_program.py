from __future__ import annotations

import functools
from typing import Any, cast

import pytest
import torch

from dinkster_inference.hooks import WeightHook
from dinkster_inference.model_base import BaseModel
from dinkster_inference.model_patcher import HookWeightPatch, LowVramPatch, ModelPatcher, ModelPatcherDynamic
from dinkster_inference.patch_program import ModuleInsertionEntry, PatchProgram
from dinkster_inference.patcher_extension import PatcherInjection
from dinkster_inference.weight_adapter.bypass import BypassInjectionManager
from dinkster_inference.weight_adapter.lora import LoRAAdapter


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


def test_dynamic_patcher_installs_hooks_registered_after_loading() -> None:
    model = torch.nn.Linear(2, 2)
    model.weight_function = []
    model.bias_function = []
    patcher = object.__new__(ModelPatcherDynamic)
    ModelPatcher.__init__(patcher, model, torch.device("cpu"), torch.device("cpu"))
    hook = WeightHook()
    patches = {
        "weight": ("diff", (torch.ones_like(model.weight),)),
        "bias": ("diff", (torch.ones_like(model.bias),)),
    }

    assert set(patcher.add_hook_patches(hook, patches)) == {"weight", "bias"}
    patcher.add_hook_patches(hook, patches)

    assert [(type(function), function.key) for function in model.weight_function] == [(HookWeightPatch, "weight")]
    assert [(type(function), function.key) for function in model.bias_function] == [(HookWeightPatch, "bias")]
    assert patcher.hook_weight_function_keys == {"weight", "bias"}


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


def test_low_vram_patch_validates_before_deferred_materialization() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    delta = torch.ones_like(model.weight)
    patcher.add_patches({"weight": ("diff", (delta,))})
    deferred = LowVramPatch(
        "weight", patcher.patches, patcher.patch_program.validate_resources
    )

    delta[0, 0] = 2.0

    with pytest.raises(RuntimeError, match="patch resource for 'weight' changed"):
        deferred(model.weight.detach().clone())


def test_low_vram_patch_revalidates_each_deferred_materialization() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    delta = torch.ones_like(model.weight)
    patcher.add_patches({"weight": ("diff", (delta,))})
    deferred = LowVramPatch(
        "weight", patcher.patches, patcher.patch_program.validate_resources
    )
    deferred(model.weight.detach().clone())

    delta[0, 0] = 2.0

    with pytest.raises(RuntimeError, match="patch resource for 'weight' changed"):
        deferred(model.weight.detach().clone())


@pytest.mark.parametrize("strength", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_strength_is_refused(strength: float) -> None:
    with pytest.raises(ValueError, match="strength_patch must be finite"):
        PatchProgram().append_weight_delta(
            target="block.weight",
            patch=("diff", (torch.tensor([1.0]),)),
            strength_patch=strength,
            strength_model=1.0,
        )


def _insertion(
    namespace: str,
    *,
    site: str = "down.0",
    recipe: str = "test.module",
    order: int = 0,
    resources: object = None,
) -> ModuleInsertionEntry:
    return ModuleInsertionEntry.create(
        namespace=namespace,
        site=site,
        recipe=recipe,
        order=order,
        resources=resources,
    )


class _Materializer:
    def __init__(self, events: list[str], *, fail: bool = False):
        self.events = events
        self.fail = fail

    def create_handle(self, patcher, site, entry, scratch):
        return entry.namespace, site["path"], scratch, entry.resources.value

    def materialize(self, patcher, handle):
        namespace, path, scratch, resources = handle
        self.events.append(f"materialize:{namespace}:{path}")
        scratch["active"] = resources
        if self.fail:
            raise RuntimeError("materialization failed")

    def teardown(self, patcher, handle):
        namespace, _, scratch, _ = handle
        scratch.pop("active", None)
        self.events.append(f"teardown:{namespace}")


class _RuntimePatch:
    def __init__(self, value: torch.Tensor, events: list[str]):
        self.value = value
        self.events = events

    def to(self, device_or_dtype):
        self.value = self.value.to(device_or_dtype)
        return self

    def models(self):
        return ["auxiliary-model"]

    def cleanup(self):
        self.events.append("cleanup")


class _ModelPatchLike:
    def __init__(self, model_patch: ModelPatcher):
        self.model_patch = model_patch
        self.encoded_image = torch.ones(1)
        self.temp_data = None

    def to(self, device_or_dtype):
        self.encoded_image = self.encoded_image.to(device_or_dtype)
        self.temp_data = None
        return self

    def models(self):
        return [self.model_patch]


def _identity(value):
    return value


def _increment(value):
    return value + 1


def _site_patcher() -> ModelPatcher:
    model = torch.nn.Linear(2, 2, bias=False)
    model.patch_site_map = {
        "down.0": {"path": "down_blocks.0", "order": 0, "dimensions": (2, 2)},
        "mid.0": {"path": "mid_block", "order": 1, "dimensions": (2, 2)},
        "up.0": {"path": "up_blocks.0", "order": 2, "dimensions": (2, 2)},
    }
    return ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))


def test_model_patcher_always_publishes_root_site() -> None:
    model = torch.nn.Linear(2, 2, bias=False)

    patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))

    assert patcher.patch_site_map["model.root"] == {"path": "", "order": 0}


def test_base_model_publishes_stable_unet_block_sites() -> None:
    diffusion_model = torch.nn.Module()
    diffusion_model.input_blocks = torch.nn.ModuleList(
        (torch.nn.Identity(), torch.nn.Identity())
    )
    diffusion_model.middle_block = torch.nn.Identity()
    diffusion_model.output_blocks = torch.nn.ModuleList((torch.nn.Identity(),))
    model = type("Model", (), {"diffusion_model": diffusion_model})()

    assert BaseModel._module_patch_sites(model) == {
        "model.root": {"path": "diffusion_model", "order": 0},
        "down.0": {"path": "diffusion_model.input_blocks.0", "order": 1},
        "down.1": {"path": "diffusion_model.input_blocks.1", "order": 2},
        "mid.0": {"path": "diffusion_model.middle_block", "order": 3},
        "up.0": {"path": "diffusion_model.output_blocks.0", "order": 4},
    }


def test_module_insertion_identity_covers_structural_inputs() -> None:
    base = _insertion("motion", resources={"weight": torch.tensor([1.0])})
    changed_site = ModuleInsertionEntry.create(
        namespace="motion",
        site="mid.0",
        recipe="test.module",
        resources={"weight": torch.tensor([1.0])},
    )
    changed_activation = ModuleInsertionEntry.create(
        namespace="motion",
        site="down.0",
        recipe="test.module",
        activation=0.5,
        resources={"weight": torch.tensor([1.0])},
    )
    changed_resource = _insertion("motion", resources={"weight": torch.tensor([2.0])})

    digests = {
        PatchProgram((entry,)).digest
        for entry in (base, changed_site, changed_activation, changed_resource)
    }
    assert len(digests) == 4


def test_module_insertion_refuses_unknown_site_with_available_sites() -> None:
    patcher = _site_patcher()
    materializer = _Materializer([])
    patcher.register_patch_materializer("test.module", materializer)

    with pytest.raises(
        ValueError, match="unknown module insertion site 'other'; available sites"
    ):
        patcher.set_module_insertions("motion", (_insertion("motion", site="other"),))


def test_module_insertion_collision_is_refused() -> None:
    with pytest.raises(ValueError, match="site, position, and order must be unique"):
        PatchProgram().replace_module_insertions(
            "motion", (_insertion("motion"), _insertion("motion"))
        )


def test_module_insertions_materialize_in_site_order_and_teardown_in_reverse() -> None:
    events: list[str] = []
    patcher = _site_patcher()
    materializer = _Materializer(events)
    patcher.register_patch_materializer("test.module", materializer)
    patcher.set_module_insertions(
        "modules",
        (
            _insertion("modules", site="up.0", order=1),
            _insertion("modules", site="down.0", order=2),
        ),
    )

    patcher.inject_model()
    patcher.eject_model()

    assert events == [
        "materialize:modules:down_blocks.0",
        "materialize:modules:up_blocks.0",
        "teardown:modules",
        "teardown:modules",
    ]
    assert patcher._patch_scratch == {}


def test_module_insertion_failure_rolls_back_materialized_entries() -> None:
    events: list[str] = []
    patcher = _site_patcher()
    patcher.register_patch_materializer("test.ok", _Materializer(events))
    patcher.register_patch_materializer("test.fail", _Materializer(events, fail=True))
    patcher.set_module_insertions(
        "ok", (_insertion("ok", recipe="test.ok", site="down.0"),)
    )
    patcher.set_module_insertions(
        "fail", (_insertion("fail", recipe="test.fail", site="mid.0"),)
    )

    with pytest.raises(RuntimeError, match="materialization failed"):
        patcher.inject_model()

    assert events == [
        "materialize:ok:down_blocks.0",
        "materialize:fail:mid_block",
        "teardown:fail",
        "teardown:ok",
    ]
    assert patcher._materialized_insertions == []
    assert patcher._patch_scratch == {}


def test_module_insertion_clone_derives_without_changing_parent() -> None:
    parent = _site_patcher()
    materializer = _Materializer([])
    parent.register_patch_materializer("test.module", materializer)
    child = parent.clone()
    child.set_module_insertions("motion", (_insertion("motion"),))

    assert parent.get_module_insertions() == ()
    assert child.get_module_insertions() == (_insertion("motion"),)
    assert parent.patch_program.digest != child.patch_program.digest


def test_injected_clone_can_transition_shared_model() -> None:
    events: list[str] = []
    parent = _site_patcher()
    parent.register_patch_materializer("test.module", _Materializer(events))
    parent.set_module_insertions("motion", (_insertion("motion"),))
    parent.inject_model()

    child = parent.clone()
    child.eject_model()
    child.inject_model()

    assert events == [
        "materialize:motion:down_blocks.0",
        "teardown:motion",
        "materialize:motion:down_blocks.0",
    ]


def test_injections_use_structural_program_materialization() -> None:
    events: list[str] = []
    patcher = _site_patcher()
    injection = PatcherInjection(
        inject=lambda _: events.append("inject"),
        eject=lambda _: events.append("eject"),
        resources={"recipe": "test"},
    )

    patcher.set_injections("test", [injection])
    patcher.inject_model()
    patcher.eject_model()

    assert patcher.get_injections("test") == [injection]
    assert events == ["inject", "eject"]
    patcher.remove_injections("test")
    assert patcher.get_injections("test") is None


def test_bypass_adapter_injection_materializes_and_restores_forward(monkeypatch) -> None:
    model = torch.nn.Module()
    model.layer = torch.nn.Linear(2, 2, bias=False)
    patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    monkeypatch.setattr(
        "dinkster_inference.model_management.get_torch_device",
        lambda: torch.device("meta"),
    )
    adapter = LoRAAdapter.create_train(model.layer.weight, rank=1, alpha=1.0)
    adapter.lora_up.weight.data.fill_(1.0)
    adapter.lora_down.weight.data.fill_(1.0)
    manager = BypassInjectionManager()
    manager.add_adapter("layer.weight", adapter)
    patcher.set_injections("motion", manager.create_injections(model))
    value = torch.ones(1, 2)
    original = model.layer(value)

    patcher.inject_model()
    patched = model.layer(value)
    patcher.eject_model()

    assert not torch.equal(patched, original)
    torch.testing.assert_close(model.layer(value), original, rtol=0, atol=0)


def test_additional_models_use_clone_scoped_program_resources() -> None:
    parent = _site_patcher()
    auxiliary = _site_patcher()
    parent.set_additional_models("control", [auxiliary])

    assert isinstance(parent.additional_models["control"], list)
    child = parent.clone()

    bound_auxiliary = parent.get_additional_models_with_key("control")[0]
    assert bound_auxiliary is not auxiliary
    assert bound_auxiliary.clone_base_uuid == auxiliary.clone_base_uuid
    cloned_auxiliary = child.get_additional_models_with_key("control")[0]
    assert cloned_auxiliary is not auxiliary
    assert cloned_auxiliary is not bound_auxiliary
    assert cloned_auxiliary.clone_base_uuid == auxiliary.clone_base_uuid
    child.remove_additional_models("control")
    assert child.get_additional_models_with_key("control") == []
    assert parent.get_additional_models_with_key("control") == [bound_auxiliary]


def test_module_insertion_revalidates_resources_before_materialization() -> None:
    patcher = _site_patcher()
    materializer = _Materializer([])
    patcher.register_patch_materializer("test.module", materializer)
    resource = torch.ones(1)
    patcher.set_module_insertions("motion", (_insertion("motion", resources=resource),))
    resource[0] = 2.0

    with pytest.raises(
        RuntimeError, match="module insertion resources for 'motion' changed"
    ):
        patcher.inject_model()


def test_object_patch_derives_program_without_changing_parent() -> None:
    parent = _site_patcher()
    child = parent.clone()

    child.add_object_patch("model_sampling", {"kind": "test"})

    assert parent.object_patches == {}
    assert child.object_patches == {"model_sampling": {"kind": "test"}}
    assert parent.patch_program.digest != child.patch_program.digest


def test_replacing_weight_patches_preserves_object_patch() -> None:
    patcher = _site_patcher()
    patcher.add_object_patch("manual_cast_dtype", torch.float16)

    patcher.patches = {
        "weight": [(1.0, ("diff", (torch.ones(2, 2),)), 1.0, None, None)]
    }

    assert patcher.object_patches == {"manual_cast_dtype": torch.float16}


def test_object_patch_resource_mutation_is_refused() -> None:
    patcher = _site_patcher()
    replacement = {"value": torch.ones(1)}
    patcher.add_object_patch("model_sampling", replacement)
    replacement["value"][0] = 2.0

    with pytest.raises(
        RuntimeError, match="object replacement for 'model_sampling' changed"
    ):
        patcher._validate_patch_program()


def test_runtime_patches_compile_existing_model_options_shape() -> None:
    patcher = _site_patcher()

    patcher.set_model_patch(_identity, "input_block_patch")
    patcher.set_model_patch_replace(_increment, "attn1", "input", 2, 3)

    transformer_options = patcher.model_options["transformer_options"]
    assert transformer_options["patches"] == {"input_block_patch": [_identity]}
    assert transformer_options["patches_replace"] == {
        "attn1": {("input", 2, 3): _increment}
    }


def test_replacing_runtime_patch_derives_without_changing_parent() -> None:
    parent = _site_patcher()
    parent.set_model_patch_replace(_identity, "attn1", "input", 2)
    child = parent.clone()

    child.set_model_patch_replace(_increment, "attn1", "input", 2)

    parent_options = parent.model_options["transformer_options"]["patches_replace"]
    child_options = child.model_options["transformer_options"]["patches_replace"]
    assert parent_options["attn1"][("input", 2)] is _identity
    assert child_options["attn1"][("input", 2)] is _increment
    assert parent.patch_program.digest != child.patch_program.digest


def test_runtime_patch_lifecycle_uses_program_resources() -> None:
    events: list[str] = []
    patcher = _site_patcher()
    runtime_patch = _RuntimePatch(torch.ones(1), events)
    patcher.set_model_patch(runtime_patch, "post_input")

    patcher.model_patches_to(torch.float16)
    assert patcher.model_patches_models() == ["auxiliary-model"]
    patcher.model_patches_call_function()

    compiled = patcher.model_options["transformer_options"]["patches"]
    assert compiled["post_input"][0].value.dtype == torch.float16
    assert events == ["cleanup"]
    patcher._validate_patch_program()


def test_model_patch_resource_exposes_auxiliary_model() -> None:
    patcher = _site_patcher()
    auxiliary = _site_patcher()
    model_patch = _ModelPatchLike(auxiliary)

    patcher.set_model_patch(model_patch, "post_input")
    patcher.model_patches_to(torch.float16)

    assert patcher.model_patches_models() == [auxiliary]
    compiled = patcher.model_options["transformer_options"]["patches"]
    assert compiled["post_input"] == [model_patch]
    assert model_patch.encoded_image.dtype == torch.float16
