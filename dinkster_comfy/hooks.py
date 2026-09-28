from __future__ import annotations
from typing import TYPE_CHECKING, Callable
import enum
import math
import torch
import numpy as np
import itertools
import logging

if TYPE_CHECKING:
    from dinkster_comfy.model_patcher import ModelPatcher, PatcherInjection
    from dinkster_comfy.model_base import BaseModel
    from dinkster_comfy.sd import CLIP
import dinkster_comfy.lora
import dinkster_comfy.model_management
import dinkster_comfy.patcher_extension
from dinkster_comfy.contribution_gain import ContributionGain, GainKeyframe, GainTimeline


def conditioning_set_values(conditioning, values={}, append=False):
    updated = []
    for tensor, metadata in conditioning:
        metadata = metadata.copy()
        for key, value in values.items():
            if append and key in metadata:
                value = metadata[key] + value
            metadata[key] = value
        updated.append([tensor, metadata])
    return updated

# #######################################################################################################
# Hooks explanation
# -------------------
# The purpose of hooks is to allow conds to influence sampling without the need for ComfyUI core code to
# make explicit special cases like it does for ControlNet and GLIGEN.
#
# This is necessary for nodes/features that are intended for use with masked or scheduled conds, or those
# that should run special code when a 'marked' cond is used in sampling.
# #######################################################################################################

class EnumHookMode(enum.Enum):
    '''
    Priority of hook memory optimization vs. speed, mostly related to WeightHooks.

    MinVram: No caching will occur for any operations related to hooks.
    MaxSpeed: Excess VRAM (and RAM, once VRAM is sufficiently depleted) will be used to cache hook weights when switching hook groups.
    '''
    MinVram = "minvram"
    MaxSpeed = "maxspeed"

class EnumHookType(enum.Enum):
    '''
    Hook types, each of which has different expected behavior.
    '''
    Weight = "weight"
    ObjectPatch = "object_patch"
    AdditionalModels = "add_models"
    TransformerOptions = "transformer_options"
    Injections = "add_injections"

class EnumWeightTarget(enum.Enum):
    Model = "model"
    Clip = "clip"

class EnumHookScope(enum.Enum):
    '''
    Determines if hook should be limited in its influence over sampling.

    AllConditioning: hook will affect all conds used in sampling.
    HookedOnly: hook will only affect the conds it was attached to.
    '''
    AllConditioning = "all_conditioning"
    HookedOnly = "hooked_only"


class _HookRef:
    pass


def default_should_register(hook: Hook, model: ModelPatcher, model_options: dict, target_dict: dict[str], registered: HookGroup):
    '''Example for how custom_should_register function can look like.'''
    return True


def create_target_dict(target: EnumWeightTarget=None, **kwargs) -> dict[str]:
    '''Creates base dictionary for use with Hooks' target param.'''
    d = {}
    if target is not None:
        d['target'] = target
    d.update(kwargs)
    return d


class Hook:
    def __init__(self, hook_type: EnumHookType=None, hook_ref: _HookRef=None, hook_id: str=None,
                 hook_keyframe: HookKeyframeGroup=None, hook_scope=EnumHookScope.AllConditioning):
        self.hook_type = hook_type
        '''Enum identifying the general class of this hook.'''
        self.hook_ref = hook_ref if hook_ref else _HookRef()
        '''Reference shared between hook clones that have the same value. Should NOT be modified.'''
        self.hook_id = hook_id
        '''Optional string ID to identify hook; useful if need to consolidate duplicates at registration time.'''
        self.hook_keyframe = hook_keyframe if hook_keyframe else HookKeyframeGroup()
        '''Gain declaration used to realize strength against the executed timeline.'''
        self.contribution_gain = ContributionGain()
        self.realized_gain = None
        self.current_gain = 1.0
        self.current_timestep = None
        self.hook_scope = hook_scope
        '''Scope of where this hook should apply in terms of the conds used in sampling run.'''
        self.custom_should_register = default_should_register
        '''Can be overridden with a compatible function to decide if this hook should be registered without the need to override .should_register'''

    @property
    def strength(self):
        return self.current_gain

    def initialize_timesteps(self, model: BaseModel):
        self.reset()

    def realize_gain(self, sigmas: torch.Tensor, model: BaseModel):
        gain = self.contribution_gain.with_timeline(
            self.hook_keyframe.contribution_gain.timeline
        )
        self.realized_gain = gain.realize(sigmas, model.model_sampling.percent_to_sigma)
        self.current_timestep = self.realized_gain.sigmas[0]
        self.current_gain = self.realized_gain.scalar_gain(self.current_timestep)

    def prepare_current_gain(self, timestep: torch.Tensor) -> bool:
        if self.realized_gain is None:
            return False
        self.current_timestep = timestep
        gain = self.realized_gain.scalar_gain(timestep)
        changed = not math.isclose(gain, self.current_gain)
        self.current_gain = gain
        return changed

    def gain_for(self, *, site: str | None = None, lane: str | None = None) -> float:
        if self.realized_gain is None or self.current_timestep is None:
            return self.current_gain
        return self.realized_gain.scalar_gain(
            self.current_timestep,
            site=site,
            lane=lane,
        )

    def tensor_gain_for(self, output, *, site=None, transformer_options=None):
        if self.realized_gain is None or self.current_timestep is None:
            return self.current_gain
        options = transformer_options or {}
        return self.realized_gain.tensor_gain(
            self.current_timestep,
            output,
            site=site,
            lanes=options.get("cond_or_uncond", ()),
        )

    def reset(self):
        self.realized_gain = None
        self.current_gain = 1.0
        self.current_timestep = None

    def clone(self):
        c: Hook = self.__class__()
        c.hook_type = self.hook_type
        c.hook_ref = self.hook_ref
        c.hook_id = self.hook_id
        c.hook_keyframe = self.hook_keyframe
        c.contribution_gain = self.contribution_gain
        c.realized_gain = self.realized_gain
        c.current_gain = self.current_gain
        c.current_timestep = self.current_timestep
        c.hook_scope = self.hook_scope
        c.custom_should_register = self.custom_should_register
        return c

    def should_register(self, model: ModelPatcher, model_options: dict, target_dict: dict[str], registered: HookGroup):
        return self.custom_should_register(self, model, model_options, target_dict, registered)

    def add_hook_patches(self, model: ModelPatcher, model_options: dict, target_dict: dict[str], registered: HookGroup):
        raise NotImplementedError("add_hook_patches should be defined for Hook subclasses")

    def __eq__(self, other: Hook):
        return self.__class__ == other.__class__ and self.hook_ref == other.hook_ref

    def __hash__(self):
        return hash(self.hook_ref)

class WeightHook(Hook):
    '''
    Hook responsible for tracking weights to be applied to some model/clip.

    Note, value of hook_scope is ignored and is treated as HookedOnly.
    '''
    def __init__(self, strength_model=1.0, strength_clip=1.0):
        super().__init__(hook_type=EnumHookType.Weight, hook_scope=EnumHookScope.HookedOnly)
        self.weights: dict = None
        self.weights_clip: dict = None
        self.need_weight_init = True
        self._strength_model = strength_model
        self._strength_clip = strength_clip
        self.hook_scope = EnumHookScope.HookedOnly # this value does not matter for WeightHooks, just for docs

    @property
    def strength_model(self):
        return self._strength_model * self.strength

    @property
    def strength_clip(self):
        return self._strength_clip * self.strength

    def add_hook_patches(self, model: ModelPatcher, model_options: dict, target_dict: dict[str], registered: HookGroup):
        if not self.should_register(model, model_options, target_dict, registered):
            return False
        weights = None

        target = target_dict.get('target', None)
        if target == EnumWeightTarget.Clip:
            strength = self._strength_clip
        else:
            strength = self._strength_model

        if self.need_weight_init:
            key_map = {}
            if target == EnumWeightTarget.Clip:
                key_map = dinkster_comfy.lora.model_lora_keys_clip(model.model, key_map)
            else:
                key_map = dinkster_comfy.lora.model_lora_keys_unet(model.model, key_map)
            weights = dinkster_comfy.lora.load_lora(self.weights, key_map, log_missing=False)
        else:
            if target == EnumWeightTarget.Clip:
                weights = self.weights_clip
            else:
                weights = self.weights
        model.add_hook_patches(hook=self, patches=weights, strength_patch=strength)
        registered.add(self)
        return True
        # TODO: add logs about any keys that were not applied

    def clone(self):
        c: WeightHook = super().clone()
        c.weights = self.weights
        c.weights_clip = self.weights_clip
        c.need_weight_init = self.need_weight_init
        c._strength_model = self._strength_model
        c._strength_clip = self._strength_clip
        return c

class ObjectPatchHook(Hook):
    def __init__(self, object_patches: dict[str]=None,
                 hook_scope=EnumHookScope.AllConditioning):
        super().__init__(hook_type=EnumHookType.ObjectPatch)
        self.object_patches = object_patches
        self.hook_scope = hook_scope

    def clone(self):
        c: ObjectPatchHook = super().clone()
        c.object_patches = self.object_patches
        return c

    def add_hook_patches(self, model: ModelPatcher, model_options: dict, target_dict: dict[str], registered: HookGroup):
        raise NotImplementedError("ObjectPatchHook is not supported yet in ComfyUI.")

class AdditionalModelsHook(Hook):
    '''
    Hook responsible for telling model management any additional models that should be loaded.

    Note, value of hook_scope is ignored and is treated as AllConditioning.
    '''
    def __init__(self, models: list[ModelPatcher]=None, key: str=None):
        super().__init__(hook_type=EnumHookType.AdditionalModels)
        self.models = models
        self.key = key

    def clone(self):
        c: AdditionalModelsHook = super().clone()
        c.models = self.models.copy() if self.models else self.models
        c.key = self.key
        return c

    def add_hook_patches(self, model: ModelPatcher, model_options: dict, target_dict: dict[str], registered: HookGroup):
        if not self.should_register(model, model_options, target_dict, registered):
            return False
        registered.add(self)
        return True

class TransformerOptionsHook(Hook):
    '''
    Hook responsible for adding wrappers, callbacks, patches, or anything else related to transformer_options.
    '''
    def __init__(self, transformers_dict: dict[str, dict[str, dict[str, list[Callable]]]]=None,
                 hook_scope=EnumHookScope.AllConditioning):
        super().__init__(hook_type=EnumHookType.TransformerOptions)
        self.transformers_dict = transformers_dict
        self.hook_scope = hook_scope
        self._skip_adding = False
        '''Internal value used to avoid double load of transformer_options when hook_scope is AllConditioning.'''

    def clone(self):
        c: TransformerOptionsHook = super().clone()
        c.transformers_dict = self.transformers_dict
        c._skip_adding = self._skip_adding
        return c

    def add_hook_patches(self, model: ModelPatcher, model_options: dict, target_dict: dict[str], registered: HookGroup):
        if not self.should_register(model, model_options, target_dict, registered):
            return False
        # NOTE: to_load_options will be used to manually load patches/wrappers/callbacks from hooks
        self._skip_adding = False
        transformers_dict = _gain_scaled_attention_patches(self, self.transformers_dict)
        if self.hook_scope == EnumHookScope.AllConditioning:
            add_model_options = {"transformer_options": transformers_dict,
                                 "to_load_options": transformers_dict}
            # skip_adding if included in AllConditioning to avoid double loading
            self._skip_adding = True
        else:
            add_model_options = {"to_load_options": transformers_dict}
        registered.add(self)
        dinkster_comfy.patcher_extension.merge_nested_dicts(model_options, add_model_options, copy_dict1=False)
        return True

    def on_apply_hooks(self, model: ModelPatcher, transformer_options: dict[str]):
        if not self._skip_adding:
            dinkster_comfy.patcher_extension.merge_nested_dicts(
                transformer_options,
                _gain_scaled_attention_patches(self, self.transformers_dict),
                copy_dict1=False,
            )


class _ContributionGainAttentionPatch:
    def __init__(self, hook, site, patch):
        self.hook = hook
        self.site = site
        self.patch = patch

    def __call__(self, *args, **kwargs):
        result = self.patch(*args, **kwargs)
        options = args[-1] if args and isinstance(args[-1], dict) else {}
        if isinstance(result, tuple):
            return tuple(
                original
                + (patched - original)
                * self.hook.tensor_gain_for(
                    patched,
                    site=self.site,
                    transformer_options=options,
                )
                if isinstance(original, torch.Tensor) and isinstance(patched, torch.Tensor)
                else patched
                for original, patched in zip(args, result)
            )
        if isinstance(result, torch.Tensor) and args and isinstance(args[0], torch.Tensor):
            gain = self.hook.tensor_gain_for(
                result,
                site=self.site,
                transformer_options=options,
            )
            return args[0] + (result - args[0]) * gain
        return result


def _gain_scaled_attention_patches(hook, transformers_dict):
    if not transformers_dict or "patches" not in transformers_dict:
        return transformers_dict
    attention_sites = {
        "attn1_patch",
        "attn2_patch",
        "attn1_output_patch",
        "attn2_output_patch",
    }
    patches = transformers_dict["patches"]
    scaled = {
        site: [
            _ContributionGainAttentionPatch(hook, site, patch) for patch in values
        ]
        if site in attention_sites
        else values
        for site, values in patches.items()
    }
    return {**transformers_dict, "patches": scaled}

WrapperHook = TransformerOptionsHook
'''Only here for backwards compatibility, WrapperHook is identical to TransformerOptionsHook.'''

class InjectionsHook(Hook):
    def __init__(self, key: str=None, injections: list[PatcherInjection]=None,
                 hook_scope=EnumHookScope.AllConditioning):
        super().__init__(hook_type=EnumHookType.Injections)
        self.key = key
        self.injections = injections
        self.hook_scope = hook_scope

    def clone(self):
        c: InjectionsHook = super().clone()
        c.key = self.key
        c.injections = self.injections.copy() if self.injections else self.injections
        return c

    def add_hook_patches(self, model: ModelPatcher, model_options: dict, target_dict: dict[str], registered: HookGroup):
        raise NotImplementedError("InjectionsHook is not supported yet in ComfyUI.")

class HookGroup:
    '''
    Stores groups of hooks, and allows them to be queried by type.

    To prevent breaking their functionality, never modify the underlying self.hooks or self._hook_dict vars directly;
    always use the provided functions on HookGroup.
    '''
    def __init__(self):
        self.hooks: list[Hook] = []
        self._hook_dict: dict[EnumHookType, list[Hook]] = {}

    def __len__(self):
        return len(self.hooks)

    def add(self, hook: Hook):
        if hook not in self.hooks:
            self.hooks.append(hook)
            self._hook_dict.setdefault(hook.hook_type, []).append(hook)

    def remove(self, hook: Hook):
        if hook in self.hooks:
            self.hooks.remove(hook)
            self._hook_dict[hook.hook_type].remove(hook)

    def get_type(self, hook_type: EnumHookType):
        return self._hook_dict.get(hook_type, [])

    def contains(self, hook: Hook):
        return hook in self.hooks

    def is_subset_of(self, other: HookGroup):
        self_hooks = set(self.hooks)
        other_hooks = set(other.hooks)
        return self_hooks.issubset(other_hooks)

    def new_with_common_hooks(self, other: HookGroup):
        c = HookGroup()
        for hook in self.hooks:
            if other.contains(hook):
                c.add(hook.clone())
        return c

    def clone(self):
        c = HookGroup()
        for hook in self.hooks:
            c.add(hook.clone())
        return c

    def clone_and_combine(self, other: HookGroup):
        c = self.clone()
        if other is not None:
            for hook in other.hooks:
                c.add(hook.clone())
        return c

    def set_keyframes_on_hooks(self, hook_kf: HookKeyframeGroup):
        if hook_kf is None:
            hook_kf = HookKeyframeGroup()
        else:
            hook_kf = hook_kf.clone()
        for hook in self.hooks:
            hook.hook_keyframe = hook_kf

    def get_hooks_for_clip_schedule(self):
        scheduled_hooks: dict[WeightHook, list[tuple[tuple[float,float], HookKeyframe]]] = {}
        # only care about WeightHooks, for now
        for hook in self.get_type(EnumHookType.Weight):
            hook: WeightHook
            hook_schedule = []
            # if no hook keyframes, assign default value
            if len(hook.hook_keyframe.keyframes) == 0:
                hook_schedule.append(((0.0, 1.0), None))
                scheduled_hooks[hook] = hook_schedule
                continue
            # find ranges of values
            prev_keyframe = hook.hook_keyframe.keyframes[0]
            for keyframe in hook.hook_keyframe.keyframes:
                if keyframe.start_percent > prev_keyframe.start_percent and not math.isclose(keyframe.strength, prev_keyframe.strength):
                    hook_schedule.append(((prev_keyframe.start_percent, keyframe.start_percent), prev_keyframe))
                    prev_keyframe = keyframe
                elif keyframe.start_percent == prev_keyframe.start_percent:
                    prev_keyframe = keyframe
            # create final range, assuming last start_percent was not 1.0
            if not math.isclose(prev_keyframe.start_percent, 1.0):
                hook_schedule.append(((prev_keyframe.start_percent, 1.0), prev_keyframe))
            scheduled_hooks[hook] = hook_schedule
        # hooks should not have their schedules in a list of tuples
        all_ranges: list[tuple[float, float]] = []
        for range_kfs in scheduled_hooks.values():
            for t_range, keyframe in range_kfs:
                all_ranges.append(t_range)
        # turn list of ranges into boundaries
        boundaries_set = set(itertools.chain.from_iterable(all_ranges))
        boundaries_set.add(0.0)
        boundaries = sorted(boundaries_set)
        real_ranges = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
        # with real ranges defined, give appropriate hooks w/ keyframes for each range
        scheduled_keyframes: list[tuple[tuple[float,float], list[tuple[WeightHook, HookKeyframe]]]] = []
        for t_range in real_ranges:
            hooks_schedule = []
            for hook, val in scheduled_hooks.items():
                keyframe = None
                # check if is a keyframe that works for the current t_range
                for stored_range, stored_kf in val:
                    # if stored start is less than current end, then fits - give it assigned keyframe
                    if stored_range[0] < t_range[1] and stored_range[1] > t_range[0]:
                        keyframe = stored_kf
                        break
                hooks_schedule.append((hook, keyframe))
            scheduled_keyframes.append((t_range, hooks_schedule))
        return scheduled_keyframes

    def reset(self):
        for hook in self.hooks:
            hook.reset()

    @staticmethod
    def combine_all_hooks(hooks_list: list[HookGroup], require_count=0) -> HookGroup:
        actual: list[HookGroup] = []
        for group in hooks_list:
            if group is not None:
                actual.append(group)
        if len(actual) < require_count:
            raise Exception(f"Need at least {require_count} hooks to combine, but only had {len(actual)}.")
        # if no hooks, then return None
        if len(actual) == 0:
            return None
        # if only 1 hook, just return itself without cloning
        elif len(actual) == 1:
            return actual[0]
        final_hook: HookGroup = None
        for hook in actual:
            if final_hook is None:
                final_hook = hook.clone()
            else:
                final_hook = final_hook.clone_and_combine(hook)
        return final_hook


class HookKeyframe:
    def __init__(self, strength: float, start_percent=0.0, guarantee_steps=1):
        self.strength = float(strength)
        self.start_percent = float(start_percent)
        self.guarantee_steps = int(guarantee_steps)
        if self.guarantee_steps < 1:
            raise ValueError("guarantee_steps must be positive")

    def clone(self):
        return HookKeyframe(
            strength=self.strength,
            start_percent=self.start_percent,
            guarantee_steps=self.guarantee_steps,
        )

class HookKeyframeGroup:
    def __init__(self):
        self.keyframes: list[HookKeyframe] = []

    @property
    def strength(self):
        return self.keyframes[0].strength if self.keyframes else 1.0

    @property
    def contribution_gain(self):
        if not self.keyframes:
            return ContributionGain()
        return ContributionGain(
            timeline=GainTimeline.from_keyframes(
                GainKeyframe(
                    keyframe_id=f"hook.{index}",
                    anchor=keyframe.start_percent,
                    gain=keyframe.strength,
                    minimum_realized_steps=keyframe.guarantee_steps,
                )
                for index, keyframe in enumerate(self.keyframes)
            )
        )

    def add(self, keyframe: HookKeyframe):
        self.keyframes.append(keyframe)
        self.keyframes = get_sorted_list_via_attr(self.keyframes, "start_percent")

    def has_guarantee_steps(self):
        for kf in self.keyframes:
            if kf.guarantee_steps > 0:
                return True
        return False

    def has_index(self, index: int):
        return index >= 0 and index < len(self.keyframes)

    def is_empty(self):
        return len(self.keyframes) == 0

    def clone(self):
        c = HookKeyframeGroup()
        for keyframe in self.keyframes:
            c.keyframes.append(keyframe.clone())
        return c


class InterpolationMethod:
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"

    _LIST = [LINEAR, EASE_IN, EASE_OUT, EASE_IN_OUT]

    @classmethod
    def get_weights(cls, num_from: float, num_to: float, length: int, method: str, reverse=False):
        diff = num_to - num_from
        if method == cls.LINEAR:
            weights = torch.linspace(num_from, num_to, length)
        elif method == cls.EASE_IN:
            index = torch.linspace(0, 1, length)
            weights = diff * np.power(index, 2) + num_from
        elif method == cls.EASE_OUT:
            index = torch.linspace(0, 1, length)
            weights = diff * (1 - np.power(1 - index, 2)) + num_from
        elif method == cls.EASE_IN_OUT:
            index = torch.linspace(0, 1, length)
            weights = diff * ((1 - np.cos(index * np.pi)) / 2) + num_from
        else:
            raise ValueError(f"Unrecognized interpolation method '{method}'.")
        if reverse:
            weights = weights.flip(dims=(0,))
        return weights

def get_sorted_list_via_attr(objects: list, attr: str) -> list:
    if not objects:
        return objects
    elif len(objects) <= 1:
        return [x for x in objects]
    # now that we know we have to sort, do it following these rules:
    # a) if objects have same value of attribute, maintain their relative order
    # b) perform sorting of the groups of objects with same attributes
    unique_attrs = {}
    for o in objects:
        val_attr = getattr(o, attr)
        attr_list: list = unique_attrs.get(val_attr, list())
        attr_list.append(o)
        if val_attr not in unique_attrs:
            unique_attrs[val_attr] = attr_list
    # now that we have the unique attr values grouped together in relative order, sort them by key
    sorted_attrs = dict(sorted(unique_attrs.items()))
    # now flatten out the dict into a list to return
    sorted_list = []
    for object_list in sorted_attrs.values():
        sorted_list.extend(object_list)
    return sorted_list

def create_transformer_options_from_hooks(model: ModelPatcher, hooks: HookGroup,  transformer_options: dict[str]=None):
    # if no hooks or is not a ModelPatcher for sampling, return empty dict
    if hooks is None or model.is_clip:
        return {}
    if transformer_options is None:
        transformer_options = {}
    for hook in hooks.get_type(EnumHookType.TransformerOptions):
        hook: TransformerOptionsHook
        hook.on_apply_hooks(model, transformer_options)
    return transformer_options

def create_hook_lora(lora: dict[str, torch.Tensor], strength_model: float, strength_clip: float):
    hook_group = HookGroup()
    hook = WeightHook(strength_model=strength_model, strength_clip=strength_clip)
    hook_group.add(hook)
    hook.weights = lora
    return hook_group

def create_hook_model_as_lora(weights_model, weights_clip, strength_model: float, strength_clip: float):
    hook_group = HookGroup()
    hook = WeightHook(strength_model=strength_model, strength_clip=strength_clip)
    hook_group.add(hook)
    patches_model = None
    patches_clip = None
    if weights_model is not None:
        patches_model = {}
        for key in weights_model:
            patches_model[key] = ("model_as_lora", (weights_model[key],))
    if weights_clip is not None:
        patches_clip = {}
        for key in weights_clip:
            patches_clip[key] = ("model_as_lora", (weights_clip[key],))
    hook.weights = patches_model
    hook.weights_clip = patches_clip
    hook.need_weight_init = False
    return hook_group

def get_patch_weights_from_model(model: ModelPatcher, discard_model_sampling=True):
    if model is None:
        return None
    patches_model: dict[str, torch.Tensor] = model.model.state_dict()
    if discard_model_sampling:
        # do not include ANY model_sampling components of the model that should act as a patch
        for key in list(patches_model.keys()):
            if key.startswith("model_sampling"):
                patches_model.pop(key, None)
    return patches_model

# NOTE: this function shows how to register weight hooks directly on the ModelPatchers
def load_hook_lora_for_models(model: ModelPatcher, clip: CLIP, lora: dict[str, torch.Tensor],
                              strength_model: float, strength_clip: float):
    key_map = {}
    if model is not None:
        key_map = dinkster_comfy.lora.model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = dinkster_comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

    hook_group = HookGroup()
    hook = WeightHook()
    hook_group.add(hook)
    loaded: dict[str] = dinkster_comfy.lora.load_lora(lora, key_map)
    if model is not None:
        new_modelpatcher = model.clone()
        k = new_modelpatcher.add_hook_patches(hook=hook, patches=loaded, strength_patch=strength_model)
    else:
        k = ()
        new_modelpatcher = None

    if clip is not None:
        new_clip = clip.clone()
        k1 = new_clip.patcher.add_hook_patches(hook=hook, patches=loaded, strength_patch=strength_clip)
    else:
        k1 = ()
        new_clip = None
    k = set(k)
    k1 = set(k1)
    for x in loaded:
        if (x not in k) and (x not in k1):
            logging.warning(f"NOT LOADED {x}")
    return (new_modelpatcher, new_clip, hook_group)

def _combine_hooks_from_values(c_dict: dict[str, HookGroup], values: dict[str, HookGroup], cache: dict[tuple[HookGroup, HookGroup], HookGroup]):
    hooks_key = 'hooks'
    # if hooks only exist in one dict, do what's needed so that it ends up in c_dict
    if hooks_key not in values:
        return
    if hooks_key not in c_dict:
        hooks_value = values.get(hooks_key, None)
        if hooks_value is not None:
            c_dict[hooks_key] = hooks_value
        return
    # otherwise, need to combine with minimum duplication via cache
    hooks_tuple = (c_dict[hooks_key], values[hooks_key])
    cached_hooks = cache.get(hooks_tuple, None)
    if cached_hooks is None:
        new_hooks = hooks_tuple[0].clone_and_combine(hooks_tuple[1])
        cache[hooks_tuple] = new_hooks
        c_dict[hooks_key] = new_hooks
    else:
        c_dict[hooks_key] = cache[hooks_tuple]

def conditioning_set_values_with_hooks(conditioning, values={}, append_hooks=True,
                                       cache: dict[tuple[HookGroup, HookGroup], HookGroup]=None):
    c = []
    if cache is None:
        cache = {}
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            if append_hooks and k == 'hooks':
                _combine_hooks_from_values(n[1], values, cache)
            else:
                n[1][k] = values[k]
        c.append(n)

    return c

def set_hooks_for_conditioning(cond, hooks: HookGroup, append_hooks=True, cache: dict[tuple[HookGroup, HookGroup], HookGroup]=None):
    if hooks is None:
        return cond
    return conditioning_set_values_with_hooks(cond, {'hooks': hooks}, append_hooks=append_hooks, cache=cache)

def set_timesteps_for_conditioning(cond, timestep_range: tuple[float,float]):
    if timestep_range is None:
        return cond
    updated = []
    timeline = GainTimeline.active_range(*timestep_range)
    for tensor, metadata in cond:
        metadata = metadata.copy()
        gain = metadata.get("contribution_gain", ContributionGain())
        metadata["contribution_gain"] = gain.with_timeline(timeline)
        updated.append([tensor, metadata])
    return updated

def set_mask_for_conditioning(cond, mask: torch.Tensor, set_cond_area: str, strength: float):
    if mask is None:
        return cond
    set_area_to_bounds = False
    if set_cond_area != 'default':
        set_area_to_bounds = True
    if len(mask.shape) < 3:
        mask = mask.unsqueeze(0)
    updated = []
    for tensor, metadata in cond:
        metadata = metadata.copy()
        gain = metadata.get("contribution_gain", ContributionGain())
        metadata["contribution_gain"] = gain.scaled(strength)
        metadata["mask"] = mask
        metadata["effect_masks"] = (mask,)
        metadata["set_area_to_bounds"] = set_area_to_bounds
        updated.append([tensor, metadata])
    return updated

def combine_conditioning(conds: list):
    combined_conds = []
    for cond in conds:
        combined_conds.extend(cond)
    return combined_conds

def combine_with_new_conds(conds: list, new_conds: list):
    combined_conds = []
    for c, new_c in zip(conds, new_conds):
        combined_conds.append(combine_conditioning([c, new_c]))
    return combined_conds

def set_conds_props(conds: list, strength: float, set_cond_area: str,
                   mask: torch.Tensor=None, hooks: HookGroup=None, timesteps_range: tuple[float,float]=None, append_hooks=True):
    final_conds = []
    cache = {}
    for c in conds:
        # first, apply lora_hook to conditioning, if provided
        c = set_hooks_for_conditioning(c, hooks, append_hooks=append_hooks, cache=cache)
        # next, apply mask to conditioning
        c = set_mask_for_conditioning(cond=c, mask=mask, strength=strength, set_cond_area=set_cond_area)
        # apply timesteps, if present
        c = set_timesteps_for_conditioning(cond=c, timestep_range=timesteps_range)
        # finally, apply mask to conditioning and store
        final_conds.append(c)
    return final_conds

def set_conds_props_and_combine(conds: list, new_conds: list, strength: float=1.0, set_cond_area: str="default",
                               mask: torch.Tensor=None, hooks: HookGroup=None, timesteps_range: tuple[float,float]=None, append_hooks=True):
    combined_conds = []
    cache = {}
    for c, masked_c in zip(conds, new_conds):
        # first, apply lora_hook to new conditioning, if provided
        masked_c = set_hooks_for_conditioning(masked_c, hooks, append_hooks=append_hooks, cache=cache)
        # next, apply mask to new conditioning, if provided
        masked_c = set_mask_for_conditioning(cond=masked_c, mask=mask, set_cond_area=set_cond_area, strength=strength)
        # apply timesteps, if present
        masked_c = set_timesteps_for_conditioning(cond=masked_c, timestep_range=timesteps_range)
        # finally, combine with existing conditioning and store
        combined_conds.append(combine_conditioning([c, masked_c]))
    return combined_conds

def set_default_conds_and_combine(conds: list, new_conds: list,
                                   hooks: HookGroup=None, timesteps_range: tuple[float,float]=None, append_hooks=True):
    combined_conds = []
    cache = {}
    for c, new_c in zip(conds, new_conds):
        # first, apply lora_hook to new conditioning, if provided
        new_c = set_hooks_for_conditioning(new_c, hooks, append_hooks=append_hooks, cache=cache)
        # next, add default_cond key to cond so that during sampling, it can be identified
        new_c = conditioning_set_values(new_c, {'default': True})
        # apply timesteps, if present
        new_c = set_timesteps_for_conditioning(cond=new_c, timestep_range=timesteps_range)
        # finally, combine with existing conditioning and store
        combined_conds.append(combine_conditioning([c, new_c]))
    return combined_conds
