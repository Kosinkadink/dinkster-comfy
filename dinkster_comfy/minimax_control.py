"""MiniMax H3 Fun ControlNet model-patch loading and application."""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from . import model_management, ops, patcher_extension, utils
from .ldm.minimax.controlnet import MiniMaxH3FunControl, is_minimax_h3_fun_state_dict
from .ldm.minimax.vae import IMAGENET_MEAN
from .model_patcher import CoreModelPatcher
from .patch_program import PatchResource
from .window_execution import WindowTensorLayout, gather_window_tensor
from .window_plan import CompositeWindowPlan, JointWindow


def _control_config(state_dict, metadata):
    num_blocks = 0
    while f"control_blocks.{num_blocks}.after_proj.weight" in state_dict:
        num_blocks += 1
    injection_layers = tuple(range(0, 50, 50 // num_blocks))
    if metadata is not None and "control_blocks_places" in metadata:
        injection_layers = tuple(json.loads(metadata["control_blocks_places"]))
        if len(injection_layers) != num_blocks:
            raise ValueError("MiniMax H3 Fun control_blocks_places metadata does not match the checkpoint")
    qkv = state_dict["control_blocks.0.attn.qkv_proj.weight"]
    head_dim = state_dict["control_blocks.0.attn.q_norm.weight"].shape[0]
    use_adaln_curves = metadata is not None and metadata.get("minimax_h3_fun_controlnet") == "adaln_basis"
    return {
        "control_in_dim": 49,
        "injection_layers": injection_layers,
        "inpaint_post_norm": metadata is not None and metadata.get("inpaint_masked_pixel_mode") == "post_norm",
        "hidden_size": state_dict["control_proj_in.weight"].shape[0],
        "num_attention_heads": qkv.shape[0] // (3 * head_dim),
        "attention_head_dim": head_dim,
        "ffn_hidden_size": state_dict["control_blocks.0.mlp.fc1.weight"].shape[0] // 2,
        "time_embed_dim": 8 if use_adaln_curves else 2688,
        "use_adaln_curves": use_adaln_curves,
    }


def load_minimax_h3_fun_control_patch(path):
    state_dict, metadata = utils.load_torch_file(path, safe_load=True, return_metadata=True)
    if not is_minimax_h3_fun_state_dict(state_dict):
        raise ValueError("checkpoint is not a MiniMax H3 Fun ControlNet model patch")
    load_device = model_management.get_torch_device()
    quant = utils.detect_layer_quantization(state_dict, "")
    if quant is not None:
        dtype = torch.bfloat16
        operations = ops.mixed_precision_ops(quant, dtype)
    else:
        dtype = model_management.unet_dtype(
            model_params=-1,
            supported_dtypes=[torch.bfloat16, torch.float32],
            weight_dtype=utils.weight_dtype(state_dict),
        )
        manual_cast_dtype = model_management.unet_manual_cast(
            dtype,
            load_device,
            supported_dtypes=[torch.bfloat16, torch.float32],
        )
        operations = ops.pick_operations(dtype, manual_cast_dtype)
    model = MiniMaxH3FunControl(
        **_control_config(state_dict, metadata),
        operations=operations,
        device=model_management.unet_offload_device(),
        dtype=dtype,
    )
    patcher = CoreModelPatcher(
        model,
        load_device=load_device,
        offload_device=model_management.unet_offload_device(),
    )
    model.load_state_dict(state_dict, assign=patcher.is_dynamic())
    return patcher


def _video_window_layout(plan: CompositeWindowPlan) -> WindowTensorLayout:
    kind = next((kind for kind in plan.kinds if kind.name == "video"), None)
    if kind is None:
        raise ValueError("window plan has no video tensor kind")
    dimensions = {"temporal": 2, "height": 3, "width": 4}
    try:
        axis_dimensions = tuple(
            sorted((mapping.axis, dimensions[mapping.axis]) for mapping in kind.axis_maps)
        )
    except KeyError as error:
        raise ValueError(f"MiniMax H3 video windows do not support axis {error.args[0]!r}") from None
    return WindowTensorLayout("video", axis_dimensions)


def _full_video_shape(shape, plan: CompositeWindowPlan, layout: WindowTensorLayout):
    result = list(shape)
    kind = next(kind for kind in plan.kinds if kind.name == "video")
    extents = {mapping.axis: mapping.extent for mapping in kind.axis_maps}
    for axis, dimension in layout.axis_dimensions:
        result[dimension] = extents[axis]
    return tuple(result)


class MiniMaxH3FunControlPatch:
    def __init__(self, model_patch, vae, control_video, mask, source_video, strength, sigma_start, sigma_end):
        self.model_patch = model_patch
        self.vae = vae
        self.control_video = control_video
        self.mask = mask
        self.source_video = source_video
        self.strength = strength
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.control_latent = None
        self.control_latent_shape = None
        self.window_control_latent = None
        self.control_stream = None
        self.active = False
        self._program_descriptor = None

    def patch_program_descriptor(self):
        if self._program_descriptor is None:
            self._program_descriptor = {
                "type": f"{type(self).__module__}.{type(self).__qualname__}",
                "model_patch": self.model_patch.patch_program_descriptor(),
                "vae": self.vae.patcher.patch_program_descriptor(),
                "control_video": None if self.control_video is None else PatchResource.bind(self.control_video).identity,
                "mask": None if self.mask is None else PatchResource.bind(self.mask).identity,
                "source_video": None if self.source_video is None else PatchResource.bind(self.source_video).identity,
                "strength": self.strength,
                "sigma_start": self.sigma_start,
                "sigma_end": self.sigma_end,
            }
        return self._program_descriptor

    def _fit_frames(self, frames, frame_count, width, height):
        indices = torch.arange(frame_count, device=frames.device).clamp(max=frames.shape[0] - 1)
        return utils.common_upscale(frames[indices], width, height, "bilinear", "center")

    def _encode(self, frames, target_shape):
        latent = self.vae.encode(frames.movedim(1, -1)).to(torch.float32)
        if tuple(latent.shape) != target_shape:
            raise ValueError(f"MiniMax H3 Fun VAE output shape {tuple(latent.shape)} does not match the target {target_shape}")
        return latent

    def prepare_control_latent(self, target_shape):
        target_shape = tuple(target_shape)
        if self.control_latent is not None and self.control_latent_shape == target_shape:
            return

        latent_frames, latent_height, latent_width = target_shape[2:]
        frame_count = max((latent_frames - 2) // 5, 0) * 17 + 5
        spatial_compression = self.vae.spacial_compression_encode()
        width = latent_width * spatial_compression
        height = latent_height * spatial_compression
        loaded_models = model_management.loaded_models(only_currently_used=True)
        try:
            hint = None
            if self.control_video is not None:
                hint = self._encode(
                    self._fit_frames(self.control_video, frame_count, width, height),
                    target_shape,
                )
            if self.mask is not None:
                mask = (self.mask.reshape(-1, 1, self.mask.shape[-2], self.mask.shape[-1]) > 0.5).to(torch.float32)
                indices = torch.arange(frame_count, device=mask.device).clamp(max=mask.shape[0] - 1)
                mask = utils.common_upscale(mask[indices], width, height, "bilinear", "center")
                visibility = 1.0 - (mask > 0.5).to(torch.float32)
                if self.source_video is None:
                    source = torch.zeros(frame_count, 3, height, width, dtype=visibility.dtype, device=visibility.device)
                else:
                    source = self._fit_frames(self.source_video, frame_count, width, height)
                visibility = visibility.to(source.device)
                masked = source * visibility
                if self.model_patch.model.inpaint_post_norm:
                    masked += (1.0 - visibility) * torch.tensor(
                        IMAGENET_MEAN,
                        dtype=source.dtype,
                        device=source.device,
                    ).view(1, 3, 1, 1)
                masked_latent = self._encode(masked, target_shape)
                if hint is None:
                    hint = torch.zeros_like(masked_latent)
                visibility_latent = F.interpolate(
                    visibility.squeeze(1)[None, None],
                    size=(latent_frames, latent_height, latent_width),
                    mode="trilinear",
                    align_corners=False,
                )
                hint = torch.cat(
                    [hint, visibility_latent.to(hint.device), masked_latent.to(hint.device)],
                    dim=1,
                )
        finally:
            model_management.load_models_gpu(loaded_models)
        self.control_latent = hint
        self.control_latent_shape = target_shape

    def diffusion_model_wrapper(self, executor, x, timestep, context, transformer_options, **kwargs):
        sigmas = transformer_options.get("sigmas")
        sigma = float(sigmas[0]) if sigmas is not None else float(timestep.flatten()[0]) / 1000.0
        self.active = self.sigma_end <= sigma <= self.sigma_start
        self.window_control_latent = None
        self.control_stream = None
        if self.active:
            target_shape = tuple(x[0].shape)
            plan = transformer_options.get("window_plan")
            window = transformer_options.get("window")
            if plan is not None or window is not None:
                if type(plan) is not CompositeWindowPlan or type(window) is not JointWindow:
                    raise TypeError("MiniMax H3 control windows require a compiled plan and joint window")
                layout = _video_window_layout(plan)
                target_shape = _full_video_shape(target_shape, plan, layout)
                self.prepare_control_latent(target_shape)
                self.window_control_latent = gather_window_tensor(self.control_latent, layout, window)
            else:
                self.prepare_control_latent(target_shape)
                self.window_control_latent = self.control_latent
        try:
            return executor(x, timestep, context, transformer_options, **kwargs)
        finally:
            self.window_control_latent = None
            self.control_stream = None

    def before_block(self, block_index, args):
        if not self.active or block_index != self.model_patch.model.injection_layers[0]:
            return
        control_latent = self.window_control_latent.to(args["img"].device)
        self.control_stream = self.model_patch.model.init_stream(
            args["img"],
            control_latent,
            args["layout"],
            args["t_emb"],
        )

    def after_block(self, block_index, args, out):
        if not self.active:
            return out
        control_index = self.model_patch.model.injection_layers.index(block_index)
        self.control_stream, skip = self.model_patch.model.step(
            control_index,
            self.control_stream,
            args["t_emb"],
            args["mod_segments"],
            args["rope_freqs"],
            transformer_options=args["transformer_options"],
        )
        skip[args["layout"].audio_pos.to(skip.device)] = 0
        out["img"].add_(skip, alpha=self.strength)
        return out

    def to(self, device_or_dtype):
        if isinstance(device_or_dtype, torch.device):
            if self.control_latent is not None:
                self.control_latent = self.control_latent.to(device_or_dtype)
            self.window_control_latent = None
            self.control_stream = None
        return self

    def cleanup(self):
        self.control_latent = None
        self.control_latent_shape = None
        self.window_control_latent = None
        self.control_stream = None
        self.active = False

    def models(self):
        return [self.model_patch]

    def register(self, model):
        model.add_wrapper(patcher_extension.WrappersMP.DIFFUSION_MODEL, self.diffusion_model_wrapper)
        for block_index in self.model_patch.model.injection_layers:
            blocks_replace = model.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
            previous = blocks_replace.get(("double_block", block_index))
            model.set_model_patch_replace(
                MiniMaxH3FunControlBlockPatch(self, block_index, previous),
                "dit",
                "double_block",
                block_index,
            )


class MiniMaxH3FunControlBlockPatch:
    def __init__(self, control_patch, block_index, previous):
        self.control_patch = control_patch
        self.block_index = block_index
        self.previous = previous

    def patch_program_descriptor(self):
        return {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "control": self.control_patch.patch_program_descriptor(),
            "block_index": self.block_index,
            "previous": self.previous,
        }

    def __call__(self, args, extra_args):
        self.control_patch.before_block(self.block_index, args)
        if self.previous is None:
            out = extra_args["original_block"](args)
        else:
            out = self.previous(args, extra_args)
        return self.control_patch.after_block(self.block_index, args, out)

    def to(self, device_or_dtype):
        self.control_patch.to(device_or_dtype)
        if hasattr(self.previous, "to"):
            self.previous = self.previous.to(device_or_dtype)
        return self

    def cleanup(self):
        self.control_patch.cleanup()
        if hasattr(self.previous, "cleanup"):
            self.previous.cleanup()

    def models(self):
        models = self.control_patch.models()
        if hasattr(self.previous, "models"):
            models += self.previous.models()
        return models


def apply_minimax_h3_fun_control(
    model,
    model_patch,
    vae,
    strength,
    start_percent=0.0,
    end_percent=1.0,
    control_video=None,
    mask=None,
    source_video=None,
):
    if strength == 0 or (control_video is None and mask is None):
        return model
    model_sampling = model.get_model_object("model_sampling")
    patch = MiniMaxH3FunControlPatch(
        model_patch,
        vae,
        control_video,
        mask,
        source_video if mask is not None else None,
        strength,
        float(model_sampling.percent_to_sigma(start_percent)),
        float(model_sampling.percent_to_sigma(end_percent)),
    )
    patched = model.clone()
    patch.register(patched)
    return patched
