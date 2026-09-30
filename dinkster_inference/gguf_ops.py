"""Comfy operations for encoded GGUF weights."""

import torch

from . import model_management, ops
from .gguf import decode_gguf_rows, decode_gguf_tensor, is_encoded_gguf_tensor


def _active(functions):
    return [function for function in functions if not hasattr(function, "active") or function.active()]


class GGUFLayer:
    comfy_cast_weights = True

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        weight = state_dict.get(prefix + "weight")
        bias = state_dict.get(prefix + "bias")
        if (
            not is_encoded_gguf_tensor(weight)
            and not is_encoded_gguf_tensor(bias)
            and not isinstance(self, torch.nn.Linear)
        ):
            return super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

        prefix_length = len(prefix)
        for key, value in state_dict.items():
            name = key[prefix_length:]
            if name == "weight":
                self.weight = torch.nn.Parameter(value, requires_grad=False)
            elif name == "bias" and value is not None:
                self.bias = torch.nn.Parameter(value, requires_grad=False)
            else:
                unexpected_keys.append(key)

        if self.weight is None and isinstance(self, torch.nn.Linear):
            missing_keys.append(prefix + "weight")

    def _gguf_weight(self, tensor, dtype, device, functions):
        if tensor is None:
            return None
        tensor = tensor.to(device=device, non_blocking=model_management.device_supports_non_blocking(device))
        if is_encoded_gguf_tensor(tensor):
            tensor = decode_gguf_tensor(tensor, dtype=dtype)
        elif tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        for function in _active(functions):
            tensor = function(tensor)
        return tensor

    def cast_bias_weight(self, input=None, dtype=None, device=None, bias_dtype=None):
        if input is not None:
            if dtype is None:
                dtype = input.dtype
            if bias_dtype is None:
                bias_dtype = dtype
            if device is None:
                device = input.device
        weight = self._gguf_weight(self.weight, dtype, device, self.weight_function)
        bias = self._gguf_weight(self.bias, bias_dtype, device, self.bias_function)
        return weight, bias

    def forward(self, *args, **kwargs):
        ops.run_every_op()
        if self.is_gguf_quantized():
            return self.forward_gguf(*args, **kwargs)
        return super().forward(*args, **kwargs)

    def is_gguf_quantized(self):
        return is_encoded_gguf_tensor(self.weight) or is_encoded_gguf_tensor(self.bias)


class GGUFOps(ops.manual_cast):
    class Linear(GGUFLayer, ops.manual_cast.Linear):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None
            self.comfy_need_lazy_init_bias = bias
            self.weight_comfy_model_dtype = dtype
            self.bias_comfy_model_dtype = dtype
            self.weight_function = []
            self.bias_function = []

        def forward_gguf(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.linear(input, weight, bias)

    class Conv2d(GGUFLayer, ops.manual_cast.Conv2d):
        def forward_gguf(self, input):
            weight, bias = self.cast_bias_weight(input)
            return self._conv_forward(input, weight, bias)

    class Embedding(GGUFLayer, ops.manual_cast.Embedding):
        def forward_gguf(self, input, out_dtype=None):
            output_dtype = out_dtype
            functions = _active(self.weight_function)
            if is_encoded_gguf_tensor(self.weight) and not functions:
                weight = self.weight.to(
                    device=input.device,
                    non_blocking=model_management.device_supports_non_blocking(input.device),
                )
                rows, inverse = torch.unique(input, sorted=True, return_inverse=True)
                weight = decode_gguf_rows(weight, rows, dtype=out_dtype)
                if out_dtype is not None:
                    weight = weight.to(dtype=out_dtype)
                return torch.nn.functional.embedding(
                    inverse.reshape(input.shape),
                    weight,
                    None,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                ).to(dtype=output_dtype)
            if self.weight.dtype in (torch.float16, torch.bfloat16):
                out_dtype = None
            weight, _ = self.cast_bias_weight(dtype=out_dtype, device=input.device)
            return torch.nn.functional.embedding(
                input,
                weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            ).to(dtype=output_dtype)

    class LayerNorm(GGUFLayer, ops.manual_cast.LayerNorm):
        def forward_gguf(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.layer_norm(
                input, self.normalized_shape, weight, bias, self.eps
            )

    class GroupNorm(GGUFLayer, ops.manual_cast.GroupNorm):
        def forward_gguf(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.group_norm(
                input, self.num_groups, weight, bias, self.eps
            )
