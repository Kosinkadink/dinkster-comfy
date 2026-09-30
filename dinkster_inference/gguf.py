"""GGUF loading and GGML block decoding."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from types import MappingProxyType

import gguf
import torch
from sentencepiece import sentencepiece_model_pb2


class GGUFError(ValueError):
    pass


GGUF_RESIDENCY_MODES = ("memory", "balanced", "eager")

GGUF_BLOCK_SHAPES = MappingProxyType(
    {
        "Q4_0": (32, 18),
        "Q8_0": (32, 34),
        "Q4_K": (256, 144),
        "Q5_K": (256, 176),
        "Q6_K": (256, 210),
    }
)


def _fp16_column(blocks, offset):
    return blocks[:, offset : offset + 2].contiguous().view(torch.float16).to(torch.float32)


def decode_q8_0_blocks(blocks, logical_shape, dtype=torch.float32):
    scales = blocks[:, :2].contiguous().view(torch.float16).to(dtype)
    quants = blocks[:, 2:].contiguous().view(torch.int8)
    return (quants * scales).reshape(logical_shape)


def decode_q4_0_blocks(blocks, logical_shape):
    scales = _fp16_column(blocks, 0)
    packed = blocks[:, 2:]
    quants = torch.cat((packed & 0x0F, packed >> 4), dim=1).to(torch.float32)
    return quants.sub_(8.0).mul_(scales).reshape(logical_shape)


def _kquant_scale_mins(blocks):
    d = _fp16_column(blocks, 0)
    dmin = _fp16_column(blocks, 2)
    packed = blocks[:, 4:16]
    low, mid, high = packed[:, 0:4], packed[:, 4:8], packed[:, 8:12]
    scales = torch.cat((low & 0x3F, (high & 0x0F) | ((low >> 6) << 4)), dim=1)
    mins = torch.cat((mid & 0x3F, (high >> 4) | ((mid >> 6) << 4)), dim=1)
    return d * scales.to(torch.float32), dmin * mins.to(torch.float32)


def decode_q4_k_blocks(blocks, logical_shape):
    group_scales, group_mins = _kquant_scale_mins(blocks)
    quants = blocks[:, 16:].reshape(-1, 4, 1, 32)
    q = torch.cat((quants & 0x0F, quants >> 4), dim=2).to(torch.float32)
    scales = group_scales.reshape(-1, 4, 2, 1)
    mins = group_mins.reshape(-1, 4, 2, 1)
    return q.mul_(scales).sub_(mins).reshape(logical_shape)


def decode_q5_k_blocks(blocks, logical_shape):
    group_scales, group_mins = _kquant_scale_mins(blocks)
    high = blocks[:, 16:48].reshape(-1, 1, 1, 32)
    low = blocks[:, 48:].reshape(-1, 4, 1, 32)
    shifts = torch.arange(8, dtype=torch.uint8, device=blocks.device).reshape(1, 4, 2, 1)
    bits = (high >> shifts) & 1
    q = (torch.cat((low & 0x0F, low >> 4), dim=2) | (bits << 4)).to(torch.float32)
    scales = group_scales.reshape(-1, 4, 2, 1)
    mins = group_mins.reshape(-1, 4, 2, 1)
    return q.mul_(scales).sub_(mins).reshape(logical_shape)


def decode_q6_k_blocks(blocks, logical_shape):
    low = blocks[:, :128].reshape(-1, 2, 2, 32)
    high = blocks[:, 128:192].reshape(-1, 2, 32)
    scales = blocks[:, 192:208].contiguous().view(torch.int8)
    d = _fp16_column(blocks, 208)
    low_a, low_b = low[:, :, 0], low[:, :, 1]
    q = (
        torch.stack(
            (
                (low_a & 0x0F) | ((high & 3) << 4),
                (low_b & 0x0F) | (((high >> 2) & 3) << 4),
                (low_a >> 4) | (((high >> 4) & 3) << 4),
                (low_b >> 4) | (((high >> 6) & 3) << 4),
            ),
            dim=2,
        ).to(torch.float32)
        - 32.0
    )
    group_scales = scales.reshape(-1, 2, 4, 2).to(torch.float32).repeat_interleave(16, dim=3)
    group_scales.mul_(d.reshape(-1, 1, 1, 1))
    return q.mul_(group_scales).reshape(logical_shape)


GGUF_BLOCK_DECODERS = MappingProxyType(
    {
        "Q4_0": decode_q4_0_blocks,
        "Q8_0": decode_q8_0_blocks,
        "Q4_K": decode_q4_k_blocks,
        "Q5_K": decode_q5_k_blocks,
        "Q6_K": decode_q6_k_blocks,
    }
)


class GGUFWeightTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, *, ggml_type, tensor_shape):
        return torch.Tensor._make_subclass(cls, data, False)

    def __init__(self, data, *, ggml_type, tensor_shape):
        self.ggml_type = ggml_type
        self.tensor_shape = torch.Size(tensor_shape)

    @property
    def shape(self):
        return self.tensor_shape

    def to(self, *args, **kwargs):
        data = super().to(*args, **kwargs)
        if type(data) is GGUFWeightTensor:
            data.ggml_type = self.ggml_type
            data.tensor_shape = self.tensor_shape
            return data
        return GGUFWeightTensor(data, ggml_type=self.ggml_type, tensor_shape=self.tensor_shape)

    def clone(self, *args, **kwargs):
        data = self.as_subclass(torch.Tensor).clone(*args, **kwargs)
        return GGUFWeightTensor(data, ggml_type=self.ggml_type, tensor_shape=self.tensor_shape)

    def detach(self):
        return self


def is_encoded_gguf_tensor(tensor):
    return isinstance(tensor, GGUFWeightTensor)


def decode_gguf_tensor(tensor, dtype=None):
    if not is_encoded_gguf_tensor(tensor):
        return tensor
    type_name = tensor.ggml_type.name
    shape = GGUF_BLOCK_SHAPES.get(type_name)
    if shape is None:
        raise GGUFError(f"unsupported encoded GGML type: {type_name}")
    block_elements, block_bytes = shape
    elements = tensor.tensor_shape.numel()
    if elements % block_elements:
        raise GGUFError(f"{type_name} tensor shape does not align to {block_elements} elements")
    raw = tensor.as_subclass(torch.Tensor).view(torch.uint8)
    if raw.numel() != elements // block_elements * block_bytes:
        raise GGUFError(f"{type_name} tensor byte count does not match its logical shape")
    blocks = raw.reshape(-1, block_bytes)
    if dtype is not None and dtype != torch.float32:
        if type_name == "Q8_0":
            return decode_q8_0_blocks(blocks, tensor.tensor_shape, dtype=dtype)
        output = torch.empty(elements, dtype=dtype, device=blocks.device)
        blocks_per_chunk = max(1, (16 * 1024 * 1024) // (block_elements * 4))
        for start in range(0, blocks.shape[0], blocks_per_chunk):
            chunk = blocks[start : start + blocks_per_chunk]
            decoded = GGUF_BLOCK_DECODERS[type_name](
                chunk, (chunk.shape[0] * block_elements,)
            )
            first = start * block_elements
            output[first : first + decoded.numel()].copy_(decoded)
        return output.reshape(tensor.tensor_shape)
    return GGUF_BLOCK_DECODERS[type_name](blocks, tuple(tensor.tensor_shape))


def decode_gguf_rows(tensor, rows, dtype=None):
    if not is_encoded_gguf_tensor(tensor) or len(tensor.tensor_shape) != 2:
        raise GGUFError("GGUF row decode requires an encoded matrix")
    type_name = tensor.ggml_type.name
    shape = GGUF_BLOCK_SHAPES.get(type_name)
    if shape is None:
        raise GGUFError(f"unsupported encoded GGML type: {type_name}")
    block_elements, block_bytes = shape
    row_width = tensor.tensor_shape[1]
    if row_width % block_elements:
        raise GGUFError(f"{type_name} matrix rows do not align to {block_elements} elements")
    blocks_per_row = row_width // block_elements
    raw = tensor.as_subclass(torch.Tensor).view(torch.uint8).reshape(-1, block_bytes)
    offsets = torch.arange(blocks_per_row, device=rows.device)
    block_indices = (rows.reshape(-1, 1) * blocks_per_row + offsets).reshape(-1)
    blocks = raw.index_select(0, block_indices)
    if type_name == "Q8_0" and dtype is not None:
        return decode_q8_0_blocks(blocks, (rows.numel(), row_width), dtype=dtype)
    return GGUF_BLOCK_DECODERS[type_name](blocks, (rows.numel(), row_width))


@dataclass(frozen=True)
class GGUFLoadInfo:
    architecture: str
    residency_mode: str
    decoded_budget_bytes: int
    decoded_weight_bytes: int
    encoded_weight_bytes: int
    metadata: dict[str, object]


_IMAGE_ARCHITECTURES = frozenset({"sdxl"})
_TEXT_ARCHITECTURES = frozenset({"t5", "t5encoder"})
_T5_REPLACEMENTS = {
    "enc.": "encoder.",
    ".blk.": ".block.",
    "token_embd": "shared",
    "output_norm": "final_layer_norm",
    "attn_q": "layer.0.SelfAttention.q",
    "attn_k": "layer.0.SelfAttention.k",
    "attn_v": "layer.0.SelfAttention.v",
    "attn_o": "layer.0.SelfAttention.o",
    "attn_norm": "layer.0.layer_norm",
    "attn_rel_b": "layer.0.SelfAttention.relative_attention_bias",
    "ffn_up": "layer.1.DenseReluDense.wi_1",
    "ffn_down": "layer.1.DenseReluDense.wo",
    "ffn_gate": "layer.1.DenseReluDense.wi_0",
    "ffn_norm": "layer.1.layer_norm",
}


def _field(reader, name, expected_type):
    value = reader.get_field(name)
    if value is None:
        return None
    if expected_type is str:
        if len(value.types) != 1 or value.types[0] != gguf.GGUFValueType.STRING:
            raise GGUFError(f"GGUF field {name!r} must be a string")
        return str(value.parts[value.data[-1]], encoding="utf-8")
    return expected_type(value.parts[value.data[-1]].item())


def _list_field(reader, name, expected_type):
    value = reader.get_field(name)
    if value is None:
        raise GGUFError(f"missing GGUF field {name!r}")
    if expected_type is str:
        return tuple(str(value.parts[index], encoding="utf-8") for index in value.data)
    return tuple(expected_type(value.parts[index][0]) for index in value.data)


def _metadata(reader):
    output = {}
    scalar_types = {
        gguf.GGUFValueType.STRING: lambda value: str(value, "utf-8"),
        gguf.GGUFValueType.INT32: int,
        gguf.GGUFValueType.FLOAT32: float,
        gguf.GGUFValueType.BOOL: bool,
    }
    for name in reader.fields:
        field = reader.get_field(name)
        if len(field.types) != 1 or field.types[0] not in scalar_types:
            continue
        value = field.parts[field.data[-1]]
        if field.types[0] != gguf.GGUFValueType.STRING:
            value = value.item()
        output[name] = scalar_types[field.types[0]](value)
    return output


def _umt5_tokenizer(reader):
    if _field(reader, "tokenizer.ggml.model", str) != "t5":
        raise GGUFError("UMT5 GGUF tokenizer metadata is not a T5 sentencepiece model")
    tokenizer = sentencepiece_model_pb2.ModelProto()
    tokenizer.trainer_spec.model_type = 1
    tokenizer.normalizer_spec.add_dummy_prefix = _field(
        reader, "tokenizer.ggml.add_space_prefix", bool
    )
    tokenizer.normalizer_spec.remove_extra_whitespaces = _field(
        reader, "tokenizer.ggml.remove_extra_whitespaces", bool
    )
    tokens = _list_field(reader, "tokenizer.ggml.tokens", str)
    scores = _list_field(reader, "tokenizer.ggml.scores", float)
    token_types = _list_field(reader, "tokenizer.ggml.token_type", int)
    if len(tokens) != len(scores) or len(tokens) != len(token_types):
        raise GGUFError("UMT5 GGUF tokenizer arrays have different lengths")
    for text, score, token_type in zip(tokens, scores, token_types):
        piece = tokenizer.SentencePiece()
        piece.piece = text
        piece.score = score
        piece.type = token_type
        tokenizer.pieces.append(piece)
    tokenizer.trainer_spec.byte_fallback = True
    tokenizer.trainer_spec.vocab_size = len(tokens)
    tokenizer.trainer_spec.max_sentence_length = 4096
    tokenizer.trainer_spec.eos_id = _field(reader, "tokenizer.ggml.eos_token_id", int)
    tokenizer.trainer_spec.pad_id = _field(reader, "tokenizer.ggml.padding_token_id", int)
    return torch.tensor(tuple(tokenizer.SerializeToString()), dtype=torch.uint8)


def _logical_shape(reader, tensor):
    field_name = f"comfy.gguf.orig_shape.{tensor.name}"
    field = reader.get_field(field_name)
    if field is None:
        return torch.Size(int(value) for value in reversed(tensor.shape))
    if (
        len(field.types) != 2
        or field.types[0] != gguf.GGUFValueType.ARRAY
        or field.types[1] != gguf.GGUFValueType.INT32
    ):
        raise GGUFError(f"GGUF field {field_name!r} must be an INT32 array")
    return torch.Size(int(field.parts[index][0]) for index in field.data)


def _model_key(name, *, prefix, has_prefix, text_model):
    if has_prefix:
        if not name.startswith(prefix):
            return None
        name = name[len(prefix) :]
    if text_model:
        for source, target in _T5_REPLACEMENTS.items():
            name = name.replace(source, target)
    return name


def _decoded_budget(load_device, encoded_bytes):
    from . import model_management

    free = model_management.get_free_memory(load_device)
    return max(0, int(free - model_management.minimum_inference_memory() - encoded_bytes))


def load_gguf_state_dict(
    path,
    *,
    text_model=False,
    residency_mode="memory",
    decoded_budget_bytes=None,
    load_device=None,
):
    if residency_mode not in GGUF_RESIDENCY_MODES:
        raise GGUFError(
            f"unknown GGUF residency mode {residency_mode!r}; expected one of {GGUF_RESIDENCY_MODES}"
        )
    if decoded_budget_bytes is not None and (
        type(decoded_budget_bytes) is not int or decoded_budget_bytes < 0
    ):
        raise GGUFError("decoded GGUF budget must be a non-negative byte count")

    reader = gguf.GGUFReader(path)
    architecture = _field(reader, "general.architecture", str)
    admitted = _TEXT_ARCHITECTURES if text_model else _IMAGE_ARCHITECTURES
    if architecture not in admitted:
        kind = "text encoder" if text_model else "diffusion model"
        raise GGUFError(f"unsupported GGUF {kind} architecture: {architecture!r}")

    prefix = "model.diffusion_model."
    names = {tensor.name for tensor in reader.tensors}
    has_prefix = not text_model and any(name.startswith(prefix) for name in names)
    tensors = []
    encoded_total = 0
    for tensor in reader.tensors:
        key = _model_key(
            tensor.name, prefix=prefix, has_prefix=has_prefix, text_model=text_model
        )
        if key is None:
            continue
        shape = _logical_shape(reader, tensor)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            data = torch.from_numpy(tensor.data)
        type_name = tensor.tensor_type.name
        if type_name in ("F32", "F16"):
            value = data.view(*shape)
        elif type_name == "BF16":
            value = (data.view(torch.int16).to(torch.int32) << 16).view(torch.float32).reshape(shape)
        elif type_name in GGUF_BLOCK_DECODERS:
            value = GGUFWeightTensor(data, ggml_type=tensor.tensor_type, tensor_shape=shape)
            encoded_total += data.nbytes
        else:
            raise GGUFError(f"unsupported GGML type {type_name!r} for tensor {tensor.name!r}")
        tensors.append((key, value))

    if residency_mode == "memory":
        budget = 0
    elif residency_mode == "eager":
        budget = sum(value.shape.numel() * 4 for _, value in tensors if is_encoded_gguf_tensor(value))
    elif decoded_budget_bytes is None:
        if load_device is None:
            from . import model_management

            load_device = model_management.get_torch_device()
        budget = _decoded_budget(load_device, encoded_total)
    else:
        budget = decoded_budget_bytes

    state_dict = {}
    decoded = 0
    quant_counts = {}
    for key, value in tensors:
        if is_encoded_gguf_tensor(value):
            type_name = value.ggml_type.name
            quant_counts[type_name] = quant_counts.get(type_name, 0) + 1
            decoded_size = value.tensor_shape.numel() * 4
            if decoded + decoded_size <= budget:
                value = decode_gguf_tensor(value)
                decoded += decoded_size
        state_dict[key] = value
    if text_model:
        state_dict["spiece_model"] = _umt5_tokenizer(reader)

    logging.info(
        "GGUF types: %s",
        ", ".join(f"{name} ({count})" for name, count in sorted(quant_counts.items())),
    )
    info = GGUFLoadInfo(
        architecture=architecture,
        residency_mode=residency_mode,
        decoded_budget_bytes=budget,
        decoded_weight_bytes=decoded,
        encoded_weight_bytes=encoded_total,
        metadata=_metadata(reader),
    )
    return state_dict, info
