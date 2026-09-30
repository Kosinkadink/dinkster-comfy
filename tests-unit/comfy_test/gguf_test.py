import struct
from types import SimpleNamespace
from unittest import mock

import gguf
import pytest
import torch

from dinkster_inference import model_management, model_patcher, sd, utils
from dinkster_inference.gguf import (
    GGUF_BLOCK_DECODERS,
    GGUF_BLOCK_SHAPES,
    GGUFError,
    GGUFLoadInfo,
    GGUFWeightTensor,
    decode_gguf_rows,
    decode_gguf_tensor,
    load_gguf_state_dict,
)
from dinkster_inference.gguf_ops import GGUFOps


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _half(block, offset):
    return struct.unpack_from("<e", block, offset)[0]


def _scale_min(index, packed):
    if index < 4:
        return packed[index] & 0x3F, packed[index + 4] & 0x3F
    return (
        (packed[index + 4] & 0x0F) | ((packed[index - 4] >> 6) << 4),
        (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4),
    )


def _pure_q4_0(block):
    scale = _half(block, 0)
    quants = block[2:]
    return [
        *(_f32(scale * ((value & 0x0F) - 8)) for value in quants),
        *(_f32(scale * ((value >> 4) - 8)) for value in quants),
    ]


def _pure_q8_0(block):
    scale = _half(block, 0)
    return [_f32(scale * value) for value in struct.unpack_from("<32b", block, 2)]


def _pure_q4_k(block):
    scale = _half(block, 0)
    minimum = _half(block, 2)
    scales = block[4:16]
    quants = block[16:]
    output = []
    for chunk in range(4):
        first_scale, first_min = _scale_min(2 * chunk, scales)
        second_scale, second_min = _scale_min(2 * chunk + 1, scales)
        d1, m1 = _f32(scale * first_scale), _f32(minimum * first_min)
        d2, m2 = _f32(scale * second_scale), _f32(minimum * second_min)
        values = quants[32 * chunk : 32 * (chunk + 1)]
        output.extend(_f32(d1 * (value & 0x0F) - m1) for value in values)
        output.extend(_f32(d2 * (value >> 4) - m2) for value in values)
    return output


def _pure_q5_k(block):
    scale = _half(block, 0)
    minimum = _half(block, 2)
    scales = block[4:16]
    high = block[16:48]
    low = block[48:]
    output = []
    for chunk in range(4):
        first_scale, first_min = _scale_min(2 * chunk, scales)
        second_scale, second_min = _scale_min(2 * chunk + 1, scales)
        d1, m1 = _f32(scale * first_scale), _f32(minimum * first_min)
        d2, m2 = _f32(scale * second_scale), _f32(minimum * second_min)
        values = low[32 * chunk : 32 * (chunk + 1)]
        first_mask, second_mask = 1 << (2 * chunk), 2 << (2 * chunk)
        output.extend(
            _f32(d1 * ((value & 0x0F) + (16 if high[index] & first_mask else 0)) - m1)
            for index, value in enumerate(values)
        )
        output.extend(
            _f32(d2 * ((value >> 4) + (16 if high[index] & second_mask else 0)) - m2)
            for index, value in enumerate(values)
        )
    return output


def _pure_q6_k(block):
    low = block[:128]
    high = block[128:192]
    scales = struct.unpack_from("<16b", block, 192)
    scale = _half(block, 208)
    output = [0.0] * 256
    for half in range(2):
        for index in range(32):
            subgroup = index // 16
            low_base, high_base, scale_base = 64 * half, 32 * half, 8 * half
            values = (
                ((low[low_base + index] & 0x0F) | ((high[high_base + index] & 3) << 4)) - 32,
                ((low[low_base + index + 32] & 0x0F) | (((high[high_base + index] >> 2) & 3) << 4)) - 32,
                ((low[low_base + index] >> 4) | (((high[high_base + index] >> 4) & 3) << 4)) - 32,
                ((low[low_base + index + 32] >> 4) | (((high[high_base + index] >> 6) & 3) << 4)) - 32,
            )
            target = 128 * half + index
            for group, value in enumerate(values):
                factor = _f32(scale * scales[scale_base + subgroup + 2 * group])
                output[target + 32 * group] = _f32(factor * value)
    return output


_PURE_DECODERS = {
    "Q4_0": _pure_q4_0,
    "Q8_0": _pure_q8_0,
    "Q4_K": _pure_q4_k,
    "Q5_K": _pure_q5_k,
    "Q6_K": _pure_q6_k,
}


def _blocks(type_name, count=5):
    _, block_bytes = GGUF_BLOCK_SHAPES[type_name]
    generator = torch.Generator().manual_seed(block_bytes)
    blocks = torch.randint(0, 256, (count, block_bytes), dtype=torch.uint8, generator=generator)
    scale_offsets = {"Q4_0": (0,), "Q8_0": (0,), "Q4_K": (0, 2), "Q5_K": (0, 2), "Q6_K": (208,)}
    scales = (0.0, 5.9604644775390625e-08, 0.5, -1.5, 65504.0)
    for row in range(count):
        for offset_index, offset in enumerate(scale_offsets[type_name]):
            encoded = torch.tensor([scales[(row + offset_index) % len(scales)]], dtype=torch.float16).view(torch.uint8)
            blocks[row, offset : offset + 2] = encoded
    return blocks


@pytest.mark.parametrize("type_name", tuple(GGUF_BLOCK_SHAPES))
def test_vector_decoder_is_bit_identical_to_pure_reference(type_name):
    blocks = _blocks(type_name)
    block_elements, _ = GGUF_BLOCK_SHAPES[type_name]
    expected = torch.tensor(
        [value for block in blocks.tolist() for value in _PURE_DECODERS[type_name](bytes(block))],
        dtype=torch.float32,
    )
    actual = GGUF_BLOCK_DECODERS[type_name](blocks, (blocks.shape[0], block_elements))
    assert torch.equal(actual.reshape(-1).view(torch.int32), expected.view(torch.int32))
    encoded = GGUFWeightTensor(
        blocks,
        ggml_type=SimpleNamespace(name=type_name),
        tensor_shape=(blocks.shape[0], block_elements),
    )
    assert torch.equal(decode_gguf_tensor(encoded, dtype=torch.float16), actual.to(torch.float16))


def test_encoded_linear_matches_eager_and_reports_physical_bytes():
    blocks = _blocks("Q8_0", count=2)
    tensor = GGUFWeightTensor(
        blocks,
        ggml_type=SimpleNamespace(name="Q8_0"),
        tensor_shape=(2, 32),
    )
    module = GGUFOps.Linear(32, 2, bias=False, dtype=torch.float32)
    module.load_state_dict({"weight": tensor})
    input = torch.arange(64, dtype=torch.float32).reshape(2, 32)
    expected = torch.nn.functional.linear(input, decode_gguf_tensor(tensor))

    assert torch.equal(module(input), expected)
    assert torch.equal(
        decode_gguf_tensor(tensor, dtype=torch.float16),
        decode_gguf_tensor(tensor).to(torch.float16),
    )
    assert model_management.module_size(module) == blocks.nbytes
    assert utils.calculate_parameters({"weight": tensor}) == 64


def test_encoded_embedding_decodes_only_referenced_rows():
    blocks = _blocks("Q8_0", count=12)
    tensor = GGUFWeightTensor(
        blocks,
        ggml_type=SimpleNamespace(name="Q8_0"),
        tensor_shape=(6, 64),
    )
    module = GGUFOps.Embedding(6, 64, dtype=torch.float32)
    module.load_state_dict({"weight": tensor})
    input = torch.tensor([[5, 1, 5], [3, 1, 3]])
    expected = torch.nn.functional.embedding(input, decode_gguf_tensor(tensor))

    with mock.patch(
        "dinkster_inference.gguf_ops.decode_gguf_rows", wraps=decode_gguf_rows
    ) as decode_rows:
        actual = module(input, out_dtype=torch.float32)

    assert torch.equal(actual, expected)
    assert decode_rows.call_count == 1
    assert torch.equal(decode_rows.call_args.args[1], torch.tensor([1, 3, 5]))
    assert torch.equal(
        decode_gguf_rows(tensor, torch.tensor([1, 3, 5]), dtype=torch.float16),
        decode_gguf_rows(tensor, torch.tensor([1, 3, 5])).to(torch.float16),
    )


def test_memory_decodes_each_forward_while_predecoded_weight_does_not():
    blocks = _blocks("Q8_0", count=2)
    encoded = GGUFWeightTensor(
        blocks,
        ggml_type=SimpleNamespace(name="Q8_0"),
        tensor_shape=(2, 32),
    )
    decoded = decode_gguf_tensor(encoded)
    memory = GGUFOps.Linear(32, 2, bias=False, dtype=torch.float32)
    balanced = GGUFOps.Linear(32, 2, bias=False, dtype=torch.float32)
    memory.load_state_dict({"weight": encoded})
    balanced.load_state_dict({"weight": decoded})
    input = torch.arange(64, dtype=torch.float32).reshape(2, 32)

    with mock.patch(
        "dinkster_inference.gguf_ops.decode_gguf_tensor", wraps=decode_gguf_tensor
    ) as decode:
        memory_outputs = (memory(input), memory(input))
        assert decode.call_count == 2
        balanced_outputs = (balanced(input), balanced(input))
        assert decode.call_count == 2

    assert torch.equal(memory_outputs[0], memory_outputs[1])
    assert torch.equal(memory_outputs[0], balanced_outputs[0])
    assert torch.equal(balanced_outputs[0], balanced_outputs[1])


def test_encoded_and_predecoded_weights_use_model_patcher_accounting():
    blocks = _blocks("Q8_0", count=2)
    encoded = GGUFWeightTensor(
        blocks,
        ggml_type=SimpleNamespace(name="Q8_0"),
        tensor_shape=(2, 32),
    )

    class MixedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoded = GGUFOps.Linear(32, 2, bias=False, dtype=torch.float32)
            self.decoded = GGUFOps.Linear(32, 2, bias=False, dtype=torch.float32)
            self.encoded.load_state_dict({"weight": encoded})
            self.decoded.load_state_dict({"weight": decode_gguf_tensor(encoded)})

    model = MixedModel()
    patcher = model_patcher.ModelPatcher(
        model, load_device=torch.device("cpu"), offload_device=torch.device("cpu")
    )

    assert patcher.model_size() == blocks.nbytes + 2 * 32 * 4
    patcher.patch_model(torch.device("cpu"), lowvram_model_memory=1024)
    assert patcher.loaded_size() == patcher.model_size()

    assert patcher.partially_unload(torch.device("cpu"), 1) == patcher.model_size()
    assert patcher.loaded_size() == 0
    assert patcher.partially_load(torch.device("cpu"), 1024) == patcher.model_size()
    assert patcher.loaded_size() == patcher.model_size()


def _write_q8(path, architecture="sdxl"):
    writer = gguf.GGUFWriter(path, architecture)
    data = _blocks("Q8_0", count=1).numpy()
    writer.add_tensor(
        "model.diffusion_model.test.weight",
        data,
        raw_dtype=gguf.GGMLQuantizationType.Q8_0,
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_residency_modes_share_one_loader_and_obey_decoded_budget(tmp_path):
    path = tmp_path / "tiny.gguf"
    _write_q8(path)

    memory, memory_info = load_gguf_state_dict(path, residency_mode="memory")
    refused, refused_info = load_gguf_state_dict(
        path, residency_mode="balanced", decoded_budget_bytes=63, decoded_dtype=torch.float16
    )
    balanced, balanced_info = load_gguf_state_dict(
        path, residency_mode="balanced", decoded_budget_bytes=64, decoded_dtype=torch.float16
    )
    eager, eager_info = load_gguf_state_dict(
        path, residency_mode="eager", decoded_dtype=torch.float16
    )

    assert isinstance(memory["test.weight"], GGUFWeightTensor)
    assert isinstance(refused["test.weight"], GGUFWeightTensor)
    assert torch.equal(balanced["test.weight"], eager["test.weight"])
    assert balanced["test.weight"].dtype == eager["test.weight"].dtype == torch.float16
    assert memory_info.decoded_weight_bytes == refused_info.decoded_weight_bytes == 0
    assert balanced_info.decoded_weight_bytes == eager_info.decoded_weight_bytes == 64
    assert balanced_info.decoded_budget_bytes == 64
    assert memory_info.encoded_weight_bytes == 34


def test_loader_rejects_unknown_mode_before_reading():
    with pytest.raises(GGUFError, match="unknown GGUF residency mode"):
        load_gguf_state_dict("absent.gguf", residency_mode="fast")


def test_loader_admits_flux_diffusion_architecture(tmp_path):
    path = tmp_path / "flux.gguf"
    _write_q8(path, architecture="flux")

    state_dict, info = load_gguf_state_dict(path)

    assert info.architecture == "flux"
    assert isinstance(state_dict["test.weight"], GGUFWeightTensor)


def test_diffusion_loader_dispatches_gguf_without_changing_safetensors_route():
    info = GGUFLoadInfo("sdxl", "balanced", 128, 128, 34, {})
    gguf_model = SimpleNamespace(attachments={})
    safe_model = SimpleNamespace(attachments={})

    with (
        mock.patch.object(
            sd.dinkster_inference.gguf,
            "load_gguf_state_dict",
            return_value=({"gguf": torch.tensor(1)}, info),
        ) as load_gguf,
        mock.patch.object(
            sd.dinkster_inference.utils,
            "load_torch_file",
            return_value=({"safe": torch.tensor(1)}, {"source": "safe"}),
        ) as load_safe,
        mock.patch.object(
            sd,
            "load_diffusion_model_state_dict",
            side_effect=(gguf_model, safe_model),
        ) as load_state,
    ):
        assert sd.load_diffusion_model(
            "model.gguf",
            model_options={"gguf_residency": "balanced", "gguf_decoded_budget_bytes": 128},
        ) is gguf_model
        assert sd.load_diffusion_model("model.safetensors") is safe_model

    load_gguf.assert_called_once_with(
        "model.gguf",
        residency_mode="memory",
        load_device=None,
    )
    load_safe.assert_called_once_with("model.safetensors", return_metadata=True)
    assert isinstance(load_state.call_args_list[0].kwargs["model_options"]["custom_operations"], GGUFOps)
    assert "custom_operations" not in load_state.call_args_list[1].kwargs["model_options"]
    assert gguf_model.attachments["gguf"] is info


def test_clip_loader_dispatches_gguf_without_changing_safetensors_route():
    info = GGUFLoadInfo("t5encoder", "memory", 0, 0, 34, {})
    gguf_clip = SimpleNamespace(patcher=SimpleNamespace(attachments={}))
    safe_clip = SimpleNamespace(patcher=SimpleNamespace(attachments={}))

    with (
        mock.patch.object(
            sd.dinkster_inference.gguf,
            "load_gguf_state_dict",
            return_value=({"gguf": torch.tensor(1)}, info),
        ) as load_gguf,
        mock.patch.object(
            sd.dinkster_inference.utils,
            "load_torch_file",
            return_value=({"safe": torch.tensor(1)}, {"source": "safe"}),
        ) as load_safe,
        mock.patch.object(
            sd,
            "load_text_encoder_state_dicts",
            side_effect=(gguf_clip, safe_clip),
        ) as load_state,
        mock.patch.object(
            sd.model_management, "text_encoder_device", return_value=torch.device("cpu")
        ),
        mock.patch.object(
            sd.model_management, "text_encoder_dtype", return_value=torch.bfloat16
        ),
    ):
        assert sd.load_clip(["encoder.gguf"]) is gguf_clip
        assert sd.load_clip(["encoder.safetensors"]) is safe_clip

    load_gguf.assert_called_once_with(
        "encoder.gguf",
        text_model=True,
        residency_mode="memory",
        decoded_budget_bytes=None,
        load_device=None,
        decoded_dtype=torch.bfloat16,
    )
    load_safe.assert_called_once_with(
        "encoder.safetensors", safe_load=True, return_metadata=True
    )
    assert load_state.call_args_list[0].kwargs["model_options"]["custom_operations"] is GGUFOps
    assert "custom_operations" not in load_state.call_args_list[1].kwargs["model_options"]
    assert gguf_clip.patcher.attachments["gguf"] == (info,)
