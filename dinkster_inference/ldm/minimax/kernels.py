# Adapted from SGLang's indexed_modulation_triton.py under Apache-2.0.

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _round_bf16_to_fp32(value):
        bits = value.to(tl.int32, bitcast=True)
        rounding_bias = 0x7FFF + ((bits >> 16) & 1)
        rounded_bits = (bits + rounding_bias) & -65536
        return rounded_bits.to(tl.float32, bitcast=True)

    @triton.jit
    def _indexed_scale_shift_bf16_kernel(
        x_ptr,
        shift_ptr,
        scale_ptr,
        indices_ptr,
        hidden_size,
        stride_x_row,
        stride_shift_row,
        stride_scale_row,
        stride_indices,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK_N)
        mask = columns < hidden_size
        index = tl.load(indices_ptr + row * stride_indices)
        x = tl.load(x_ptr + row * stride_x_row + columns, mask=mask, other=0.0).to(
            tl.float32
        )
        shift = tl.load(
            shift_ptr + index * stride_shift_row + columns, mask=mask, other=0.0
        ).to(tl.float32)
        scale = tl.load(
            scale_ptr + index * stride_scale_row + columns, mask=mask, other=0.0
        ).to(tl.float32)
        one_plus_scale = _round_bf16_to_fp32(1.0 + scale)
        scaled = _round_bf16_to_fp32(x * one_plus_scale)
        tl.store(x_ptr + row * stride_x_row + columns, scaled + shift, mask=mask)

    @triton.jit
    def _indexed_gate_bf16_kernel(
        x_ptr,
        gate_ptr,
        other_ptr,
        indices_ptr,
        hidden_size,
        stride_x_row,
        stride_gate_row,
        stride_other_row,
        stride_indices,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK_N)
        mask = columns < hidden_size
        index = tl.load(indices_ptr + row * stride_indices)
        x = tl.load(x_ptr + row * stride_x_row + columns, mask=mask, other=0.0).to(
            tl.float32
        )
        gate = tl.load(
            gate_ptr + index * stride_gate_row + columns, mask=mask, other=0.0
        ).to(tl.float32)
        other = tl.load(
            other_ptr + row * stride_other_row + columns, mask=mask, other=0.0
        ).to(tl.float32)
        output = tl.inline_asm_elementwise(
            asm="fma.rn.f32 $0, $1, $2, $3;",
            constraints="=f,f,f,f",
            args=[gate, other, x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        tl.store(x_ptr + row * stride_x_row + columns, output, mask=mask)


def _can_use_indexed_bf16(x, modulation, indices):
    return (
        triton is not None
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and modulation.dtype == torch.bfloat16
        and indices.is_cuda
        and indices.dtype in (torch.int32, torch.int64)
        and x.ndim == 2
        and modulation.ndim == 2
        and indices.ndim == 1
        and x.shape[0] == indices.shape[0]
        and x.shape[1] == modulation.shape[1]
        and x.is_contiguous()
        and indices.is_contiguous()
        and modulation.stride(1) == 1
    )


def try_indexed_scale_shift_bf16_(x, shift, scale, indices):
    if not (
        _can_use_indexed_bf16(x, shift, indices)
        and scale.dtype == torch.bfloat16
        and scale.shape == shift.shape
        and scale.stride(1) == 1
    ):
        return False
    rows, hidden_size = x.shape
    if rows:
        _indexed_scale_shift_bf16_kernel[(rows,)](
            x,
            shift,
            scale,
            indices,
            hidden_size,
            x.stride(0),
            shift.stride(0),
            scale.stride(0),
            indices.stride(0),
            BLOCK_N=triton.next_power_of_2(hidden_size),
            num_warps=8,
        )
    return True


def try_indexed_gate_bf16_(x, gate, other, indices):
    if not (
        _can_use_indexed_bf16(x, gate, indices)
        and other.dtype == torch.bfloat16
        and other.shape == x.shape
        and other.is_contiguous()
    ):
        return False
    rows, hidden_size = x.shape
    if rows:
        _indexed_gate_bf16_kernel[(rows,)](
            x,
            gate,
            other,
            indices,
            hidden_size,
            x.stride(0),
            gate.stride(0),
            other.stride(0),
            indices.stride(0),
            BLOCK_N=triton.next_power_of_2(hidden_size),
            num_warps=8,
        )
    return True
