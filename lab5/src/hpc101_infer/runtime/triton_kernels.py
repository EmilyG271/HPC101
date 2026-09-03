"""Triton kernels used by the INT4 inference fast path.

The kernel deliberately matches this checkpoint format: uint8 little-nibble
weights, row-major [out_features, padded_in_features // 2], and per-output-row
per-group scales/zero points. Importing this module is safe on CPU-only hosts;
Triton is loaded lazily by the Linear module.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only without Triton
    triton = None
    tl = None


def _positive_env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


_SPLIT_K_OVERRIDE = _positive_env_int("HPC101_INT4_SPLIT_K", 16)
_BLOCK_N_OVERRIDE = _positive_env_int("HPC101_INT4_BLOCK_N", 128)
_BLOCK_K_OVERRIDE = _positive_env_int("HPC101_INT4_BLOCK_K", 16)
_NUM_WARPS_OVERRIDE = _positive_env_int("HPC101_INT4_NUM_WARPS", 2)
_NUM_STAGES_OVERRIDE = _positive_env_int("HPC101_INT4_NUM_STAGES", 3)
_GEMV_BLOCK_N_OVERRIDE = _positive_env_int("HPC101_INT4_BLOCK_N", 4)
_GEMV_BLOCK_K_OVERRIDE = _positive_env_int("HPC101_INT4_GEMV_BLOCK_K", 1024)
_GEMV_NUM_WARPS_OVERRIDE = _positive_env_int("HPC101_INT4_GEMV_NUM_WARPS", 4)
_DEQUANT_BLOCK_N_OVERRIDE = _positive_env_int(
    "HPC101_DEQUANT_BLOCK_N", 128
)
_DEQUANT_BLOCK_K_OVERRIDE = _positive_env_int(
    "HPC101_DEQUANT_BLOCK_K", 64
)
_DEQUANT_NUM_WARPS_OVERRIDE = _positive_env_int(
    "HPC101_DEQUANT_NUM_WARPS", 8
)


if triton is not None:

    @triton.jit
    def _int4_row_gemv_kernel(
        x_ptr, q_ptr, s_ptr, z_ptr, bias_ptr, out_ptr,
        m, n, k, in_features, group_size,
        stride_xm, stride_xk, stride_qn, stride_qk,
        stride_sn, stride_sg, stride_om, stride_on,
        has_zero: tl.constexpr, has_bias: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        grid_n = tl.cdiv(n, BLOCK_N)
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < m
        n_mask = offs_n < n
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, k, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < in_features
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            byte_offsets = offs_k // 2
            q = tl.load(
                q_ptr + offs_n[:, None] * stride_qn
                + byte_offsets[None, :] * stride_qk,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0,
            )
            low = (offs_k[None, :] & 1) == 0
            code = tl.where(low, q & 0xF, q >> 4).to(tl.float32)
            group = offs_k // group_size
            scale = tl.load(
                s_ptr + offs_n[:, None] * stride_sn + group[None, :] * stride_sg,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            if has_zero:
                zero = tl.load(
                    z_ptr + offs_n[:, None] * stride_sn
                    + group[None, :] * stride_sg,
                    mask=n_mask[:, None] & k_mask[None, :],
                    other=0,
                ).to(tl.float32)
                weight = (code - zero) * scale
            else:
                weight = (code - 8.0) * scale
            products = x[:, None, :].to(tl.float32) * weight[None, :, :]
            acc += tl.sum(products, axis=2)

        if has_bias:
            bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            acc += bias[None, :]
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )

    @triton.jit
    def _int4_dequant_kernel(
        q_ptr, s_ptr, z_ptr, out_ptr,
        n, in_features, group_size, out_stride,
        stride_qn, stride_qk, stride_sn, stride_sg,
        has_zero: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        n_mask = offs_n < n
        k_mask = offs_k < in_features
        mask = n_mask[:, None] & k_mask[None, :]

        byte_offsets = offs_k // 2
        q = tl.load(
            q_ptr + offs_n[:, None] * stride_qn
            + byte_offsets[None, :] * stride_qk,
            mask=mask,
            other=0,
        )
        low = (offs_k[None, :] & 1) == 0
        code = tl.where(low, q & 0xF, q >> 4).to(tl.float32)
        group = offs_k // group_size
        scale = tl.load(
            s_ptr + offs_n[:, None] * stride_sn + group[None, :] * stride_sg,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if has_zero:
            zero = tl.load(
                z_ptr + offs_n[:, None] * stride_sn + group[None, :] * stride_sg,
                mask=mask,
                other=0,
            ).to(tl.float32)
            weight = (code - zero) * scale
        else:
            weight = (code - 8.0) * scale
        tl.store(
            out_ptr + offs_n[:, None] * out_stride + offs_k[None, :],
            weight,
            mask=mask,
        )

    @triton.jit
    def _int4_gemm_kernel(
        x_ptr, q_ptr, s_ptr, z_ptr, bias_ptr, out_ptr,
        m, n, k, in_features, group_size,
        stride_xm, stride_xk, stride_qn, stride_qk,
        stride_sn, stride_sg, stride_om, stride_on,
        has_zero: tl.constexpr, has_bias: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        grid_n = tl.cdiv(n, BLOCK_N)
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < m
        n_mask = offs_n < n
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, k, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < k
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=m_mask[:, None] & (offs_k[None, :] < in_features),
                other=0.0,
            )
            byte_offsets = offs_k // 2
            q = tl.load(
                q_ptr + offs_n[None, :] * stride_qn + byte_offsets[:, None] * stride_qk,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0,
            )
            low = (offs_k[:, None] & 1) == 0
            code = tl.where(low, q & 0xF, q >> 4).to(tl.float32)
            group = offs_k // group_size
            scale = tl.load(
                s_ptr + offs_n[None, :] * stride_sn + group[:, None] * stride_sg,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            if has_zero:
                zero = tl.load(
                    z_ptr + offs_n[None, :] * stride_sn + group[:, None] * stride_sg,
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0,
                ).to(tl.float32)
                weight = (code - zero) * scale
            else:
                weight = (code - 8.0) * scale
            # Triton's dot path accumulates in FP32 while retaining the model's
            # FP16/BF16 activation dtype for the tensor-core operand.
            acc += tl.dot(x, weight.to(x.dtype))

        if has_bias:
            bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            acc += bias[None, :]
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )


    @triton.jit
    def _int4_gate_up_kernel(
        x_ptr, gate_q_ptr, gate_s_ptr, gate_z_ptr,
        up_q_ptr, up_s_ptr, up_z_ptr, out_ptr,
        m, n, k, in_features, group_size,
        stride_xm, stride_xk,
        gate_stride_qn, gate_stride_qk, gate_stride_sn, gate_stride_sg,
        up_stride_qn, up_stride_qk, up_stride_sn, up_stride_sg,
        stride_om, stride_on,
        has_zero: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        grid_n = tl.cdiv(n, BLOCK_N)
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < m
        n_mask = offs_n < n
        gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, k, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < k
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=m_mask[:, None] & (offs_k[None, :] < in_features),
                other=0.0,
            )
            byte_offsets = offs_k // 2
            gate_q = tl.load(
                gate_q_ptr + offs_n[None, :] * gate_stride_qn
                + byte_offsets[:, None] * gate_stride_qk,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0,
            )
            up_q = tl.load(
                up_q_ptr + offs_n[None, :] * up_stride_qn
                + byte_offsets[:, None] * up_stride_qk,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0,
            )
            low = (offs_k[:, None] & 1) == 0
            gate_code = tl.where(low, gate_q & 0xF, gate_q >> 4).to(tl.float32)
            up_code = tl.where(low, up_q & 0xF, up_q >> 4).to(tl.float32)
            group = offs_k // group_size
            gate_scale = tl.load(
                gate_s_ptr + offs_n[None, :] * gate_stride_sn
                + group[:, None] * gate_stride_sg,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            up_scale = tl.load(
                up_s_ptr + offs_n[None, :] * up_stride_sn
                + group[:, None] * up_stride_sg,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            if has_zero:
                gate_zero = tl.load(
                    gate_z_ptr + offs_n[None, :] * gate_stride_sn
                    + group[:, None] * gate_stride_sg,
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0,
                ).to(tl.float32)
                up_zero = tl.load(
                    up_z_ptr + offs_n[None, :] * up_stride_sn
                    + group[:, None] * up_stride_sg,
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0,
                ).to(tl.float32)
                gate_weight = (gate_code - gate_zero) * gate_scale
                up_weight = (up_code - up_zero) * up_scale
            else:
                gate_weight = (gate_code - 8.0) * gate_scale
                up_weight = (up_code - 8.0) * up_scale
            gate_acc += tl.dot(x, gate_weight.to(x.dtype))
            up_acc += tl.dot(x, up_weight.to(x.dtype))

        # Match torch.nn.functional.gelu(..., approximate="tanh").
        cube = gate_acc * gate_acc * gate_acc
        inner = 0.7978845608028654 * (gate_acc + 0.044715 * cube)
        activated = gate_acc * tl.sigmoid(2.0 * inner)
        output = activated * up_acc
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            output,
            mask=m_mask[:, None] & n_mask[None, :],
        )


def int4_linear(
    inputs: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    bias: torch.Tensor | None,
    in_features: int,
    out_features: int,
    group_size: int,
    padded_in_features: int,
    symmetric: bool,
) -> torch.Tensor:
    """Fused INT4 matmul for arbitrary leading dimensions of ``inputs``."""
    if triton is None or tl is None:
        raise RuntimeError("Triton is not available")
    if inputs.ndim < 2 or inputs.shape[-1] != in_features:
        raise ValueError("inputs has incompatible shape")
    if qweight.dtype != torch.uint8 or scales.ndim != 2:
        raise ValueError("invalid quantized weight tensors")

    original_shape = inputs.shape[:-1]
    x = inputs.reshape(-1, in_features).contiguous()
    has_zero = not symmetric
    z_ptr = zeros if zeros is not None else qweight
    if (
        os.environ.get("HPC101_INT4_GEMV", "1") != "0"
        and x.shape[0] <= 4
    ):
        output = torch.empty(
            (x.shape[0], out_features), device=inputs.device, dtype=inputs.dtype
        )
        block_n = _GEMV_BLOCK_N_OVERRIDE
        grid = (
            triton.cdiv(x.shape[0], 4)
            * triton.cdiv(out_features, block_n),
        )
        _int4_row_gemv_kernel[grid](
            x, qweight, scales, z_ptr, bias, output,
            x.shape[0], out_features, padded_in_features, in_features,
            group_size,
            x.stride(0), x.stride(1),
            qweight.stride(0), qweight.stride(1),
            scales.stride(0), scales.stride(1),
            output.stride(0), output.stride(1),
            has_zero=has_zero, has_bias=bias is not None,
            BLOCK_M=4, BLOCK_N=block_n, BLOCK_K=_GEMV_BLOCK_K_OVERRIDE,
            num_warps=_GEMV_NUM_WARPS_OVERRIDE,
        )
        return output.reshape(*original_shape, out_features)

    split_k = _SPLIT_K_OVERRIDE
    if (
        split_k > 1
        and x.shape[0] <= 4
        and out_features <= 4096
        and padded_in_features >= 4096
    ):
        partials = torch.empty(
            (split_k, x.shape[0], out_features),
            device=inputs.device,
            dtype=torch.float32,
        )
        block_m = 4
        block_n = _BLOCK_N_OVERRIDE
        grid_n = triton.cdiv(out_features, block_n)
        grid = (triton.cdiv(x.shape[0], block_m) * grid_n * split_k,)
        k_chunk = (padded_in_features + split_k - 1) // split_k
        _int4_gemm_split_k_kernel[grid](
            x, qweight, scales, z_ptr, partials,
            x.shape[0], out_features, padded_in_features, in_features,
            group_size, k_chunk,
            x.stride(0), x.stride(1),
            qweight.stride(0), qweight.stride(1),
            scales.stride(0), scales.stride(1),
            partials.stride(0), partials.stride(1), partials.stride(2),
            has_zero=has_zero,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=_BLOCK_K_OVERRIDE,
            num_warps=_NUM_WARPS_OVERRIDE, num_stages=_NUM_STAGES_OVERRIDE,
        )
        output = partials.sum(dim=0, dtype=inputs.dtype)
        if bias is not None:
            output += bias
        return output.reshape(*original_shape, out_features)

    # The padded columns are zero by construction, so a K tile only needs the
    # original activation width; the corresponding last quantization group is
    # still addressed correctly because groups are based on absolute columns.
    output = torch.empty(
        (x.shape[0], out_features), device=inputs.device, dtype=inputs.dtype
    )
    # The 32-row tile keeps the tensor-core dot path efficient on the H800.
    # It also avoids compiling many small-M variants across the variable batch
    # shapes produced by the queue scheduler.
    # Decode batches of 2-3 waste fewer lanes with a small tile while retaining
    # the same tensor-core dot path. Larger prefill shapes use 32 rows.
    block_m = 4 if x.shape[0] <= 4 else 32
    block_n = _BLOCK_N_OVERRIDE
    grid = lambda meta: (
        triton.cdiv(x.shape[0], meta["BLOCK_M"])
        * triton.cdiv(out_features, meta["BLOCK_N"]),
    )
    _int4_gemm_kernel[grid](
        x, qweight, scales, z_ptr, bias, output,
        x.shape[0], out_features, padded_in_features, in_features, group_size,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        scales.stride(0), scales.stride(1),
        output.stride(0), output.stride(1),
        has_zero=has_zero, has_bias=bias is not None,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=_BLOCK_K_OVERRIDE,
        num_warps=_NUM_WARPS_OVERRIDE, num_stages=_NUM_STAGES_OVERRIDE,
    )
    return output.reshape(*original_shape, out_features)


def dequantize_int4_weight(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    out_features: int,
    in_features: int,
    group_size: int,
    padded_in_features: int,
    symmetric: bool,
    output: torch.Tensor,
) -> torch.Tensor:
    """Dequantize one Linear weight into a caller-owned dense buffer."""
    if triton is None or tl is None:
        raise RuntimeError("Triton is not available")
    if qweight.dtype != torch.uint8 or scales.ndim != 2:
        raise ValueError("invalid quantized weight tensors")
    expected = (out_features, in_features)
    if tuple(output.shape) != expected:
        raise ValueError("output buffer has incompatible shape")
    if output.device != qweight.device or output.dtype not in (
        torch.float16,
        torch.bfloat16,
    ):
        raise ValueError("output buffer must be a CUDA FP16/BF16 tensor")
    if symmetric and zeros is not None:
        raise ValueError("symmetric quantization must not provide zeros")
    if not symmetric and zeros is None:
        raise ValueError("asymmetric quantization requires zeros")

    has_zero = not symmetric
    z_ptr = zeros if zeros is not None else qweight
    block_n = _DEQUANT_BLOCK_N_OVERRIDE
    block_k = _DEQUANT_BLOCK_K_OVERRIDE
    grid = (
        triton.cdiv(out_features, block_n),
        triton.cdiv(in_features, block_k),
    )
    _int4_dequant_kernel[grid](
        qweight,
        scales,
        z_ptr,
        output,
        out_features,
        in_features,
        group_size,
        output.stride(0),
        qweight.stride(0),
        qweight.stride(1),
        scales.stride(0),
        scales.stride(1),
        has_zero=has_zero,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=_DEQUANT_NUM_WARPS_OVERRIDE,
    )
    return output


def fused_gate_up_int4(
    inputs: torch.Tensor,
    gate_qweight: torch.Tensor,
    gate_scales: torch.Tensor,
    gate_zeros: torch.Tensor | None,
    up_qweight: torch.Tensor,
    up_scales: torch.Tensor,
    up_zeros: torch.Tensor | None,
    gate_in_features: int,
    gate_out_features: int,
    gate_group_size: int,
    gate_padded_in_features: int,
    gate_symmetric: bool,
    up_in_features: int,
    up_out_features: int,
    up_group_size: int,
    up_padded_in_features: int,
    up_symmetric: bool,
) -> torch.Tensor:
    """Compute ``gelu(gate(x), tanh=True) * up(x)`` in one Triton launch."""
    if triton is None or tl is None:
        raise RuntimeError("Triton is not available")
    if (
        gate_in_features != up_in_features
        or gate_out_features != up_out_features
        or gate_group_size != up_group_size
        or gate_padded_in_features != up_padded_in_features
        or gate_symmetric != up_symmetric
    ):
        raise ValueError("gate and up projections have incompatible shapes")
    if inputs.ndim < 2 or inputs.shape[-1] != gate_in_features:
        raise ValueError("inputs has incompatible shape")
    if gate_qweight.dtype != torch.uint8 or up_qweight.dtype != torch.uint8:
        raise ValueError("invalid quantized weight tensors")

    original_shape = inputs.shape[:-1]
    x = inputs.reshape(-1, gate_in_features).contiguous()
    output = torch.empty(
        (x.shape[0], gate_out_features),
        device=inputs.device,
        dtype=inputs.dtype,
    )
    block_m = 4 if x.shape[0] <= 4 else 32
    grid = lambda meta: (
        triton.cdiv(x.shape[0], meta["BLOCK_M"])
        * triton.cdiv(gate_out_features, meta["BLOCK_N"]),
    )
    gate_z_ptr = gate_zeros if gate_zeros is not None else gate_qweight
    up_z_ptr = up_zeros if up_zeros is not None else up_qweight
    _int4_gate_up_kernel[grid](
        x, gate_qweight, gate_scales, gate_z_ptr,
        up_qweight, up_scales, up_z_ptr, output,
        x.shape[0], gate_out_features, gate_padded_in_features,
        gate_in_features, gate_group_size,
        x.stride(0), x.stride(1),
        gate_qweight.stride(0), gate_qweight.stride(1),
        gate_scales.stride(0), gate_scales.stride(1),
        up_qweight.stride(0), up_qweight.stride(1),
        up_scales.stride(0), up_scales.stride(1),
        output.stride(0), output.stride(1),
        has_zero=not gate_symmetric,
        BLOCK_M=block_m, BLOCK_N=_BLOCK_N_OVERRIDE,
        BLOCK_K=_BLOCK_K_OVERRIDE,
        num_warps=_NUM_WARPS_OVERRIDE,
        num_stages=_NUM_STAGES_OVERRIDE,
    )
    return output.reshape(*original_shape, gate_out_features)


if triton is not None:

    @triton.jit
    def _int4_gemm_split_k_kernel(
        x_ptr, q_ptr, s_ptr, z_ptr, partial_ptr,
        m, n, k, in_features, group_size, k_chunk,
        stride_xm, stride_xk, stride_qn, stride_qk,
        stride_sn, stride_sg, stride_ps, stride_pm, stride_pn,
        has_zero: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        grid_n = tl.cdiv(n, BLOCK_N)
        grid_m = tl.cdiv(m, BLOCK_M)
        num_tiles = grid_m * grid_n
        split_id = pid // num_tiles
        tile_id = pid % num_tiles
        pid_m = tile_id // grid_n
        pid_n = tile_id % grid_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < m
        n_mask = offs_n < n
        k_start = split_id * k_chunk
        k_end = tl.minimum(k_start + k_chunk, k)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(k_start, k_end, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < k_end
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=m_mask[:, None] & (offs_k[None, :] < in_features),
                other=0.0,
            )
            byte_offsets = offs_k // 2
            q = tl.load(
                q_ptr + offs_n[None, :] * stride_qn + byte_offsets[:, None] * stride_qk,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0,
            )
            low = (offs_k[:, None] & 1) == 0
            code = tl.where(low, q & 0xF, q >> 4).to(tl.float32)
            group = offs_k // group_size
            scale = tl.load(
                s_ptr + offs_n[None, :] * stride_sn + group[:, None] * stride_sg,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            if has_zero:
                zero = tl.load(
                    z_ptr + offs_n[None, :] * stride_sn + group[:, None] * stride_sg,
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0,
                ).to(tl.float32)
                weight = (code - zero) * scale
            else:
                weight = (code - 8.0) * scale
            acc += tl.dot(x, weight.to(x.dtype))

        tl.store(
            partial_ptr + split_id * stride_ps + offs_m[:, None] * stride_pm
            + offs_n[None, :] * stride_pn,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )
