"""Triton kernels used by the INT4 inference fast path.

The kernel deliberately matches this checkpoint format: uint8 little-nibble
weights, row-major [out_features, padded_in_features // 2], and per-output-row
per-group scales/zero points. Importing this module is safe on CPU-only hosts;
Triton is loaded lazily by the Linear module.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only without Triton
    triton = None
    tl = None


if triton is not None:

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
    # The padded columns are zero by construction, so a K tile only needs the
    # original activation width; the corresponding last quantization group is
    # still addressed correctly because groups are based on absolute columns.
    output = torch.empty(
        (x.shape[0], out_features), device=inputs.device, dtype=inputs.dtype
    )
    has_zero = not symmetric
    z_ptr = zeros if zeros is not None else qweight
    # Decode is dominated by M=1/2. A 32-row tile wastes most of the dot
    # product for those shapes, so specialize the tile height to the actual
    # active batch. Larger M values use the tensor-core-friendly fallback tile.
    block_m = 1 if x.shape[0] == 1 else 2 if x.shape[0] <= 2 else 4 if x.shape[0] <= 4 else 32
    block_n = 128 if out_features >= 4096 else 256
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
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=128,
        num_warps=4, num_stages=3,
    )
    return output.reshape(*original_shape, out_features)
