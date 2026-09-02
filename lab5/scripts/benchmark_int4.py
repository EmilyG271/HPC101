from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch
from torch.nn import functional as F

from hpc101_infer.quantization.packing import dequantize_weight
from hpc101_infer.quantization.types import QuantizedWeight
from hpc101_infer.runtime import triton_kernels
from hpc101_infer.runtime.triton_kernels import fused_gate_up_int4, int4_linear


SHAPES = (
    ("q_o_proj", 4096, 4096),
    ("kv_proj", 1024, 4096),
    ("down_proj", 4096, 14336),
    ("gate_up_proj", 14336, 4096),
)


def make_weight(
    out_features: int,
    in_features: int,
    group_size: int,
    device: torch.device,
) -> tuple[QuantizedWeight, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(1234)
    quantized = QuantizedWeight(
        qweight=torch.randint(
            0,
            256,
            (out_features, in_features // 2),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        ),
        scales=(torch.rand((out_features, in_features // group_size), generator=generator, device=device) * 0.01).to(torch.float16),
        zeros=torch.randint(
            0,
            16,
            (out_features, in_features // group_size),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        ),
        original_shape=(out_features, in_features),
        padded_shape=(out_features, in_features),
        bits=4,
        group_size=group_size,
        symmetric=False,
        packing="uint8_little_nibble",
    )
    return quantized, dequantize_weight(quantized, dtype=torch.bfloat16)


def benchmark(
    name: str,
    rows: int,
    out_features: int,
    in_features: int,
    splits: Sequence[int],
    iterations: int,
) -> None:
    device = torch.device("cuda")
    inputs = torch.randn(
        (rows, in_features), device=device, dtype=torch.bfloat16
    )
    quantized, weight = make_weight(out_features, in_features, 128, device)
    reference = inputs @ weight.T

    for split_k in splits:
        triton_kernels._SPLIT_K_OVERRIDE = split_k
        output = int4_linear(
            inputs,
            quantized.qweight,
            quantized.scales,
            quantized.zeros,
            None,
            in_features,
            out_features,
            128,
            in_features,
            False,
        )
        max_error = (output.float() - reference.float()).abs().max().item()
        for _ in range(3):
            int4_linear(
                inputs,
                quantized.qweight,
                quantized.scales,
                quantized.zeros,
                None,
                in_features,
                out_features,
                128,
                in_features,
                False,
            )
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            int4_linear(
                inputs,
                quantized.qweight,
                quantized.scales,
                quantized.zeros,
                None,
                in_features,
                out_features,
                128,
                in_features,
                False,
            )
        end.record()
        torch.cuda.synchronize()
        milliseconds = start.elapsed_time(end) / iterations
        print(
            f"{name:14s} M={rows} N={out_features:5d} K={in_features:5d} "
            f"split_k={split_k:2d} time_ms={milliseconds:8.3f} "
            f"max_abs_error={max_error:.6f}",
            flush=True,
        )

    del inputs, quantized, weight, reference
    torch.cuda.empty_cache()


def run_gate_up(
    inputs: torch.Tensor,
    gate: QuantizedWeight,
    up: QuantizedWeight,
) -> torch.Tensor:
    gate = int4_linear(
        inputs,
        gate.qweight,
        gate.scales,
        gate.zeros,
        None,
        gate.original_shape[1],
        gate.original_shape[0],
        gate.group_size,
        gate.padded_shape[1],
        gate.symmetric,
    )
    up = int4_linear(
        inputs,
        up.qweight,
        up.scales,
        up.zeros,
        None,
        up.original_shape[1],
        up.original_shape[0],
        up.group_size,
        up.padded_shape[1],
        up.symmetric,
    )
    return F.gelu(gate, approximate="tanh") * up


def benchmark_gate_up(rows: int, out_features: int, in_features: int, iterations: int) -> None:
    device = torch.device("cuda")
    inputs = torch.randn((rows, in_features), device=device, dtype=torch.bfloat16)
    gate, gate_weight = make_weight(out_features, in_features, 128, device)
    up, up_weight = make_weight(out_features, in_features, 128, device)
    gate_logits = inputs @ gate_weight.T
    up_logits = inputs @ up_weight.T
    reference = F.gelu(gate_logits, approximate="tanh") * up_logits

    fused = fused_gate_up_int4(
        inputs,
        gate.qweight,
        gate.scales,
        gate.zeros,
        up.qweight,
        up.scales,
        up.zeros,
        in_features,
        out_features,
        128,
        in_features,
        False,
        in_features,
        out_features,
        128,
        in_features,
        False,
    )
    max_error = (fused.float() - reference.float()).abs().max().item()

    for name, function in (
        ("separate", lambda: run_gate_up(inputs, gate, up)),
        ("fused", lambda: fused_gate_up_int4(
            inputs,
            gate.qweight,
            gate.scales,
            gate.zeros,
            up.qweight,
            up.scales,
            up.zeros,
            in_features,
            out_features,
            128,
            in_features,
            False,
            in_features,
            out_features,
            128,
            in_features,
            False,
        )),
    ):
        for _ in range(3):
            function()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            function()
        end.record()
        torch.cuda.synchronize()
        milliseconds = start.elapsed_time(end) / iterations
        print(
            f"gate_up_{name:8s} M={rows} N={out_features:5d} K={in_features:5d} "
            f"time_ms={milliseconds:8.3f} max_abs_error={max_error:.6f}",
            flush=True,
        )

    del inputs, gate, up, gate_weight, up_weight, gate_logits, up_logits, reference
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--splits",
        type=int,
        nargs="+",
        default=(1, 2, 4, 8, 16),
    )
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=128)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--gemv-block-k", type=int, default=1024)
    parser.add_argument("--gemv-num-warps", type=int, default=4)
    parser.add_argument(
        "--shapes",
        nargs="+",
        choices=tuple(name for name, _, _ in SHAPES),
        default=tuple(name for name, _, _ in SHAPES),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    triton_kernels._BLOCK_N_OVERRIDE = args.block_n
    triton_kernels._BLOCK_K_OVERRIDE = args.block_k
    triton_kernels._NUM_WARPS_OVERRIDE = args.num_warps
    triton_kernels._NUM_STAGES_OVERRIDE = args.num_stages
    triton_kernels._GEMV_BLOCK_N_OVERRIDE = args.block_n
    triton_kernels._GEMV_BLOCK_K_OVERRIDE = args.gemv_block_k
    triton_kernels._GEMV_NUM_WARPS_OVERRIDE = args.gemv_num_warps
    for name, out_features, in_features in SHAPES:
        if name in args.shapes:
            benchmark(
                name,
                args.rows,
                out_features,
                in_features,
                args.splits,
                args.iterations,
            )
    benchmark_gate_up(args.rows, 14336, 4096, args.iterations)


if __name__ == "__main__":
    main()
