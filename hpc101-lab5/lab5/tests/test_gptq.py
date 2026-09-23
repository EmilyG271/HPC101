from __future__ import annotations

import pytest
import torch

from hpc101_infer.layers.linear import QuantizedLinear
from hpc101_infer.quantization.methods.gptq import quantize_weight_gptq
from hpc101_infer.quantization.packing import dequantize_weight


def _quantize(
    weight: torch.Tensor,
    activations: torch.Tensor,
    *,
    block_size: int,
    symmetric: bool = False,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    quantized, metadata = quantize_weight_gptq(
        weight,
        activations,
        group_size=4,
        block_size=block_size,
        symmetric=symmetric,
    )
    return dequantize_weight(quantized), metadata


def test_gptq_padding_dead_columns_and_block_equivalence() -> None:
    torch.manual_seed(7)
    activations = torch.randn(32, 10)
    activations[:, 2] = 0.0
    weight = torch.randn(5, 10)

    small_block, small_metadata = _quantize(
        weight, activations, block_size=3
    )
    full_block, full_metadata = _quantize(
        weight, activations, block_size=10
    )

    assert small_block.shape == weight.shape
    assert small_metadata["dead_columns"] >= 1
    assert small_metadata["block_size"] == 3
    assert small_metadata["damp_retries"] == 0
    assert small_metadata["effective_damp_percent"] == pytest.approx(1.0)
    assert small_metadata["eigval_correction"] == 0.0
    assert full_metadata["dead_columns"] >= 1
    assert torch.allclose(small_block, full_block, atol=1e-6, rtol=1e-6)


def test_gptq_survives_rank_deficient_hessian() -> None:
    torch.manual_seed(11)
    activations = torch.randn(12, 8)
    activations[:, 1] = activations[:, 0]
    activations[:, 3] = activations[:, 2]
    weight = torch.randn(4, 8)

    decoded, metadata = _quantize(weight, activations, block_size=4)

    assert decoded.shape == weight.shape
    assert bool(torch.isfinite(decoded).all())
    assert metadata["damp_retries"] >= 0
    assert metadata["effective_damp_percent"] > 0.0
    assert metadata["eigval_correction"] >= 0.0


def test_quantized_linear_weight_update_and_device_migration() -> None:
    torch.manual_seed(13)
    activations = torch.randn(24, 8)
    weight_one = torch.randn(4, 8)
    weight_two = torch.randn(4, 8)
    inputs = torch.randn(3, 8)

    quantized_one, _ = quantize_weight_gptq(
        weight_one, activations, group_size=4, block_size=4, symmetric=False
    )
    quantized_two, _ = quantize_weight_gptq(
        weight_two, activations, group_size=4, block_size=4, symmetric=False
    )
    module = QuantizedLinear.from_quantized_weight(quantized_one)

    first_output = module(inputs)
    assert torch.allclose(
        first_output,
        torch.nn.functional.linear(
            inputs,
            dequantize_weight(quantized_one, dtype=torch.float32),
            module.bias,
        ),
        atol=1e-6,
    )

    module.qweight.copy_(quantized_two.qweight)
    module.scales.copy_(quantized_two.scales)
    assert module.zeros is not None and quantized_two.zeros is not None
    module.zeros.copy_(quantized_two.zeros)
    second_output = module(inputs)
    assert torch.allclose(
        second_output,
        torch.nn.functional.linear(
            inputs,
            dequantize_weight(quantized_two, dtype=torch.float32),
            module.bias,
        ),
        atol=1e-6,
    )
    assert not torch.allclose(first_output, second_output)

    if torch.cuda.is_available():
        cuda_module = module.to(device="cuda")
        cuda_output = cuda_module(inputs.to(device="cuda"))
        assert cuda_module.qweight.is_cuda
        assert torch.allclose(
            second_output,
            cuda_output.cpu(),
            atol=2e-2,
            rtol=2e-2,
        )
