from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from hpc101_infer.quantization.packing import pack_int4
from hpc101_infer.quantization.types import (
    LayerContext,
    LayerQuantizationResult,
    QuantizedWeight,
    SCALE_DTYPES,
)


@dataclass(frozen=True)
class GPTQOptions:
    block_size: int
    damp_percent: float


@dataclass(frozen=True)
class GPTQModuleState:
    activations: torch.Tensor
    activation_tokens: int


@dataclass(frozen=True)
class GPTQLayerState:
    modules: Mapping[str, GPTQModuleState]
    options: GPTQOptions


def _calibration_option(
    calibration: Mapping[str, Any],
    name: str,
    default: int | float,
) -> int | float:
    gptq = calibration.get("gptq", calibration)
    if not isinstance(gptq, Mapping):
        raise TypeError("config.calibration['gptq'] must be a mapping")
    return gptq.get(name, default)


def _parse_gptq_options(calibration: Mapping[str, Any]) -> GPTQOptions:
    block_size = _calibration_option(calibration, "block_size", 128)
    damp_percent = _calibration_option(calibration, "damp_percent", 0.01)

    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size <= 0
    ):
        raise ValueError("GPTQ block_size must be a positive integer")
    if (
        not isinstance(damp_percent, (int, float))
        or isinstance(damp_percent, bool)
        or not 0.0 < float(damp_percent) < 1.0
    ):
        raise ValueError("GPTQ damp_percent must be in (0, 1)")

    gptq = calibration.get("gptq", calibration)
    if isinstance(gptq, Mapping) and gptq.get("desc_act", False):
        raise ValueError("GPTQ desc_act is not supported by this checkpoint format")
    return GPTQOptions(
        block_size=block_size,
        damp_percent=float(damp_percent),
    )


def quantize_weight_gptq(
    weight: torch.Tensor,
    activations: torch.Tensor,
    group_size: int,
    *,
    block_size: int = 128,
    damp_percent: float = 0.01,
    symmetric: bool = True,
    scale_dtype: torch.dtype = torch.float16,
) -> tuple[QuantizedWeight, dict[str, float | int]]:
    """
    使用 GPTQ 算法将权重量化为 INT4。

    参数：
        weight: 待量化的权重张量，形状为 (out_features, in_features)。
        activations: 校准数据集的输入激活，形状为 (calibration_tokens, in_features)。
        group_size: 量化粒度，即每个 group 中的列数。
        block_size: 分块计算时每个 block 中的列数。
        damp_percent: 阻尼比例，用于改善 Hessian 的数值稳定性。
        symmetric: 是否使用对称量化。
        scale_dtype: 缩放因子的 dtype。

    返回：
        quantized_weight: 量化后的权重对象，具体详见 QuantizedWeight 类的定义。
        metadatas: 量化过程中的统计信息，不影响评测，用于分析和调试。
    """

    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("weight must be a floating-point matrix")
    if activations.ndim != 2 or not activations.is_floating_point():
        raise ValueError("activations must be a floating-point matrix")
    if activations.shape[1] != weight.shape[1]:
        raise ValueError("activation width does not match weight input size")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not 0.0 < damp_percent < 1.0:
        raise ValueError("damp_percent must be in (0, 1)")
    if scale_dtype not in SCALE_DTYPES.values():
        raise ValueError("unsupported scale dtype")
    if not torch.isfinite(weight).all():
        raise ValueError("weight must contain only finite values")

    # GPTQ minimizes the calibration-weighted output error rather than the
    # unweighted weight error. Work in FP32 for the Hessian and error
    # propagation; only the small quantization metadata is cast at the end.
    # The pipeline calls this routine one Linear at a time, so at most one
    # large Hessian is resident on the GPU.
    device = weight.device
    out_features, in_features = weight.shape
    padded_in_features = math.ceil(in_features / group_size) * group_size
    num_groups = padded_in_features // group_size

    work = weight.detach().to(device=device, dtype=torch.float32).clone()
    if padded_in_features != in_features:
        work = F.pad(work, (0, padded_in_features - in_features))

    # H = X^T X / N. A damped Cholesky inverse is more stable than explicitly
    # forming a pseudo-inverse for the nearly rank-deficient matrices produced
    # by short calibration sets.
    calibration = activations.detach().to(device=device, dtype=torch.float32)
    tokens = calibration.shape[0]
    hessian = calibration.transpose(0, 1).matmul(calibration) / float(max(tokens, 1))
    del calibration
    if padded_in_features != in_features:
        # Padding columns have no calibration activation. Extend H with
        # independent unit diagonal entries so its inverse matches the padded
        # weight layout used by the block propagation below.
        padded_hessian = torch.zeros(
            (padded_in_features, padded_in_features),
            device=device,
            dtype=torch.float32,
        )
        padded_hessian[:in_features, :in_features] = hessian
        padded_hessian.diagonal()[in_features:] = 1.0
        hessian = padded_hessian

    eps = torch.finfo(torch.float32).eps
    diagonal = torch.diagonal(hessian).clone()
    if not bool(torch.isfinite(diagonal).all()):
        raise ValueError("activation Hessian contains non-finite diagonal entries")
    dead = diagonal <= eps
    if padded_in_features != in_features:
        dead[in_features:] = True
    dead_columns = int(dead.sum().item())
    if dead_columns:
        # Make dead features independent so the factorization stays valid even
        # when the number of calibration tokens is smaller than in_features.
        hessian[dead, :] = 0.0
        hessian[:, dead] = 0.0
        hessian.diagonal()[dead] = 1.0
        diagonal = torch.diagonal(hessian).clone()

    mean_diagonal = diagonal[~dead].mean() if bool((~dead).any()) else diagonal.mean()
    if not bool(torch.isfinite(mean_diagonal)) or float(mean_diagonal) <= 0.0:
        mean_diagonal = torch.tensor(1.0, device=device, dtype=torch.float32)
    damping = float(damp_percent) * mean_diagonal
    hessian.diagonal().add_(damping)

    # Retry with increasing diagonal jitter for pathological calibration data.
    # Do not materialize an additional dense identity matrix: for the 14,336
    # input features of Gemma's MLP, that temporary allocation is large enough
    # to exhaust a 10-GiB MIG slice.
    hessian_inverse: torch.Tensor | None = None
    inverse_diagonal: torch.Tensor | None = None
    for attempt in range(6):
        if attempt:
            # Start above FP32 roundoff and grow to a conservative diagonal
            # regularizer for rank-deficient short calibration sets.
            extra_jitter = mean_diagonal * (10.0 ** (-3 + attempt))
            hessian.diagonal().add_(extra_jitter)
        chol, info = torch.linalg.cholesky_ex(hessian, check_errors=False)
        if int(info.item()) == 0:
            hessian_inverse = torch.cholesky_inverse(chol)
            del chol
            break
    if hessian_inverse is None:
        # A dense pseudo-inverse has an enormous SVD workspace and can OOM even
        # when the normal GPTQ state fits. Use a diagonal Hessian approximation
        # as a safe last resort; this keeps the quantizer bounded and still
        # applies activation-aware per-column weighting.
        inverse_diagonal = hessian.diagonal().clamp_min(eps).reciprocal()
    del hessian

    encoded = torch.empty(
        (out_features, padded_in_features), dtype=torch.uint8, device=device
    )
    scales = torch.empty(
        (out_features, num_groups), dtype=torch.float32, device=device
    )
    zeros: torch.Tensor | None
    if symmetric:
        zeros = None
    else:
        zeros = torch.empty(
            (out_features, num_groups), dtype=torch.uint8, device=device
        )

    # GPTQ uses fixed group parameters found from the original (unmodified)
    # weight. The error-compensated work matrix must not change the scale after
    # the first column of a group has been processed; doing so is a common
    # source of avoidable calibration drift.
    for group in range(num_groups):
        group_start = group * group_size
        group_end = group_start + group_size
        group_weights = work[:, group_start:group_end]
        if symmetric:
            scales[:, group] = group_weights.abs().amax(dim=1).div(7.0).clamp_min(eps)
        else:
            minimum = group_weights.amin(dim=1)
            maximum = group_weights.amax(dim=1)
            minimum = torch.minimum(minimum, torch.zeros_like(minimum))
            maximum = torch.maximum(maximum, torch.zeros_like(maximum))
            scale = (maximum - minimum).div(15.0).clamp_min(eps)
            scales[:, group] = scale
            assert zeros is not None
            zeros[:, group] = torch.round(-minimum / scale).clamp(0, 15).to(torch.uint8)

    # Quantize columns in GPTQ order. Quantization error is propagated through
    # H^{-1}; block_size bounds the temporary correction matrix kept in memory.
    predicted_loss = torch.zeros((), device=device, dtype=torch.float64)
    for block_start in range(0, padded_in_features, block_size):
        block_end = min(block_start + block_size, padded_in_features)
        block_error = torch.empty(
            (out_features, block_end - block_start),
            device=device,
            dtype=torch.float32,
        )
        for column in range(block_start, block_end):
            group = column // group_size
            scale = scales[:, group]
            if symmetric:
                quantized_value = torch.round(work[:, column] / scale).clamp(-8, 7)
                encoded[:, column] = (quantized_value.to(torch.int16) + 8).to(
                    torch.uint8
                )
                reconstructed = quantized_value * scale
            else:
                assert zeros is not None
                zero = zeros[:, group].to(torch.float32)
                quantized_code = torch.round(work[:, column] / scale + zero).clamp(
                    0, 15
                )
                encoded[:, column] = quantized_code.to(torch.uint8)
                reconstructed = (quantized_code - zero) * scale

            if hessian_inverse is not None:
                diagonal_value = hessian_inverse[column, column].clamp_min(eps)
            else:
                assert inverse_diagonal is not None
                diagonal_value = inverse_diagonal[column]
            error = (work[:, column] - reconstructed) / diagonal_value
            block_error[:, column - block_start] = error
            predicted_loss += error.double().square().sum() * diagonal_value.double()

            if hessian_inverse is not None and column + 1 < block_end:
                work[:, column + 1 : block_end].sub_(
                    error[:, None] * hessian_inverse[column, column + 1 : block_end][None, :]
                )

        if hessian_inverse is not None and block_end < padded_in_features:
            work[:, block_end:].sub_(
                block_error.matmul(hessian_inverse[block_start:block_end, block_end:])
            )
        del block_error

    quantized = QuantizedWeight(
        qweight=pack_int4(encoded),
        scales=scales.to(scale_dtype),
        zeros=zeros,
        original_shape=(out_features, in_features),
        padded_shape=(out_features, padded_in_features),
        bits=4,
        group_size=group_size,
        symmetric=symmetric,
        packing="uint8_little_nibble",
    )
    metadata: dict[str, float | int] = {
        "activation_tokens": tokens,
        "block_size": block_size,
        "damp_percent": damp_percent,
        "dead_columns": dead_columns,
        "predicted_loss": float(predicted_loss.item()),
    }
    return quantized, metadata


class GPTQQuantizationMethod:
    name = "gptq"
    version = "1"

    def calibrate_layer(self, context: LayerContext) -> GPTQLayerState:
        if context.activations is None:
            raise ValueError("GPTQ requires calibration activations")

        options = _parse_gptq_options(context.config.calibration or {})
        modules = dict(context.layer.named_modules())
        states: dict[str, GPTQModuleState] = {}
        for name in context.target_modules:
            module = modules.get(name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"target is not a Linear module: {name}")
            if name not in context.activations:
                raise ValueError(f"missing calibration activations for {name}")

            activations = context.activations[name]
            if activations.ndim != 2 or activations.shape[1] != module.in_features:
                raise ValueError(
                    f"invalid calibration activation shape for {name}: "
                    f"expected [tokens, {module.in_features}], "
                    f"got {tuple(activations.shape)}"
                )
            if activations.shape[0] == 0:
                raise ValueError(f"calibration activations for {name} are empty")
            if not activations.is_floating_point():
                raise TypeError(f"calibration activations for {name} must be floating")
            if not torch.isfinite(activations).all():
                raise ValueError(f"calibration activations for {name} are not finite")

            states[name] = GPTQModuleState(
                activations=activations.detach(),
                activation_tokens=activations.shape[0],
            )

        return GPTQLayerState(modules=states, options=options)

    def quantize_layer(
        self, context: LayerContext, state: GPTQLayerState
    ) -> LayerQuantizationResult:
        if not isinstance(state, GPTQLayerState):
            raise TypeError("state must be a GPTQLayerState")

        scale_dtype = SCALE_DTYPES[context.config.scale_dtype]
        modules = dict(context.layer.named_modules())
        weights: dict[str, QuantizedWeight] = {}
        module_metadata: dict[str, dict[str, float | int]] = {}

        for name in context.target_modules:
            module = modules.get(name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"target is not a Linear module: {name}")
            module_state = state.modules.get(name)
            if module_state is None:
                raise ValueError(f"missing GPTQ calibration state for {name}")

            try:
                weights[name], module_metadata[name] = quantize_weight_gptq(
                    module.weight,
                    module_state.activations,
                    context.config.group_size,
                    block_size=state.options.block_size,
                    damp_percent=state.options.damp_percent,
                    symmetric=context.config.symmetric,
                    scale_dtype=scale_dtype,
                )
            except RuntimeError as error:
                raise RuntimeError(
                    f"GPTQ failed for layer {context.layer_index} module {name} "
                    f"with {module_state.activation_tokens} calibration tokens: {error}"
                ) from error

        return LayerQuantizationResult(
            weights=weights,
            metadata={
                "gptq": {
                    "block_size": state.options.block_size,
                    "damp_percent": state.options.damp_percent,
                    "modules": module_metadata,
                }
            },
        )
