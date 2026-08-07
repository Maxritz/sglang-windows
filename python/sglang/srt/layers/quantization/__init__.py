# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/v0.5.5/vllm/model_executor/layers/quantization/__init__.py
from __future__ import annotations

import builtins
import inspect
from typing import TYPE_CHECKING, Dict, Optional, Type

import torch


# Define empty classes as placeholders when vllm is not available
class DummyConfig:
    def override_quantization_method(self, *args, **kwargs):
        return None


CompressedTensorsConfig = DummyConfig


from sglang.srt.layers.quantization.base_config import QuantizationConfig


def _optional_import(names, module_name):
    """Import quantization backend configs, falling back to DummyConfig.

    Some backends (e.g. quark -> aiter) pull in GPU-specific libraries that are
    not present on every platform (notably ROCm/Windows wheels).  They are
    only selected when a model is actually quantized with that method, so a
    missing optional backend should not break importing the quantization
    registry.  On hosts where the backend is installable the real class is used
    unchanged.
    """
    import importlib

    try:
        mod = importlib.import_module(module_name)
        return tuple(getattr(mod, n) for n in names)
    except ImportError:
        for n in names:
            globals()[n] = DummyConfig
        return tuple(DummyConfig for _ in names)


AutoRoundConfig, = _optional_import(("AutoRoundConfig",), "sglang.srt.layers.quantization.auto_round")
AWQConfig, AWQCPUConfig, AWQMarlinConfig = _optional_import(
    ("AWQConfig", "AWQCPUConfig", "AWQMarlinConfig"),
    "sglang.srt.layers.quantization.awq",
)
BitsAndBytesConfig, = _optional_import(("BitsAndBytesConfig",), "sglang.srt.layers.quantization.bitsandbytes")
BlockInt8Config, = _optional_import(("BlockInt8Config",), "sglang.srt.layers.quantization.blockwise_int8")
(CompressedTensorsConfig,) = _optional_import(
    ("CompressedTensorsConfig",), "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors"
)
Fp8Config, = _optional_import(("Fp8Config",), "sglang.srt.layers.quantization.fp8")
GGUFConfig, = _optional_import(("GGUFConfig",), "sglang.srt.layers.quantization.gguf")
CPUGPTQConfig, GPTQAscendConfig, GPTQConfig, GPTQMarlinConfig = _optional_import(
    ("CPUGPTQConfig", "GPTQAscendConfig", "GPTQConfig", "GPTQMarlinConfig"),
    "sglang.srt.layers.quantization.gptq",
)
HummingConfig, = _optional_import(("HummingConfig",), "sglang.srt.layers.quantization.humming")
MlxQuantizationConfig, = _optional_import(("MlxQuantizationConfig",), "sglang.srt.layers.quantization.mlx")
ModelOptFp4Config, ModelOptFp8Config, ModelOptMixedPrecisionConfig = _optional_import(
    ("ModelOptFp4Config", "ModelOptFp8Config", "ModelOptMixedPrecisionConfig"),
    "sglang.srt.layers.quantization.modelopt_quant",
)
ModelSlimConfig, = _optional_import(("ModelSlimConfig",), "sglang.srt.layers.quantization.modelslim.modelslim")
MoeWNA16Config, = _optional_import(("MoeWNA16Config",), "sglang.srt.layers.quantization.moe_wna16")
Mxfp4Config, = _optional_import(("Mxfp4Config",), "sglang.srt.layers.quantization.mxfp4")
Mxfp4W4A8Config, = _optional_import(("Mxfp4W4A8Config",), "sglang.srt.layers.quantization.npu_mxfp4")
Mxfp4W4A4Config, = _optional_import(("Mxfp4W4A4Config",), "sglang.srt.layers.quantization.npu_mxfp4_w4a4")
NvFp4OnlineConfig, = _optional_import(("NvFp4OnlineConfig",), "sglang.srt.layers.quantization.nvfp4_online")
PetitNvFp4Config, = _optional_import(("PetitNvFp4Config",), "sglang.srt.layers.quantization.petit")
QuarkConfig, = _optional_import(("QuarkConfig",), "sglang.srt.layers.quantization.quark.quark")
QuarkInt4Fp8Config, = _optional_import(("QuarkInt4Fp8Config",), "sglang.srt.layers.quantization.quark_int4fp8_moe")
W4AFp8Config, = _optional_import(("W4AFp8Config",), "sglang.srt.layers.quantization.w4afp8")
W8A8Fp8Config, = _optional_import(("W8A8Fp8Config",), "sglang.srt.layers.quantization.w8a8_fp8")
W8A8Int8Config, = _optional_import(("W8A8Int8Config",), "sglang.srt.layers.quantization.w8a8_int8")
from sglang.srt.platforms import current_platform
from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_gfx95_supported,
    is_mps,
    is_npu,
)

_is_gfx95_supported = is_gfx95_supported()

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKOutput

# Base quantization methods
BASE_QUANTIZATION_METHODS: Dict[str, Type[QuantizationConfig]] = {
    "fp8": Fp8Config,
    "mxfp8": Fp8Config,
    "blockwise_int8": BlockInt8Config,
    "modelopt": ModelOptFp8Config,  # Auto-detect, defaults to FP8
    "modelopt_fp8": ModelOptFp8Config,
    "modelopt_fp4": ModelOptFp4Config,
    "nvfp4_online": NvFp4OnlineConfig,
    "modelopt_mixed": ModelOptMixedPrecisionConfig,
    "w8a8_int8": W8A8Int8Config,
    "w8a8_fp8": W8A8Fp8Config,
    "awq": AWQConfig,
    "awq_marlin": AWQMarlinConfig,
    "bitsandbytes": BitsAndBytesConfig,
    "gguf": GGUFConfig,
    "gptq": GPTQConfig,
    "gptq_marlin": GPTQMarlinConfig,
    "moe_wna16": MoeWNA16Config,
    "compressed-tensors": CompressedTensorsConfig,
    "w4afp8": W4AFp8Config,
    "petit_nvfp4": PetitNvFp4Config,
    "quark": QuarkConfig,
    "quark_mxfp4": QuarkConfig,
    "auto-round": AutoRoundConfig,
    "auto-round-int8": W8A8Int8Config,
    "modelslim": ModelSlimConfig,
    "quark_int4fp8_moe": QuarkInt4Fp8Config,
    "humming": HummingConfig,
    "mxfp_w4a8": Mxfp4W4A8Config,
}


if is_cpu() or is_cuda() or _is_gfx95_supported:
    BASE_QUANTIZATION_METHODS.update(
        {
            "mxfp4": Mxfp4Config,
        }
    )


if is_npu():
    BASE_QUANTIZATION_METHODS.update(
        {
            "gptq": GPTQAscendConfig,
            # On NPU, `mxfp4` means single-level W4A4 MXFP4 for dense LLM (the
            # upstream `Mxfp4Config` OCP-MoE path is only registered on
            # cpu/cuda/hip above, so there is no collision here).
            "mxfp4": Mxfp4W4A4Config,
        }
    )


if is_mps():
    BASE_QUANTIZATION_METHODS.update(
        {
            "mlx_q4": MlxQuantizationConfig,
            "mlx_q8": MlxQuantizationConfig,
        }
    )

# subset of above quant methods, supported on CPU
CPU_QUANTIZATION_METHODS = {
    "fp8": Fp8Config,
    "w8a8_int8": W8A8Int8Config,
    "compressed-tensors": CompressedTensorsConfig,
    "awq": AWQCPUConfig,
    "gptq": CPUGPTQConfig,
    "mxfp4": Mxfp4Config,
}

QUANTIZATION_METHODS = {**BASE_QUANTIZATION_METHODS}


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(
            f"Invalid quantization method: {quantization}. "
            f"Available methods: {list(QUANTIZATION_METHODS.keys())}"
        )
    from sglang.srt.utils import is_cpu

    if is_cpu() and cpu_has_amx_support():
        if quantization not in CPU_QUANTIZATION_METHODS:
            raise ValueError(
                f"Invalid quantization method on CPU: {quantization}. "
                f"Available methods on CPU: {list(QUANTIZATION_METHODS.keys())}"
            )
        else:
            return CPU_QUANTIZATION_METHODS[quantization]

    if current_platform.is_out_of_tree():
        config = current_platform.get_quantization_config(quantization)

        # If the platform has a quantization config, use it else use the default
        if config is not None:
            return config

    return QUANTIZATION_METHODS[quantization]


original_isinstance = builtins.isinstance
