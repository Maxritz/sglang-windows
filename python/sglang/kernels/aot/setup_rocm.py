# Copyright 2025 SGLang Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import platform
import sys
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent.resolve()
arch = platform.machine().lower()


def _get_version():
    with open(root / "pyproject.toml") as f:
        for line in f:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')


operator_namespace = "sgl_kernel"
include_dirs = [
    root / "include",
    root / "include" / "impl",
    root / "csrc",
]

# Compatibility shims so SGL's CUDA-style kernels (hipLaunchKernelGGL,
# cuda_bf16, cudaFuncSetAttribute, ...) build under ROCm on RDNA4/gfx1201.
include_dirs += [
    root / "include" / "hip-compat",
]

# Source list mirrors the proven port-test build (port-test/compile_port.ps1):
#   - es_sm100_mxfp8_blockscaled_group_quant.cu is EXCLUDED on RDNA4: its launcher
#     (es_..._group_quant_hip.cuh) uses Hopper/Blackwell bulk-TMA PTX + CuTe, which
#     do not exist on gfx1201. SM100 (CUDA) builds register it under !SGLANG_RDNA4.
#   - causal_conv1d.hip is the hipcub port (the .cu pulls NVIDIA CUB).
sources = [
    "csrc/allreduce/custom_all_reduce.hip",
    "csrc/allreduce/deterministic_all_reduce.hip",
    "csrc/allreduce/quick_all_reduce.cu",
    "csrc/common_extension_rocm.cu",
    "csrc/elementwise/activation.cu",
    "csrc/elementwise/copy.cu",
    "csrc/elementwise/deepseek_v4_topk.cu",
    "csrc/elementwise/dsv4_norm_rope.cu",
    "csrc/elementwise/pos_enc.cu",
    "csrc/elementwise/topk.cu",
    "csrc/expert_specialization/es_sm100_mxfp8_blockscaled.cu",
    "csrc/gemm/fp8_gemm_kernel.cu",
    "csrc/grammar/apply_token_bitmask_inplace_cuda.cu",
    "csrc/kvcacheio/transfer.cu",
    "csrc/mamba/causal_conv1d.hip",
    "csrc/memory/weak_ref_tensor.cu",
    "csrc/moe/moe_align_kernel.cu",
    "csrc/moe/moe_topk_softmax_kernels.cu",
    "csrc/moe/moe_topk_sigmoid_kernels.cu",
    "csrc/quantization/gguf/gguf_kernel.cu",
    "csrc/speculative/eagle_utils.cu",
    "csrc/speculative/ngram_utils.cu",
]

cxx_flags = ["-O3"]
if sys.platform == "win32":
    torch_include = Path(torch.__file__).parent / "include"
    cxx_flags.append(
        "-DSGL_TORCH_INCLUDE_DIR=" + (torch_include / "ATen" / "core" / "TensorBase.h").as_posix()
    )
if sys.platform == "win32":
    # The core at::/_ops/TensorMaker symbols live in torch_cpu.lib, and the HIP
    # stream/allocator symbols in c10_hip.lib -- both missing from torch.lib on
    # this ROCm Windows wheel. Match the proven port-test link line.
    libraries = [
        "torch_hip",
        "torch_cpu",
        "torch_python",
        "c10",
        "c10_hip",
        "amdhip64",
        "hipblas",
        "hipsolver",
        "hipsparse",
        "rocsolver",
        "rocsparse",
        "hiprtc",
    ]
    extra_link_args = ["/LIBPATH:E:/ROCM-7.13.0-Windows/lib"]
else:
    libraries = ["hiprtc", "amdhip64", "c10", "torch", "torch_python"]
    extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib", f"-L/usr/lib/{arch}-linux-gnu"]

default_target = "gfx942"
amdgpu_target = os.environ.get("AMDGPU_TARGET", default_target)

if torch.cuda.is_available():
    try:
        amdgpu_target = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception as e:
        print(f"Warning: Failed to detect GPU properties: {e}")
else:
    print(f"Warning: torch.cuda not available. Using default target: {amdgpu_target}")

if amdgpu_target not in ["gfx942", "gfx950", "gfx1201"]:
    print(
        f"Warning: Unsupported GPU architecture detected '{amdgpu_target}'. "
        f"Expected 'gfx942', 'gfx950', or 'gfx1201' (RDNA4)."
    )
    sys.exit(1)

fp8_macro = (
    "-DHIP_FP8_TYPE_FNUZ" if amdgpu_target == "gfx942" else "-DHIP_FP8_TYPE_E4M3"
)

# Dynamic shared-memory budget for the TopK kernels.
# - gfx942 (MI300/MI325): LDS is typically 64KB per workgroup -> keep dynamic smem <= ~48KB
#   (leaves room for static shared allocations in the kernel).
# - gfx95x (MI350): LDS is larger (e.g. 160KB per CU) -> allow the original 128KB dynamic smem.
topk_dynamic_smem_bytes = 48 * 1024 if amdgpu_target in ("gfx942", "gfx1201") else 32 * 1024 * 4

hipcc_flags = [
    "-DNDEBUG",
    f"-DOPERATOR_NAMESPACE={operator_namespace}",
    "-O3",
    "-std=c++17",
    f"--offload-arch={amdgpu_target}",
    "-DENABLE_BF16",
    "-DENABLE_FP8",
    fp8_macro,
    f"-DSGL_TOPK_DYNAMIC_SMEM_BYTES={topk_dynamic_smem_bytes}",
    "-DSGLANG_RDNA4",
    "-DUSE_ROCM",
    # Do NOT define C10_CUDA_NO_CMAKE_CONFIGURE_FILE: c10/cuda/impl/cuda_cmake_macros.h
    # is supplied by our include/ shim and defines C10_CUDA_BUILD_SHARED_LIBS so that
    # C10_CUDA_API symbols (e.g. CUDACachingAllocator::allocator, a data import) are
    # referenced via their __imp_ thunks. Skipping it makes them plain extern and the
    # link fails on Windows. Upstream defined the flag because the HIP build never
    # generated that header; our shim makes the include valid and necessary.
    "-include",
    str(root / "include" / "hip-compat" / "cuda_runtime_api.h"),
]
if sys.platform == "win32":
    # clang for the windows-msvc target defaults its C++ stdlib defaultlib to STATIC
    # libcpmt.lib; the torch wheel DLLs are /MD. Force the dynamic CRT to match.
    hipcc_flags.append("-fms-runtime-lib=dll")

ext_modules = [
    CUDAExtension(
        name="sgl_kernel.common_ops",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "nvcc": hipcc_flags,
            "cxx": cxx_flags,
        },
        libraries=libraries,
        extra_link_args=extra_link_args,
        py_limited_api=False,
    ),
]

setup(
    name="sglang-kernel",
    version=_get_version(),
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
