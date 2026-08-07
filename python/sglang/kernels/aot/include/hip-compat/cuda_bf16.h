// CUDA/HIP bfloat16 compatibility shim for ROCm on Windows.
//
// The ROCm 7.13 Windows SDK ships cuda_fp16.h / cuda_runtime.h in its default
// include path but, unlike the Linux SDK, does NOT declare __nv_bfloat16
// (amd_hip_bfloat16.h only defines the hip_bfloat16 / __hip_bfloat16 structs,
// which are distinct types). SGLang's CUDA sources include <cuda_bf16.h> and
// reference __nv_bfloat16 directly (e.g.
// grammar/apply_token_bitmask_inplace_cuda.cu, moe/moe_topk_*_kernels.cu).
//
// IMPORTANT include ordering rule:
// This header MUST be included BEFORE <torch/all.h>. torch pulls in the clang
// HIP runtime's virtual (in-memory) hip/impl/hip_vec_bf16_impl.h, which lazily
// defines `using __nv_bfloat16 = __hip_bfloat16;`. That virtual definition is
// guarded against an already-declared __nv_bfloat16, so declaring it first here
// makes our definition win and the virtual one gets suppressed. Declaring it
// AFTER torch redefines __nv_bfloat16 -> "type alias redefinition".
//
// __nv_bfloat16 maps to ROCm's __hip_bfloat16 (NOT the public hip_bfloat16
// struct): the ROCm intrinsics the CUDA sources rely on (e.g. __bfloat162float)
// take __hip_bfloat16, reinterpret_cast<const __nv_bfloat16*> must alias the
// in-memory 16-bit BFloat16 storage, and __hip_bfloat16 has the float ctor.
// Both are reachable via <hip/hip_bfloat16.h> (which pulls hip_runtime.h ->
// amd_hip_bf16.h).
#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/amd_detail/amd_hip_bf16.h>

using __nv_bfloat16 = __hip_bfloat16;
