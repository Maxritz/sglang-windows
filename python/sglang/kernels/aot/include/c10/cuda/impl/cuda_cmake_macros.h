// This ROCm-on-Windows torch wheel ships the c10/cuda headers but omits the
// CMake-generated c10/cuda/impl/cuda_cmake_macros.h that upstream torch's
// CUDAMacros.h pulls in.
//
// CUDAMacros.h (c10/cuda) uses this file to decide how C10_CUDA_API symbols
// are annotated on Windows:
//   #if defined(C10_CUDA_BUILD_SHARED_LIBS) -> EXPORT=dllexport, IMPORT=dllimport
//   #else                                   -> both empty
//
// With the wheel built as shared DLLs (c10_hip.dll / torch_hip.dll), a
// downstream extension is a CONSUMER and MUST have the import annotations
// active. Without them, every C10_CUDA_API symbol (functions and data such as
// c10::cuda::CUDACachingAllocator::allocator) is referenced as a plain extern;
// data imports in particular can only be satisfied via their __imp_ thunk, so
// the link fails with unresolved externals. Hence we define
// C10_CUDA_BUILD_SHARED_LIBS here, making C10_CUDA_API == __declspec(dllimport)
// and routing all C10_CUDA_API references through the torch import libs.
#define C10_CUDA_BUILD_SHARED_LIBS
#pragma once
// The wheel's torch/csrc headers also reference TORCH_CUDA_CPP_API /
// TORCH_CUDA_CU_API / TORCH_CUDA_API / ... which upstream generates in this
// file. This wheel never defines them; empty is correct here because those
// symbols are plain function imports satisfied by the torch import libs
// (torch.lib / torch_cpu.lib / torch_python.lib) via their thunks, and none of
// them are data symbols.
#ifndef TORCH_CUDA_CPP_API
#define TORCH_CUDA_CPP_API
#endif
#ifndef TORCH_CUDA_CU_API
#define TORCH_CUDA_CU_API
#endif
#ifndef TORCH_CUDA_API
#define TORCH_CUDA_API
#endif
#ifndef TORCH_HIP_CUDA_API
#define TORCH_HIP_CUDA_API
#endif
#ifndef TORCH_GLOBAL_CUDA_API
#define TORCH_GLOBAL_CUDA_API
#endif
#ifndef TORCH_CUDA_CXX17_API
#define TORCH_CUDA_CXX17_API
#endif
#ifndef TORCH_CUDA_LIB_EXPORT
#define TORCH_CUDA_LIB_EXPORT
#endif
#ifndef TORCH_CUDA_LIB_API
#define TORCH_CUDA_LIB_API
#endif
