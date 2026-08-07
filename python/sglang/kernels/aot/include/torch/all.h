#pragma once
// This ROCm-on-Windows torch wheel ships the headers under torch/include
// but omits the upstream torch/all.h aggregate (not generated for the ROCm
// build) AND it merged c10/core/Tensor.h into ATen/core/TensorBody.h. Provide a
// minimal aggregate pulling only the symbols sgl_kernel actually uses, from
// headers confirmed present in this wheel.
#include <c10/core/Scalar.h>
#include <c10/core/Allocator.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/SymInt.h>
#include <c10/core/WrapDimMinimal.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/TensorBase.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAMiscFunctions.h>
#include <ATen/cuda/CUDAContext.h>
#if defined(USE_ROCM)
#include <ATen/hip/HIPEvent.h>
#else
#include <ATen/cuda/CUDAEvent.h>
#endif
#include <torch/library.h>
// Real C++ frontend from the wheel (torch/csrc/api/include): provides the torch
// namespace used throughout sgl_kernel (torch::Tensor, torch::kInt32/kUInt8,
// torch::empty/zeros/cat/... via autograd variable_factories). NOTE: the cuda
// twin CUDACachingAllocator.h is intentionally NOT pulled here - this wheel
// ships a hipify-gen'd duplicate of it under c10/hip that redeclares
// c10::FreeMemoryCallback, which redefines once both twins load.
#include <torch/types.h>
