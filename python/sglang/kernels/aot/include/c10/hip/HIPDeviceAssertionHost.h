#pragma once
// Redirect: hipify-gen'd duplicate of c10/cuda/CUDADeviceAssertionHost.h
// (redefines C10_CUDA_DSA_ASSERTION_COUNT / DeviceAssertionData / CUDAKernelLaunchRegistry
// when both twins load). Pull the cuda twin so the DSA machinery is defined once.
#include <c10/cuda/CUDADeviceAssertionHost.h>
