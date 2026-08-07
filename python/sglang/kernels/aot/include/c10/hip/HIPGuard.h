#pragma once
// Redirect: hipify-gen'd duplicate of c10/cuda/CUDAGuard.h. This wheel's c10/hip
// twins keep CUDA-named content, so including both twins redefines. Pull the
// cuda twin (loaded first via the force-include shim) so CUDAGuard/CUDAGuardImpl
// are defined exactly once.
#include <c10/cuda/CUDAGuard.h>
