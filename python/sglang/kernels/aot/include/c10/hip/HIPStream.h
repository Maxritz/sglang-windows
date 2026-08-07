#pragma once
// Redirect: hipify-gen'd duplicate of c10/cuda/CUDAStream.h (redefines the
// CUDAStream class / stream priorities / default args). Pull the cuda twin,
// which the force-include shim already loads.
#include <c10/cuda/CUDAStream.h>
