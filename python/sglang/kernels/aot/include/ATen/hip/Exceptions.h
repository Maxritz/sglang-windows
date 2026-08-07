#pragma once
// Redirect: hipify duplicate of ATen/cuda/Exceptions.h. Pull the cuda twin (hipified
// under USE_ROCM) so CuDNNError / _hipsolver_backend_suggestion are defined once and
// TORCH_DSA_KERNEL_ARGS is not macro-redefined against its c10/cuda counterpart.
#include <ATen/cuda/Exceptions.h>
