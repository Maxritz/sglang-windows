#pragma once
// Redirect: hipify-gen'd duplicate of c10/cuda/CUDAFunctions.h (default-argument
// redefinition when both twins load). Pull the cuda twin so current_device and
// friends are declared exactly once.
#include <c10/cuda/CUDAFunctions.h>
