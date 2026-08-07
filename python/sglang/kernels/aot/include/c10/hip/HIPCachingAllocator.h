#pragma once
// Redirect: this wheel ships separate c10/cuda and c10/hip CachingAllocator headers
// with identical FreeMemoryCallback definitions. Pull the cuda (hipified-under-USE_ROCM)
// twin so the symbol is defined exactly once instead of triggering a redefinition error.
#include <c10/cuda/CUDACachingAllocator.h>
