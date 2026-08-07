#pragma once
// This ROCm wheel lacks c10/core/Tensor.h (the upstream home of the c10::Tensor
// class); the class definition lives in ATen/core/TensorBody.h instead. Route
// any #include <c10/core/Tensor.h> there so the aggregate / forward decls used by
// the rest of the header tree resolve. Only at::Tensor is re-exported here.
#include <ATen/core/TensorBody.h>
