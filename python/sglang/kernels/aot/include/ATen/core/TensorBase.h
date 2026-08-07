#pragma once
// Redirect shim for the ROCm-Windows torch wheel:
// ATen/core/TensorBase.h in this wheel *declares* the templated
// data_ptr<T>()/mutable_data_ptr<T>()/const_data_ptr<T>() accessors
// (lines ~632-651) but never *defines* them -- upstream torch inlines the
// definitions at the bottom of this same header. Without them, clang emits
// out-of-line calls to undefined TensorBase::data_ptr<T>() etc. at link time.
//
// Canonical port pattern: include the real twin, then supply the missing
// out-of-line (header-only) template definitions.
//
// clang (hipcc) supports #include_next to skip past this shim. MSVC cl does
// not, so setup_rocm.py defines SGL_TORCH_INCLUDE_DIR as the absolute path
// (forward slashes) to the wheel's ATen/core/TensorBase.h, pulled in through a
// single stringized include directive.
#if defined(__clang__)
#include_next <ATen/core/TensorBase.h>
#else
#ifndef SGL_TORCH_INCLUDE_DIR
#error "SGL_TORCH_INCLUDE_DIR must point at the torch ATen/core/TensorBase.h when compiling with MSVC cl"
#endif
#define SGL_TORCH_INCLUDE_STR_IMPL(x) #x
#define SGL_TORCH_INCLUDE_STR(x) SGL_TORCH_INCLUDE_STR_IMPL(x)
#include SGL_TORCH_INCLUDE_STR(SGL_TORCH_INCLUDE_DIR)
#endif

namespace at {

template <typename T, std::enable_if_t<!std::is_const_v<T>, int>>
const T* TensorBase::const_data_ptr() const {
  return static_cast<const T*>(const_data_ptr());
}

template <typename T, std::enable_if_t<std::is_const_v<T>, int>>
const std::remove_const_t<T>* TensorBase::const_data_ptr() const {
  return static_cast<const std::remove_const_t<T>*>(const_data_ptr());
}

template <typename T>
T* TensorBase::mutable_data_ptr() const {
  static_assert(!std::is_const_v<T>, "mutable_data_ptr<T>() requires a non-const T");
  return static_cast<T*>(mutable_data_ptr());
}

template <typename T>
T* TensorBase::data_ptr() const {
  return static_cast<T*>(data_ptr());
}

} // namespace at
