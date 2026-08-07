# sglang-kernel (prior sgl-kernel)

[Kernel Library](https://github.com/sgl-project/sglang/tree/main/python/sglang/kernels/aot) for LLM inference engines

<div align="center">

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://github.com/sgl-project/sglang/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/sglang-kernel)](https://pypi.org/project/sglang-kernel)

</div>

`sglang-kernel` provides optimized compute primitives for LLM inference engines, enabling efficient inference for large language models and vision-language models through custom kernel operations. The source tree lives under the `python/sglang/kernels/aot/` directory and the Python import path remains `sgl_kernel`.

## Installation
Requires torch == 2.11.0

```bash
# Latest version
pip3 install sglang-kernel --upgrade
```

## Building from Source
Requires
- CMake ≥3.31,
- Python ≥3.10
- scikit-build-core
- ninja(optional)

### Use Makefile to build from the sgl-kernel source tree

```bash
make build
```

### Limit build resource usage (CPU / parallelism)

By default, `make build` uses all available CPU cores. You can override build parallelism and NVCC compile threads:

```bash
# Limit parallel jobs (controls both make and cmake parallelism)
make build MAX_JOBS=2

# Additionally limit NVCC internal threads (reduces CPU and peak memory)
make build MAX_JOBS=2 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"
```

### Building on Windows for AMD RDNA4 (gfx1201)

The ROCm Windows port builds `common_ops.pyd` in place with `setup_rocm.py` (hipcc/clang driven through ninja):

```bash
# From a "Developer PowerShell for VS" prompt (vcvars64 loaded) with the aot dir as CWD
$env:DISTUTILS_USE_SDK = "1"
python setup_rocm.py build_ext --inplace
```

Requirements and gotchas:

- **AMD ROCm for Windows** (hipcc/clang + HIP/hipBLAS/hipRTC libs), **CMake ≥3.31**, **ninja**, Python ≥3.10, torch `2.11.0+rocm`.
- **All translation units must compile with hipcc/clang.** MSVC `cl` cannot parse ROCm HIP headers. This is why the extension entry point is `csrc/common_extension_rocm.cu` and `csrc/memory/weak_ref_tensor.cu` (rather than `.cc`): MSVC emitted dllimport thunks for `c10::ValueError`'s inherited constructor that the torch DLLs do not export (`LNK2001`), while clang inlines it to the exported base ctor.
- Build flags baked into `setup_rocm.py`: `-DUSE_ROCM`, `C10_CUDA_BUILD_SHARED_LIBS=1` (mandatory), `-fms-runtime-lib=dll` (torch wheel DLLs are `/MD`), `__HIP_NO_HALF_OPERATORS__=1` / `__HIP_NO_HALF_CONVERSIONS__=1`, and gfx1201/RDNA4 defines.
- `include/` contains small shims that the torch wheels do not ship for HIP: `ATen/core/TensorBase.h` (out-of-line `data_ptr<T>()`), `ATen/hip/`, `c10/hip/`, `hip-compat/`, and `torch/all.h`.
- **hipify-generated files are not committed.** `setup_rocm.py` regenerates `.hip` twins (e.g. `include/torch/all_hip.h` and the `csrc/**/*.hip` files under `.gitignore`) from the `.cu`/`.h` sources. Exceptions tracked by hand: `csrc/allreduce/{custom_all_reduce,deterministic_all_reduce}.hip` (originally hip-native) and `csrc/mamba/causal_conv1d.hip` (source of truth).
- Deployed output lands in `python/sgl_kernel/sm100/` (`torch.cuda.get_device_properties(0).major=12, minor=0` on RDNA4 → `load_utils` maps it to the `sm100` subdir).

Smoke check (validates `rotary_embedding`, `moe_align_block_size`, `apply_token_bitmask_inplace_cuda`, `weak_ref_tensor`):

```bash
.venv312\Scripts\python.exe -c "import sgl_kernel; import torch; \
  a = torch.randn(4, 8, 128, dtype=torch.float16, device='cuda'); \
  sgl_kernel.rotary_embedding(a, torch.randn(4, 8, 8, dtype=torch.float16, device='cuda'), torch.randint(0, 128, (4,), dtype=torch.int32, device='cuda'), a, 128, 1)"
```

Note: `apply_token_bitmask_inplace_cuda`'s bitmask convention is bit=1 = allowed; an "allow all" mask is `torch.full((n,), -1, dtype=torch.int32)`, not `torch.ones`.

## Contribution

### Steps to add a new kernel:

1. Implement the kernel in [csrc](https://github.com/sgl-project/sglang/tree/main/python/sglang/kernels/aot/csrc)
2. Expose the interface in [include/sgl_kernel_ops.h](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/aot/include/sgl_kernel_ops.h)
3. Create torch extension in [csrc/common_extension.cc](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/aot/csrc/common_extension.cc)
4. Update [CMakeLists.txt](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/aot/CMakeLists.txt) to include new CUDA source
5. Expose Python interface in [python](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/aot/python/sgl_kernel)
6. Add test and benchmark

### Development Tips

1. When creating torch extensions, add the function definition with `m.def`, and device binding with `m.impl`:

- How to write schema: [Schema reference](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func)

   ```cpp
   // We need def with schema here for torch.compile
   m.def(
    "bmm_fp8(Tensor A, Tensor B, Tensor! D, Tensor A_scale, Tensor B_scale, Tensor workspace_buffer, "
    "int cublas_handle) -> ()");
   m.impl("bmm_fp8", torch::kCUDA, &bmm_fp8);
   ```

### Adapting C++ Native Types for Torch Compatibility

Third-party C++ libraries often use int and float, but PyTorch bindings require int64_t and double due to Python's type mapping.

Use make_pytorch_shim from sgl_kernel_torch_shim.h to handle conversions automatically:

```cpp

// Add type conversion for int -> int64_t
template <>
struct pytorch_library_compatible_type<int> {
  using type = int64_t;
  static int convert_from_type(int64_t arg) {
    TORCH_CHECK(arg <= std::numeric_limits<int>::max(), "value too large");
    TORCH_CHECK(arg >= std::numeric_limits<int>::min(), "value too small");
    return arg;
  }
};
```
```cpp
// Wrap your function
m.impl("fwd", torch::kCUDA, make_pytorch_shim(&mha_fwd));
```

### Testing & Benchmarking

1. Add pytest tests in [tests/](https://github.com/sgl-project/sglang/tree/main/python/sglang/kernels/aot/tests), if you need to skip some test, please use `@pytest.mark.skipif`

```python
@pytest.mark.skipif(
    skip_condition, reason="Nvfp4 Requires compute capability of 10 or above."
)
```

2. Add benchmarks using [triton benchmark](https://triton-lang.org/main/python-api/generated/triton.testing.Benchmark.html) in [benchmark/](https://github.com/sgl-project/sglang/tree/main/python/sglang/kernels/aot/benchmark)

   **We recommend using `triton.testing.do_bench_cudagraph` for kernel benchmarking**:

   Compared to `triton.testing.do_bench`, `do_bench_cudagraph` provides:
   - Reduced CPU overhead impact for more accurate kernel performance measurements
   - Incorporation of PDL (Programmatic Dependent Launch) effects into individual kernel results
   - More realistic performance data on PDL-supported architectures (SM >= 90)

3. Run test suite

## Kernel Size Analysis

Analyze CUDA kernel sizes in compiled wheel files to identify oversized kernels and template-instantiation bloat:

This tool requires `cubloaty` (install with `pip install cubloaty`) to work.

```bash
# Install cubloaty
pip install cubloaty

# Analyze a wheel file
python analyze_whl_kernel_sizes.py path/to/sglang_kernel-*.whl

# Custom output file
python analyze_whl_kernel_sizes.py path/to/sglang_kernel-*.whl --output my_analysis.txt
```

The tool generates:
- A text report with:
  - Kernel groups (by name prefix)
  - Individual kernel sizes (sorted by size)

Use this to identify large kernels and potential template instantiation bloat.

## FAQ
- Q: Segmentation fault with CUDA 12.6
- A: Update ptxas to 12.8, reference: [segment fault error](https://github.com/Dao-AILab/flash-attention/issues/1453)
