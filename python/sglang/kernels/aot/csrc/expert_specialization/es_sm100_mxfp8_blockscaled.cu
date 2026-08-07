#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>

#include "es_sm100_mxfp8_blockscaled_hip.cuh"

#if !defined(SGLANG_RDNA4)
#include "es_sm100_mxfp8_blockscaled_launcher.cuh"
#endif

namespace expert_specialization {
namespace native {

template <typename OutT>
void es_sm100_mxfp8_blockscaled_grouped_mm_native(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& sfa,
    const torch::Tensor& sfb,
    torch::Tensor& d,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& blockscale_offsets,
    cudaStream_t stream) {
  int num_experts = (int)problem_sizes.size(0);
  auto int32_opts = torch::TensorOptions().dtype(torch::kInt32).device(a.device());
  torch::Tensor tile_base = torch::empty(num_experts + 1, int32_opts);
  torch::Tensor tiles_n = torch::empty(num_experts, int32_opts);

  es_mxfp8_precompute<<<1, num_experts, 0, stream>>>(
      problem_sizes.data_ptr<int>(), num_experts, tile_base.data_ptr<int>(), tiles_n.data_ptr<int>());

  int total_tiles = tile_base.cpu().data_ptr<int>()[num_experts];
  if (total_tiles > 0) {
    es_mxfp8_blockscaled_mm<OutT><<<total_tiles, 128, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(a.data_ptr()),
        reinterpret_cast<const uint8_t*>(b.data_ptr()),
        reinterpret_cast<const uint8_t*>(sfa.data_ptr()),
        reinterpret_cast<const uint8_t*>(sfb.data_ptr()),
        reinterpret_cast<OutT*>(d.data_ptr()),
        problem_sizes.data_ptr<int>(),
        expert_offsets.data_ptr<int>(),
        blockscale_offsets.data_ptr<int>(),
        tile_base.data_ptr<int>(),
        tiles_n.data_ptr<int>(),
        num_experts);
  }
}

}  // namespace native
}  // namespace expert_specialization

void es_sm100_mxfp8_blockscaled_grouped_mm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& sfa,
    const torch::Tensor& sfb,
    torch::Tensor& d,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& blockscale_offsets) {
  TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");
  TORCH_CHECK(problem_sizes.size(1) == 3, "problem_sizes must have shape (num_experts, 3)");
  TORCH_CHECK(
      problem_sizes.size(0) == expert_offsets.size(0), "Number of experts in problem_sizes must match expert_offsets");
  TORCH_CHECK(problem_sizes.dtype() == torch::kInt32, "problem_sizes must be int32");
  TORCH_CHECK(a.dim() == 2, "a must be a 2D tensor of shape (num_tokens, k)");
  TORCH_CHECK(b.dim() == 3, "b must be a 3D tensor of shape (num_experts, k, n)");
  TORCH_CHECK(a.size(1) == b.size(1) && a.size(1) % 128 == 0, "k should align 128");
  TORCH_CHECK(b.size(2) % 128 == 0, "n should align 128");
  TORCH_CHECK(a.strides()[1] == 1, "a must be row major");
  TORCH_CHECK(b.strides()[1] == 1, "a must be column major");

#if defined(SGLANG_RDNA4)
  auto stream = at::cuda::getCurrentCUDAStream();
  if (d.dtype() == torch::kBFloat16) {
    expert_specialization::native::es_sm100_mxfp8_blockscaled_grouped_mm_native<at::BFloat16>(
        a, b, sfa, sfb, d, problem_sizes, expert_offsets, blockscale_offsets, stream);
  } else if (d.dtype() == torch::kHalf) {
    expert_specialization::native::es_sm100_mxfp8_blockscaled_grouped_mm_native<at::Half>(
        a, b, sfa, sfb, d, problem_sizes, expert_offsets, blockscale_offsets, stream);
  } else {
    TORCH_CHECK(false, "dtype must be kFloat16 or kBFloat16");
  }
#elif defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  auto stream = at::cuda::getCurrentCUDAStream();
  if (d.dtype() == torch::kBFloat16) {
    expert_specialization::es_sm100_mxfp8_blockscaled_group_mm_dispatch_out_dtype<cutlass::bfloat16_t>(
        a, b, sfa, sfb, d, problem_sizes, expert_offsets, blockscale_offsets, stream);
  } else if (d.dtype() == torch::kHalf) {
    expert_specialization::es_sm100_mxfp8_blockscaled_group_mm_dispatch_out_dtype<cutlass::half_t>(
        a, b, sfa, sfb, d, problem_sizes, expert_offsets, blockscale_offsets, stream);
  } else {
    TORCH_CHECK(false, "dtype must be kFloat16 or kBFloat16");
  }
#else
  TORCH_CHECK(false, "No implemented es_sm100_mxfp8_blockscaled_grouped_mm for current device");
#endif
}
