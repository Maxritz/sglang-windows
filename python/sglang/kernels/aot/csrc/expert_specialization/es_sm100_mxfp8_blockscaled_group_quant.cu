#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>

#include "es_sm100_mxfp8_blockscaled_group_quant_hip.cuh"

#if !defined(SGLANG_RDNA4)
#include "es_sm100_mxfp8_blockscaled_group_quant.cuh"
#endif

namespace expert_specialization {
namespace native {

template <typename TIn>
void es_sm100_mxfp8_blockscaled_grouped_quant_native(
    const torch::Tensor& input,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& blockscale_offsets,
    torch::Tensor& quant_output,
    torch::Tensor& scale_factor,
    cudaStream_t stream) {
  int num_experts = (int)problem_sizes.size(0);
  auto int32_opts = torch::TensorOptions().dtype(torch::kInt32).device(input.device());
  torch::Tensor tile_base = torch::empty(num_experts + 1, int32_opts);
  torch::Tensor m_tiles = torch::empty(num_experts, int32_opts);

  es_mxfp8_quant_precompute<<<1, num_experts, 0, stream>>>(
      problem_sizes.data_ptr<int>(), num_experts, tile_base.data_ptr<int>(), m_tiles.data_ptr<int>());

  int total_tiles = tile_base.cpu().data_ptr<int>()[num_experts];
  if (total_tiles > 0) {
    es_mxfp8_blockscaled_quant<TIn><<<total_tiles, kQuantTileM, 0, stream>>>(
        reinterpret_cast<const TIn*>(input.data_ptr()),
        problem_sizes.data_ptr<int>(),
        expert_offsets.data_ptr<int>(),
        blockscale_offsets.data_ptr<int>(),
        reinterpret_cast<uint8_t*>(quant_output.data_ptr()),
        reinterpret_cast<uint8_t*>(scale_factor.data_ptr()),
        tile_base.data_ptr<int>(),
        m_tiles.data_ptr<int>(),
        num_experts);
  }
}

}  // namespace native
}  // namespace expert_specialization

void es_sm100_mxfp8_blockscaled_grouped_quant(
    const torch::Tensor& input,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& blockscale_offsets,
    torch::Tensor& quant_output,
    torch::Tensor& scale_factor) {
  TORCH_CHECK(input.dim() == 2, "input must be 2D tensor");
  TORCH_CHECK(input.size(1) % 128 == 0, "k must align to 128");
  TORCH_CHECK(input.strides()[1] == 1, "input must be row major");
  TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");

  auto groups = problem_sizes.size(0);
  TORCH_CHECK(
      expert_offsets.dim() == 1 && expert_offsets.size(0) == groups,
      "expert_offsets must be 1D and have size equal to the number of groups");
  TORCH_CHECK(
      blockscale_offsets.dim() == 1 && blockscale_offsets.size(0) == groups,
      "blockscale_offsets must be 1D and have size equal to the number of groups");

  auto stream = at::cuda::getCurrentCUDAStream();
#if defined(SGLANG_RDNA4)
  if (input.dtype() == torch::kBFloat16) {
    expert_specialization::native::es_sm100_mxfp8_blockscaled_grouped_quant_native<at::BFloat16>(
        input, problem_sizes, expert_offsets, blockscale_offsets, quant_output, scale_factor, stream);
  } else if (input.dtype() == torch::kFloat16) {
    expert_specialization::native::es_sm100_mxfp8_blockscaled_grouped_quant_native<at::Half>(
        input, problem_sizes, expert_offsets, blockscale_offsets, quant_output, scale_factor, stream);
  } else {
    TORCH_CHECK(false, "dtype must be kFloat16 or kBFloat16");
  }
#elif defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  if (input.dtype() == torch::kBFloat16) {
    expert_specialization::launch_es_sm100_mxfp8_blockscaled_grouped_quant<__nv_bfloat16>(
        input, problem_sizes, expert_offsets, blockscale_offsets, quant_output, scale_factor);
  } else if (input.dtype() == torch::kFloat16) {
    expert_specialization::launch_es_sm100_mxfp8_blockscaled_grouped_quant<__half>(
        input, problem_sizes, expert_offsets, blockscale_offsets, quant_output, scale_factor);
  } else {
    TORCH_CHECK(false, "dtype must be kFloat16 or kBFloat16");
  }
#else
  TORCH_CHECK(false, "No implemented es_sm100_mxfp8_blockscaled_grouped_quant for current device");
#endif
}
