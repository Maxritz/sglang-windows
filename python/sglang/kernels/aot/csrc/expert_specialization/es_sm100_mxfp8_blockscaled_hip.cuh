#pragma once

// Native gfx12 (RDNA4) implementation of the MXFP8 block-scaled grouped GEMM.
// Replaces the CUTLASS SM100 tcgen05 implementation for gfx1200/gfx1201.
// Verified fragment layouts: FP8 16x16x16 WMMA — A/B consecutive
//   (lane = row/col + 16*k2, byte slot e -> k = k2*8 + e).
// C fragment: row = (lane>>4)*8 + i, col = lane&15.
// Scales are MXFP8 e8m0 (uint8) -> value 2^(b - 127).

#include <hip/hip_runtime.h>

#if defined(SGLANG_RDNA4)

namespace expert_specialization {

namespace native {

constexpr int kTileM = 64;
constexpr int kTileN = 128;
constexpr int kTileK = 32;  // WMMA k=16 x2 within a 32-element scale block

typedef int __attribute__((ext_vector_type(2))) i32x2_t;
typedef float __attribute__((ext_vector_type(8))) f32x8_t;

__device__ __forceinline__ float e8m0_to_float(uint8_t b) {
  return __int_as_float((uint32_t)b << 23);
}

// Build the fp8 WMMA A/B operand (2 dwords, consecutive k layout).
__device__ __forceinline__ i32x2_t pack_fp8_operand(const uint8_t* base, int k2) {
  uint32_t lo = 0, hi = 0;
#pragma unroll
  for (int e = 0; e < 4; e++) lo |= (uint32_t)base[k2 * 8 + e] << (8 * e);
#pragma unroll
  for (int e = 0; e < 4; e++) hi |= (uint32_t)base[k2 * 8 + 4 + e] << (8 * e);
  return {static_cast<int>(lo), static_cast<int>(hi)};
}

__device__ __forceinline__ f32x8_t wmma_fp8_16x16x16(i32x2_t a, i32x2_t b, f32x8_t c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}

// Per-expert: tile counts and prefix sums so the GEMM can locate (expert, tile).
__global__ void es_mxfp8_precompute(
    const int* problem_sizes,
    int num_experts,
    int* tile_base,
    int* tiles_n) {
  __shared__ int counts[1024];
  __shared__ int total;
  int e = threadIdx.x;
  if (e < num_experts) {
    int m = problem_sizes[e * 3 + 0];
    int n = problem_sizes[e * 3 + 1];
    int tm = (m + kTileM - 1) / kTileM;
    int tn = (n + kTileN - 1) / kTileN;
    counts[e] = tm * tn;
    tiles_n[e] = tn;
  }
  __syncthreads();
  int acc = 0;
  for (int i = 0; i < e; i++) acc += counts[i];
  tile_base[e] = acc;
  __syncthreads();
  if (e == 0) {
    int t = 0;
    for (int i = 0; i < num_experts; i++) t += counts[i];
    total = t;
    tile_base[num_experts] = t;
  }
}

template <typename OutT>
__global__ void __launch_bounds__(128) es_mxfp8_blockscaled_mm(
    const uint8_t* a,        // (num_tokens, k) row major
    const uint8_t* b,        // (num_experts, k, n), k fastest (n stride k)
    const uint8_t* sfa,      // (blockscale_offset, k/32) row major
    const uint8_t* sfb,      // (num_experts, k/32, n), k/32 fastest (n stride k/32)
    OutT* d,                 // (num_tokens, n) row major
    const int* problem_sizes,
    const int* expert_offsets,
    const int* blockscale_offsets,
    const int* tile_base,
    const int* tiles_n,
    int num_experts) {
  __shared__ uint8_t a_s[64][32];   // A tile: 64 rows x 32 k
  __shared__ uint8_t b_s[32][128];  // B tile: 32 k x 128 n
  __shared__ uint8_t sfa_s[64];
  __shared__ uint8_t sfb_s[128];

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;
  int row16 = lane & 15;
  int col16 = lane & 15;
  int k2 = lane >> 4;

  // Locate (expert, mtile, ntile) via binary search on tile_base.
  int bid = blockIdx.x;
  int lo = 0, hi = num_experts;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (tile_base[mid] <= bid) lo = mid + 1;
    else hi = mid;
  }
  int e = lo - 1;
  int m = problem_sizes[e * 3 + 0];
  int n = problem_sizes[e * 3 + 1];
  int k = problem_sizes[e * 3 + 2];
  int tile = bid - tile_base[e];
  int mtile = tile / tiles_n[e];
  int ntile = tile % tiles_n[e];
  int m0 = mtile * kTileM;
  int n0 = ntile * kTileN;

  int64_t expert_offset = expert_offsets[e];
  int64_t blockscale_offset = blockscale_offsets[e];
  const uint8_t* a_p = a + expert_offset * k;
  const uint8_t* b_p = b + static_cast<int64_t>(e) * k * n;
  const uint8_t* sfa_p = sfa + blockscale_offset * (k / 32);
  const uint8_t* sfb_p = sfb + static_cast<int64_t>(e) * (k / 32) * n;
  OutT* d_p = d + expert_offset * n;

  int kblk = (k + kTileK - 1) / kTileK;

  float acc[8][8];
#pragma unroll
  for (int nt = 0; nt < 8; nt++)
#pragma unroll
    for (int i = 0; i < 8; i++) acc[nt][i] = 0.0f;

  for (int kb = 0; kb < kblk; kb++) {
    // Stage A tile: 64 rows x 32 bytes.
    {
      int row = tid >> 1;
      int half = tid & 1;
      int mrow = m0 + row;
      if (mrow < m) {
        const uint8_t* src = a_p + static_cast<int64_t>(mrow) * k + kb * kTileK + half * 16;
        uint8_t* dst = &a_s[row][half * 16];
        *reinterpret_cast<uint32_t*>(dst) = *reinterpret_cast<const uint32_t*>(src);
        *reinterpret_cast<uint32_t*>(dst + 4) = *reinterpret_cast<const uint32_t*>(src + 4);
        *reinterpret_cast<uint32_t*>(dst + 8) = *reinterpret_cast<const uint32_t*>(src + 8);
        *reinterpret_cast<uint32_t*>(dst + 12) = *reinterpret_cast<const uint32_t*>(src + 12);
      }
    }
    // Stage B tile: 32 k x 128 n. Global b is k-fastest (n stride k),
    // so a 32-byte run is one k-block for a fixed column.
    {
      int col = tid;
      int ncol = n0 + col;
      const uint8_t* src = b_p + static_cast<int64_t>(ncol) * k + kb * kTileK;
#pragma unroll
      for (int i = 0; i < 32; i++) b_s[i][col] = src[i];
    }
    if (tid < 64) {
      int mrow = m0 + tid;
      sfa_s[tid] = (mrow < m) ? sfa_p[static_cast<int64_t>(mrow) * (k / 32) + kb] : 0;
    }
    if (tid < 128) {
      sfb_s[tid] = sfb_p[static_cast<int64_t>(n0 + tid) * (k / 32) + kb];
    }
    __syncthreads();

    // 8 n-tiles x 2 WMMA k16 steps, per warp (4 warps x 16 rows = 64-row tile).
    for (int nt = 0; nt < 8; nt++) {
      f32x8_t c = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int k16 = 0; k16 < 2; k16++) {
        int rowk = k16 * 16 + k2 * 8;
        i32x2_t af = pack_fp8_operand(&a_s[warp * 16 + row16][k16 * 16], k2);
        i32x2_t bf;
        {
          uint32_t b0 = 0, b1 = 0;
          const uint8_t* bcol = &b_s[0][16 * nt + col16];
#pragma unroll
          for (int e = 0; e < 4; e++) b0 |= (uint32_t)bcol[(rowk + e) * 128] << (8 * e);
#pragma unroll
          for (int e = 0; e < 4; e++) b1 |= (uint32_t)bcol[(rowk + 4 + e) * 128] << (8 * e);
          bf = {static_cast<int>(b0), static_cast<int>(b1)};
        }
        c = wmma_fp8_16x16x16(af, bf, c);
      }
      float sB = e8m0_to_float(sfb_s[16 * nt + col16]);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        float sA = e8m0_to_float(sfa_s[warp * 16 + k2 * 8 + i]);
        acc[nt][i] += c[i] * sA * sB;
      }
    }
    __syncthreads();
  }

  // Epilogue.
  {
    int rbase = m0 + warp * 16 + k2 * 8;
    for (int nt = 0; nt < 8; nt++) {
      int col = n0 + 16 * nt + col16;
      if (col >= n) continue;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        int r = rbase + i;
        if (r < m) {
          d_p[static_cast<int64_t>(r) * n + col] = static_cast<OutT>(acc[nt][i]);
        }
      }
    }
  }
}

}  // namespace native

}  // namespace expert_specialization

#endif  // SGLANG_RDNA4
