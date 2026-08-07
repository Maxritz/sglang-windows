#pragma once

// Native gfx12 (RDNA4) implementation of fp8_scaled_mm (row/column-wise scaled
// FP8 x FP8 -> FP32 GEMM with optional bias). Replaces the CUTLASS
// sm90/sm100/sm120 rowwise-scaled GEMM for gfx1200/gfx1201.
//
// Math: D[m,n] = (sum_k A[m,k] * B[k,n]) * scale_a[m] * scale_b[n] (+ bias[n])
//   A: fp8 e4m3 (M,K) row-major (K-contiguous), B: fp8 e4m3 (N,K) stored
//   K-contiguous (offset = k*ldb + n; torch transposes the (N,K) weight).
//
// Uses the hardware-verified gfx1201 WMMA layout (see
// reference/gfx1201-wmma-fragments.md): A/B operands consecutive
// (lane = row/col + 16*k2, byte e -> k = k2*8 + e), C fragment
// row = (lane>>4)*8 + i, col = lane&15. CTA 64x128, 4 warps, k-block 32.

#include <hip/hip_runtime.h>

#if defined(SGLANG_RDNA4)

namespace sglang_kernel_rdna4 {

constexpr int kF8TileM = 64;
constexpr int kF8TileN = 128;
constexpr int kF8TileK = 32;

typedef int __attribute__((ext_vector_type(2))) f8_i32x2_t;
typedef float __attribute__((ext_vector_type(8))) f8_f32x8_t;

__device__ __forceinline__ f8_i32x2_t fp8_pack_operand(const uint8_t* base, int k2) {
  uint32_t lo = 0, hi = 0;
#pragma unroll
  for (int e = 0; e < 4; e++) lo |= (uint32_t)base[k2 * 8 + e] << (8 * e);
#pragma unroll
  for (int e = 0; e < 4; e++) hi |= (uint32_t)base[k2 * 8 + 4 + e] << (8 * e);
  return {static_cast<int>(lo), static_cast<int>(hi)};
}

__device__ __forceinline__ f8_f32x8_t fp8_wmma(f8_i32x2_t a, f8_i32x2_t b, f8_f32x8_t c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}

// Stage a byte-run for one row/column, zero-padding beyond K.
// Byte-wise so it is safe for unaligned src (b offset kbase*ldb+ncol with odd
// ldb) and unaligned dst (b_s column base). `nbytes` is the slot width: 16 for
// A half-rows (a_s[64][32] split into two 16-byte halves), 32 for B k-runs.
__device__ __forceinline__ void stage_run(uint8_t* dst, const uint8_t* src, int valid, int nbytes) {
#pragma unroll
  for (int i = 0; i < nbytes; i++) dst[i] = (i < valid) ? src[i] : 0;
}

template <typename OutT>
__global__ void __launch_bounds__(128) fp8_scaled_mm_kernel(
    const uint8_t* a,        // (M, K) row major, K-contiguous
    const uint8_t* b,        // (N, K) stored K-contiguous: offset = k*ldb + n
    int ldb,
    const float* sa,         // M per-row scales (or 1 scalar)
    int scalar_a,
    const float* sb,         // N per-column scales
    const OutT* bias,        // N per-column bias (optional)
    OutT* d,                 // (M, N) row major
    int M, int N, int K) {
  __shared__ uint8_t a_s[64][32];   // A tile: 64 rows x 32 k
  __shared__ uint8_t b_s[32][128];  // B tile: 32 k x 128 n

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;  // 4 warps x 16 rows = 64-row tile
  int row16 = lane & 15;
  int col16 = lane & 15;
  int k2 = lane >> 4;

  int m0 = blockIdx.y * kF8TileM;
  int n0 = blockIdx.x * kF8TileN;

  int kblk = (K + kF8TileK - 1) / kF8TileK;

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
      int kbase = kb * kF8TileK + half * 16;
      uint8_t* dst = &a_s[row][half * 16];
      if (mrow < M) {
        int valid = K - kbase;
        if (valid > 16) valid = 16;
        stage_run(dst, a + static_cast<int64_t>(mrow) * K + kbase, valid, 16);
      } else {
        *reinterpret_cast<uint32_t*>(dst) = 0;
        *reinterpret_cast<uint32_t*>(dst + 4) = 0;
        *reinterpret_cast<uint32_t*>(dst + 8) = 0;
        *reinterpret_cast<uint32_t*>(dst + 12) = 0;
      }
    }
    // Stage B tile: 32 k x 128 n. Global b is K-contiguous (n stride ldb),
    // so a 32-byte k-run is one k-block for a fixed column. b_s is [k][n]
    // (k stride 128), so each byte goes to b_s[i][col] strided by 128.
    {
      int col = tid;
      int ncol = n0 + col;
      int kbase = kb * kF8TileK;
      int valid = K - kbase;
      if (valid > 32) valid = 32;
      if (ncol < N) {
        const uint8_t* src = b + static_cast<int64_t>(ncol) * ldb + kbase;
#pragma unroll
        for (int i = 0; i < 32; i++) b_s[i][col] = (i < valid) ? src[i] : 0;
      } else {
#pragma unroll
        for (int i = 0; i < 32; i++) b_s[i][col] = 0;
      }
    }
    __syncthreads();

    // 8 n-tiles x 2 WMMA k16 steps, per warp.
    for (int nt = 0; nt < 8; nt++) {
      f8_f32x8_t c = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int k16 = 0; k16 < 2; k16++) {
        int rowk = k16 * 16 + k2 * 8;
        f8_i32x2_t af = fp8_pack_operand(&a_s[warp * 16 + row16][k16 * 16], k2);
        f8_i32x2_t bf;
        {
          uint32_t b0 = 0, b1 = 0;
          const uint8_t* bcol = &b_s[0][16 * nt + col16];
#pragma unroll
          for (int e = 0; e < 4; e++) b0 |= (uint32_t)bcol[(rowk + e) * 128] << (8 * e);
#pragma unroll
          for (int e = 0; e < 4; e++) b1 |= (uint32_t)bcol[(rowk + 4 + e) * 128] << (8 * e);
          bf = {static_cast<int>(b0), static_cast<int>(b1)};
        }
        c = fp8_wmma(af, bf, c);
      }
#pragma unroll
      for (int i = 0; i < 8; i++) acc[nt][i] += c[i];
    }
    __syncthreads();
  }

  // Epilogue: D = acc * scale_a[row] * scale_b[col] (+ bias[col]).
  {
    int rbase = m0 + warp * 16 + k2 * 8;
    for (int nt = 0; nt < 8; nt++) {
      int col = n0 + 16 * nt + col16;
      if (col >= N) continue;
      float sbv = sb[col];
      float biasv = bias ? static_cast<float>(bias[col]) : 0.0f;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        int r = rbase + i;
        if (r < M) {
          float sav = scalar_a ? sa[0] : sa[r];
          d[static_cast<int64_t>(r) * N + col] =
              static_cast<OutT>(acc[nt][i] * sav * sbv + biasv);
        }
      }
    }
  }
}

}  // namespace sglang_kernel_rdna4

#endif  // SGLANG_RDNA4
