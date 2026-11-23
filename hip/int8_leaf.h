// The INT8 leaf of the high tier's ConvRot GEMM: a 16x16x32 int8 multiply with an int32 accumulator, per wave.
//
//   ROCWMMA_I8_LEAF (AMD gfx11xx)  rocWMMA fragments, v_wmma_i8_16x16x32_i8
//
// The int8 counterpart of mma_leaf.h's bf16 leaf (16x16x16). The K step is 32, the int8 WMMA's native depth, so the
// INT8 GEMM walk advances K by 32, not 16. rocWMMA's fragment lane layouts are private and differ between RDNA3 and
// RDNA4, so nothing outside this file touches a fragment's elements: an accumulator's values reach the kernel only
// through store_acc, into memory.
//
// Every call is collective: all 32 threads of the wave make it, with the same (thread-independent) arguments and under
// the same control flow. p may point to global or __shared__ memory. The kernels depend on lockstep: they reuse a
// __shared__ buffer right after a collective read of it, with a barrier before the next write's readers.
//
//   load_a       A(m, k) = p[m * ld + k]
//   load_b       B(k, n) = p[n * ld + k]   (a [N, K] row-major weight, one column per row of memory)
//   mma          c += a * b
//   store_acc    p[m * ld + n] = c(m, n)
//   load_acc     c(m, n) = p[m * ld + n]
#pragma once

#if defined(ROCWMMA_I8_LEAF)

#include <rocwmma/rocwmma.hpp>

#include <cstdint>

namespace hip_block::i8leaf {

inline constexpr const char* kName = "rocWMMA v_wmma_i8_16x16x32_i8";
inline constexpr int kK = 32;  // the int8 WMMA's native depth

using FragA = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 32, rocwmma::int8_t, rocwmma::row_major>;
using FragB = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 32, rocwmma::int8_t, rocwmma::col_major>;
using FragAcc = rocwmma::fragment<rocwmma::accumulator, 16, 16, 32, int>;

// The 16x16 output tile: lane t owns row t / 2 and columns (t & 1) * 8 .. + 8, as the bf16 leaf's dequant.
__device__ __forceinline__ int row_of_thread() { return static_cast<int>(threadIdx.x) / 2; }
__device__ __forceinline__ int col_of_thread() { return (static_cast<int>(threadIdx.x) & 1) * 8; }

// One wave per workgroup: a barrier orders the wave's own __shared__ accesses without the cache invalidate __syncthreads
// adds.
__device__ __forceinline__ void barrier() { __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront"); }

__device__ __forceinline__ void clear(FragAcc& c) { rocwmma::fill_fragment(c, 0); }
__device__ __forceinline__ void load_a(FragA& a, const std::int8_t* p, int ld) {
  rocwmma::load_matrix_sync(a, p, static_cast<std::uint32_t>(ld));
}
__device__ __forceinline__ void load_b(FragB& b, const std::int8_t* p, int ld) {
  rocwmma::load_matrix_sync(b, p, static_cast<std::uint32_t>(ld));
}
__device__ __forceinline__ void mma(FragAcc& c, const FragA& a, const FragB& b) { rocwmma::mma_sync(c, a, b, c); }
__device__ __forceinline__ void store_acc(int* p, const FragAcc& c, int ld) {
  rocwmma::store_matrix_sync(p, c, static_cast<std::uint32_t>(ld), rocwmma::mem_row_major);
}
__device__ __forceinline__ void load_acc(FragAcc& c, const int* p, int ld) {
  rocwmma::load_matrix_sync(c, p, static_cast<std::uint32_t>(ld), rocwmma::mem_row_major);
}

}  // namespace hip_block::i8leaf

#endif
