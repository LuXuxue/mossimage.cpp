// The INT8 leaf of the high tier's ConvRot GEMM: a 16x16x16 int8 WMMA with an int32 accumulator, per wave.
//
//   gfx11 (RDNA3)  __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32, raw intrinsics
//
// No rocWMMA: its fragment and accumulator layouts are opaque, and its load_matrix_sync/store_matrix_sync helpers
// force a particular LDS layout that prevents the multi-warp, software-pipelined tiled kernel. The raw gfx11 layout is
// public and simple: a 16-byte K-step fragment is held whole by every lane (row = lane % 16), and accumulator element e
// is row 2*e + lane/16, column lane%16. This matches the ROCm-optimised ComfyUI kernels on gfx1103.
//
// Every call is collective: all 32 threads of the wave make it, with the same (thread-independent) arguments and under
// the same control flow. p may point to global or __shared__ memory. The kernels depend on lockstep: they reuse a
// __shared__ buffer right after a collective read of it, with a barrier before the next write's readers.
//
//   load_a       A(m, k) = p[m * ld + k]
//   load_b       B(k, n) = p[n * ld + k]   (a [N, K] row-major weight, one column per row of memory)
//   mma          c += a * b
//   store_acc    p[m * ld + n] = c(m, n)   (row-major 16x16 tile into a shared int buffer)
#pragma once

#if defined(ROCWMMA_I8_LEAF)

#include <cstdint>

namespace hip_block::i8leaf {

using v4i = int __attribute__((ext_vector_type(4)));
using v8i = int __attribute__((ext_vector_type(8)));

inline constexpr const char* kName = "gfx11 native v_wmma_i32_16x16x16_iu8";
inline constexpr int kK = 16;

using FragA = v4i;
using FragB = v4i;
using FragAcc = v8i;

// Epilogue addressing used by the single-wave GEMM kernel: a lane owns row lane/2 and two 8-column groups.
__device__ __forceinline__ int row_of_thread() { return static_cast<int>(threadIdx.x) / 2; }
__device__ __forceinline__ int col_of_thread() { return (static_cast<int>(threadIdx.x) & 1) * 8; }

// One wave per workgroup: a barrier orders the wave's own __shared__ accesses without the cache invalidate __syncthreads
// adds.
__device__ __forceinline__ void barrier() { __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront"); }

__device__ __forceinline__ void clear(FragAcc& c) { c = {0, 0, 0, 0, 0, 0, 0, 0}; }

// One lane reads a whole 16-byte K-step of row lane%16. With a 16-byte-aligned LDS stride this is one ds_read_b128.
__device__ __forceinline__ void load_a(FragA& a, const std::int8_t* p, int ld) {
  a = *reinterpret_cast<const v4i*>(p + (static_cast<int>(threadIdx.x) & 15) * ld);
}
__device__ __forceinline__ void load_b(FragB& b, const std::int8_t* p, int ld) {
  b = *reinterpret_cast<const v4i*>(p + (static_cast<int>(threadIdx.x) & 15) * ld);
}

__device__ __forceinline__ void mma(FragAcc& c, const FragA& a, const FragB& b) {
  c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
}

// gfx11 accumulator element e: row 2*e + lane/16, column lane%16. Write a full 16x16 row-major int tile.
__device__ __forceinline__ void store_acc(int* p, const FragAcc& c, int ld) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    p[(2 * e + (lane >> 4)) * ld + (lane & 15)] = c[e];
  }
}

}  // namespace hip_block::i8leaf

#endif
