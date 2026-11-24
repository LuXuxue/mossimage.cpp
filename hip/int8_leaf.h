// The INT8 leaf of the high tier's ConvRot GEMM: a 16x16x16 int8 WMMA with an int32 accumulator, per wave.
//
//   gfx11 (RDNA3)  __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32, raw intrinsics
//
// No rocWMMA: its fragment and accumulator layouts are opaque, and its load_matrix_sync/store_matrix_sync helpers
// force a particular LDS layout that prevents the multi-warp, software-pipelined tiled kernel. The raw gfx11 layout is
// public and simple: a 16-byte K-step fragment is held whole by every lane (row = lane % 16), and accumulator element e
// is row 2*e + lane/16, column lane%16. This matches the ROCm-optimised ComfyUI kernels on gfx1103.
//
// The tiled GEMM stages operands through LDS and writes its epilogue itself, so the only leaf operations it needs are
// `clear` and `mma`. Every call is collective: all 32 threads of the wave make it with the same (thread-independent)
// arguments and under the same control flow. The kernels depend on lockstep: they reuse a __shared__ buffer right after
// a collective read of it, with a barrier before the next write's readers.
#pragma once

#if defined(ROCWMMA_I8_LEAF)

#include <cstdint>

namespace hip_block::i8leaf {

using v4i = int __attribute__((ext_vector_type(4)));
using v8i = int __attribute__((ext_vector_type(8)));

using FragA = v4i;
using FragB = v4i;
using FragAcc = v8i;

__device__ __forceinline__ void clear(FragAcc& c) { c = {0, 0, 0, 0, 0, 0, 0, 0}; }

__device__ __forceinline__ void mma(FragAcc& c, const FragA& a, const FragB& b) {
  c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
}

}  // namespace hip_block::i8leaf

#endif
