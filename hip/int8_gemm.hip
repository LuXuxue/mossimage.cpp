// The INT8 ConvRot GEMM for the high tier: y [m, n] = bf16((x @ w^T) * xs * ws + bias), x [m, k] and w [n, k] int8,
// one per-row scale each. A single 16x16 wave tile; the WMMA int8->int32 leaves are int8_leaf.h.
#if defined(ROCWMMA_I8_LEAF)

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>

#include "device_math.h"
#include "hip_block.h"
#include "int8_leaf.h"

namespace hip_block {

using detail::bf;
using detail::f;
using detail::rbf;

constexpr int kWave = 32;

struct Int8GemmArgs {
  const std::int8_t* x;
  std::int64_t ldx;
  const float* x_scale;
  const std::int8_t* w;
  const float* w_scale;
  const float* bias;
  bf16_bits* y;
  std::int64_t ldy;
  int m, n, k;
};

__global__ __launch_bounds__(kWave) void int8_gemm_kernel(const Int8GemmArgs a, int x0 = 0) {
  const int m0 = (static_cast<int>(blockIdx.x) + x0) * 16;
  const int n0 = static_cast<int>(blockIdx.y) * 16;
  const int row = i8leaf::row_of_thread();
  const int col = i8leaf::col_of_thread();

  // mma_sync(c, a, b, c) accumulates into c: c must be initialized before the walk. A "zero A" MMA would still read
  // the uninitialized c, so fill it explicitly.
  i8leaf::FragAcc acc;
  i8leaf::clear(acc);

  for (int k0 = 0; k0 < a.k; k0 += i8leaf::kK) {
    i8leaf::FragA fa;
    i8leaf::FragB fb;
    i8leaf::load_a(fa, a.x + static_cast<std::int64_t>(m0) * a.ldx + k0, a.ldx);
    i8leaf::load_b(fb, a.w + static_cast<std::int64_t>(n0) * a.k + k0, a.k);
    i8leaf::mma(acc, fa, fb);
  }

  // Store the 16x16 int32 tile to a shared buffer, then dequantize the lane's owned elements.
  alignas(16) __shared__ int s_acc[16 * 16];
  i8leaf::store_acc(s_acc, acc, 16);
  i8leaf::barrier();

  const int r = m0 + row;
  if (r < a.m) {
    const float xs = a.x_scale[r];
    // Each lane owns eight contiguous columns of the 16x16 tile (see i8leaf::col_of_thread): columns col .. col + 7.
    // store_acc wrote the whole tile row-major into s_acc, so all sixteen columns must be covered by the wave's
    // two column halves, without overlap or gaps.
    for (int i = 0; i < 8; ++i) {
      const int c = n0 + col + i;
      if (c >= n0 + 16 || c >= a.n) break;
      const float ws = a.w_scale[c];
      float v = static_cast<float>(s_acc[row * 16 + (c - n0)]) * xs * ws;
      if (a.bias != nullptr) v += a.bias[c];
      a.y[static_cast<std::int64_t>(r) * a.ldy + c] = bf(v);
    }
  }
}

}  // namespace hip_block

namespace mi::hip {

// Launch the INT8 ConvRot GEMM: y [m, n] = bf16((x @ w^T) * xs * ws + bias). x [m, k] int8, w [n, k] int8.
inline void int8_gemm(hip_block::bf16_bits* y, std::int64_t ldy, const std::int8_t* x, std::int64_t ldx,
                      const float* x_scale, const std::int8_t* w, const float* w_scale, const float* bias, int m, int n,
                      int k, hipStream_t stream) {
  hip_block::Int8GemmArgs a{x, ldx, x_scale, w, w_scale, bias, y, ldy, m, n, k};
  const int gx = (m + 15) / 16, gy = (n + 15) / 16;
  // Chunk the grid's x dimension: a 4.2 MP image's token count would otherwise exceed the device's grid limit.
  constexpr int kChunkX = 4096;
  for (int x0 = 0; x0 < gx; x0 += kChunkX) {
    const int xn = std::min(kChunkX, gx - x0);
    hip_block::int8_gemm_kernel<<<dim3(xn, gy, 1), hip_block::kWave, 0, stream>>>(a, x0);
  }
}

}  // namespace mi::hip

#endif
