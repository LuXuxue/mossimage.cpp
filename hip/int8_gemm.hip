// The INT8 ConvRot GEMM for the high tier: y [m, n] = bf16((x @ w^T) * xs * ws + bias), x [m, k] and w [n, k] int8,
// one per-row scale each. A single 16x16 wave tile; the WMMA int8->int32 leaves are int8_leaf.h.
#if defined(ROCWMMA_I8_LEAF)

#include <hip/hip_runtime.h>

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

  // Zero the accumulator: build it as the mma of a zero A fragment with any B (a zero A contributes nothing).
  i8leaf::FragA zero_a;
  i8leaf::FragB zero_b;
  {
    // A zero tile in local memory, identical across the wave; load_a is collective over the same address.
    alignas(16) __shared__ int8_t s_zero[16 * 32];
    for (int i = 0; i < 16 * 32 / kWave; ++i) s_zero[i * kWave + static_cast<int>(threadIdx.x)] = 0;
    i8leaf::barrier();
    i8leaf::load_a(zero_a, s_zero, 32);
    i8leaf::load_b(zero_b, s_zero, 32);
  }
  i8leaf::FragAcc acc;
  i8leaf::mma(acc, zero_a, zero_b);  // acc = 0
  i8leaf::barrier();

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
    for (int c = n0 + col; c < n0 + 16 && c < a.n; c += 4) {
      const float ws = a.w_scale[c];
      float v = static_cast<float>(s_acc[row * 16 + (c - n0)]) * xs * ws;
      if (a.bias != nullptr) v += a.bias[c];
      a.y[static_cast<std::int64_t>(r) * a.ldy + c] = rbf(v);
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
  std::fprintf(stderr, "[i8gemm-init] m=%d n=%d gx=%d gy=%d\n", m, n, gx, gy);
  constexpr int kChunkX = 64;  // 780M iGPU grid limit
  for (int x0 = 0; x0 < gx; x0 += kChunkX) {
    const int xn = std::min(kChunkX, gx - x0);
    hip_block::int8_gemm_kernel<<<dim3(xn, gy, 1), hip_block::kWave, 0, stream>>>(a, x0);
    std::fprintf(stderr, "[i8gemm] chunk x0=%d xn=%d gy=%d\n", x0, xn, gy);
  }
}

}  // namespace mi::hip

#endif
