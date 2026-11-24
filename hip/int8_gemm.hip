// The INT8 ConvRot GEMM for the high tier: y [m, n] = bf16((x @ w^T) * xs * ws + bias), x [m, k] and w [n, k] int8,
// one per-row scale each. A single 16x16 wave tile; the WMMA int8->int32 leaves are int8_leaf.h.
#if defined(ROCWMMA_I8_LEAF)

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "device_math.h"
#include "hip_block.h"
#include "int8_gemm.h"
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
  const bf16_bits* lora_delta;  // optional [m, n] bf16 low-rank term, added before the bf16 rounding
  std::int64_t ldl;
};

// One workgroup computes a 64x32 output tile: four A fragments and two B fragments produce eight int32 accumulators.
// The weight and activation fragments are therefore reused four times / twice respectively instead of being reloaded
// for every 16x16 output tile.
constexpr int kTileM = 64;
constexpr int kTileN = 32;
constexpr int kMf = kTileM / 16, kNf = kTileN / 16;

__global__ __launch_bounds__(kWave) void int8_gemm_kernel(const Int8GemmArgs a, int x0 = 0) {
  const int m0 = (static_cast<int>(blockIdx.x) + x0) * kTileM;
  const int n0 = static_cast<int>(blockIdx.y) * kTileN;
  const int row = i8leaf::row_of_thread();
  const int col = i8leaf::col_of_thread();

  i8leaf::FragAcc acc[kMf][kNf];
#pragma unroll
  for (int mf = 0; mf < kMf; ++mf) {
#pragma unroll
    for (int nf = 0; nf < kNf; ++nf) i8leaf::clear(acc[mf][nf]);
  }

  // A partial M tile is staged through shared memory with zero rows: load_a fetches 16 whole rows, so a direct load
  // would read past the end of x and fault asynchronously at large token counts.
  alignas(16) __shared__ std::int8_t s_a[kTileM * 16];
  const bool partial_m = m0 + kTileM > a.m;
  for (int k0 = 0; k0 < a.k; k0 += i8leaf::kK) {
    i8leaf::FragA fa[kMf];
    i8leaf::FragB fb[kNf];
    if (partial_m) {
      for (int i = threadIdx.x; i < kTileM * 16; i += kWave) {
        const int rr = i / 16, cc = i % 16;
        const int r = m0 + rr;
        s_a[i] = r < a.m ? a.x[static_cast<std::int64_t>(r) * a.ldx + k0 + cc] : 0;
      }
      __syncthreads();
#pragma unroll
      for (int mf = 0; mf < kMf; ++mf) i8leaf::load_a(fa[mf], s_a + mf * 16 * 16, 16);
      __syncthreads();
    } else {
#pragma unroll
      for (int mf = 0; mf < kMf; ++mf)
        i8leaf::load_a(fa[mf], a.x + static_cast<std::int64_t>(m0 + mf * 16) * a.ldx + k0, a.ldx);
    }
#pragma unroll
    for (int nf = 0; nf < kNf; ++nf)
      i8leaf::load_b(fb[nf], a.w + static_cast<std::int64_t>(n0 + nf * 16) * a.k + k0, a.k);
#pragma unroll
    for (int mf = 0; mf < kMf; ++mf) {
#pragma unroll
      for (int nf = 0; nf < kNf; ++nf) i8leaf::mma(acc[mf][nf], fa[mf], fb[nf]);
    }
  }

  alignas(16) __shared__ int s_acc[kTileM * kTileN];
#pragma unroll
  for (int mf = 0; mf < kMf; ++mf) {
#pragma unroll
    for (int nf = 0; nf < kNf; ++nf) {
      i8leaf::store_acc(s_acc + mf * 16 * kTileN + nf * 16, acc[mf][nf], kTileN);
    }
  }
  i8leaf::barrier();

#pragma unroll
  for (int mf = 0; mf < kMf; ++mf) {
    const int r = m0 + mf * 16 + row;
    if (r >= a.m) continue;
    const float xs = a.x_scale[r];
#pragma unroll
    for (int nf = 0; nf < kNf; ++nf) {
      for (int i = 0; i < 8; ++i) {
        const int c = n0 + nf * 16 + col + i;
        if (c >= n0 + kTileN || c >= a.n) break;
        const float ws = a.w_scale[c];
        float v = static_cast<float>(s_acc[(mf * 16 + row) * kTileN + (c - n0)]) * xs * ws;
        if (a.bias != nullptr) v += a.bias[c];
        if (a.lora_delta != nullptr) v += f(a.lora_delta[r * a.ldl + c]);
        a.y[static_cast<std::int64_t>(r) * a.ldy + c] = bf(v);
      }
    }
  }
}

}  // namespace hip_block

namespace mi::hip {

// Launch the INT8 ConvRot GEMM: y [m, n] = bf16((x @ w^T) * xs * ws + bias). x [m, k] int8, w [n, k] int8.
void int8_gemm(hip_block::bf16_bits* y, std::int64_t ldy, const std::int8_t* x, std::int64_t ldx,
                      const float* x_scale, const std::int8_t* w, const float* w_scale, const float* bias, int m, int n,
                      int k, hipStream_t stream, const hip_block::bf16_bits* lora_delta, std::int64_t ldl) {
  hip_block::Int8GemmArgs a{x, ldx, x_scale, w, w_scale, bias, y, ldy, m, n, k, lora_delta, ldl};
  const int gx = (m + hip_block::kTileM - 1) / hip_block::kTileM;
  const int gy = (n + hip_block::kTileN - 1) / hip_block::kTileN;
  // Chunk the grid's x dimension: a 4.2 MP image's token count would otherwise exceed the device's grid limit.
  constexpr int kChunkX = 4096;
  for (int x0 = 0; x0 < gx; x0 += kChunkX) {
    const int xn = std::min(kChunkX, gx - x0);
    hip_block::int8_gemm_kernel<<<dim3(xn, gy, 1), hip_block::kWave, 0, stream>>>(a, x0);
  }
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) throw std::runtime_error(std::string("hip_block::int8_gemm: ") + hipGetErrorString(e));
}

}  // namespace mi::hip

#endif
