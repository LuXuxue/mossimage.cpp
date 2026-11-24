// The INT8 ConvRot GEMM for the high tier: y [m, n] = bf16((x @ w^T) * xs * ws + bias), x [m, k] and w [n, k] int8,
// one per-row scale each. The int8 WMMA int32 leaves are in int8_leaf.h (raw gfx11 intrinsics).
//
// The kernel is the tiled, software-pipelined design used by the ROCm-optimised ComfyUI backend: an LDS-staged 128x128
// tile computed by several warps, with the next K tile's global reads prefetched into registers while the current tile's
// fragments are multiplied, and a two-deep register pipeline over the 16-byte WMMA K-steps. This removes the repeated
// global loads of the earlier one-wave 128x32 tile and is what closes most of the gap to ComfyUI's GEMM throughput.
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

using detail::add;
using detail::bf;
using detail::f;

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
  const bf16_bits* gate;        // optional [n] fp32 gate: y = residual + bf16(y)*gate, after the first bf16 rounding
  const bf16_bits* residual;    // optional [m, ldr], may alias y
  std::int64_t ldr;
};

// Stage a ROWS x BKB byte tile from global into registers, then write it to LDS. One 16-byte chunk per lane per pass;
// out-of-range rows and the K tail are zero-filled so the WMMA reads never leave the tile.
template <int ROWS, int BKB, int THREADS>
struct TileStager {
  static constexpr int kPerRow = BKB / 16;
  static constexpr int kChunks = ROWS * kPerRow;
  static constexpr int kPerThread = kChunks / THREADS;
  static constexpr int kStride = BKB + 16;  // 16-byte pad keeps every LDS row aligned for one ds_read_b128
  static_assert(kChunks % THREADS == 0, "THREADS must divide the tile's 16-byte chunks");

  uint4 regs[kPerThread];

  __device__ __forceinline__ void load(const std::int8_t* src, int row0, int rows_total, int kbyte0,
                                       int kbytes) {
    const int tid = static_cast<int>(threadIdx.x);
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) {
      const int c = tid * kPerThread + i;
      const int row = row0 + c / kPerRow;
      const int kbyte = kbyte0 + (c % kPerRow) * 16;
      regs[i] = (row < rows_total && kbyte < kbytes)
                    ? *reinterpret_cast<const uint4*>(src + static_cast<std::int64_t>(row) * kbytes + kbyte)
                    : make_uint4(0, 0, 0, 0);
    }
  }

  __device__ __forceinline__ void store(std::int8_t* lds) const {
    const int tid = static_cast<int>(threadIdx.x);
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) {
      const int c = tid * kPerThread + i;
      *reinterpret_cast<uint4*>(lds + (c / kPerRow) * kStride + (c % kPerRow) * 16) = regs[i];
    }
  }
};

// C[m, n] = epilogue(A[m, k] @ B[n, k]^T), B in its natural [N, K] row-major form. WARPS_M*WARPS_N waves of 32 threads
// tile a BM x BN output tile; each wave owns TM*TN 16x16 WMMA tiles. K walks in BKB-byte stages, each holding kSteps
// 16-byte WMMA K-steps. The next stage's global loads and the next K-step's fragments are issued ahead of the MMAs.
template <int BM, int BN, int BKB, int WARPS_M, int WARPS_N, int TM, int TN>
__global__ __launch_bounds__(WARPS_M* WARPS_N* kWave) void int8_gemm_tiled(
    const Int8GemmArgs a, int x_blocks) {
  constexpr int kThreads = WARPS_M * WARPS_N * kWave;
  constexpr int kStride = BKB + 16;
  constexpr int kSteps = BKB / 16;
  static_assert(BM == WARPS_M * TM * 16 && BN == WARPS_N * TN * 16, "warp grid must tile the block");
  static_assert(TileStager<BM, BKB, kThreads>::kStride == kStride, "staging stride must match read stride");

  __shared__ __align__(16) std::int8_t As[BM * kStride];
  __shared__ __align__(16) std::int8_t Bs[BN * kStride];

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wm = warp / WARPS_N, wn = warp % WARPS_N;
  const int frag_r = lane & 15;  // gfx11: every lane holds the whole K-step of row frag_r

  // Grouped block order for L2 locality: consecutive blocks advance along M within a group of kGroupM block-rows,
  // so concurrently resident blocks share the same B columns. Grid x indexes N, y indexes M, as in the reference.
  constexpr int kGroupM = 4;
  const int blocks_n = gridDim.x;
  const int blocks_m = gridDim.y;
  const int bid = blockIdx.y * blocks_n + blockIdx.x;
  const int per_group = kGroupM * blocks_n;
  const int group = bid / per_group;
  const int idx = bid - group * per_group;
  const int group_rows = std::min(kGroupM, blocks_m - group * kGroupM);
  const int mb = group * kGroupM + idx % group_rows;
  const int nb = idx / group_rows;
  const int m0 = mb * BM, n0 = nb * BN;

  i8leaf::FragAcc acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) i8leaf::clear(acc[i][j]);
  }

  TileStager<BM, BKB, kThreads> sa, sb;
  sa.load(a.x, m0, a.m, 0, a.k);
  sb.load(a.w, n0, a.n, 0, a.k);
  sa.store(As);
  sb.store(Bs);
  __syncthreads();

  for (int k0 = 0; k0 < a.k; k0 += BKB) {
    const int knext = k0 + BKB;
    const bool has_next = knext < a.k;
    if (has_next) {
      sa.load(a.x, m0, a.m, knext, a.k);
      sb.load(a.w, n0, a.n, knext, a.k);
    }

    i8leaf::FragA af[2][TM];
    i8leaf::FragB bf[2][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
      af[0][i] = *reinterpret_cast<const i8leaf::v4i*>(As + (wm * (TM * 16) + i * 16 + frag_r) * kStride);
#pragma unroll
    for (int j = 0; j < TN; ++j)
      bf[0][j] = *reinterpret_cast<const i8leaf::v4i*>(Bs + (wn * (TN * 16) + j * 16 + frag_r) * kStride);

#pragma unroll
    for (int kk = 0; kk < kSteps; ++kk) {
      const int cur = kk & 1, nxt = cur ^ 1;
      if (kk + 1 < kSteps) {
        const int kb = (kk + 1) * 16;
#pragma unroll
        for (int i = 0; i < TM; ++i)
          af[nxt][i] = *reinterpret_cast<const i8leaf::v4i*>(As + (wm * (TM * 16) + i * 16 + frag_r) * kStride + kb);
#pragma unroll
        for (int j = 0; j < TN; ++j)
          bf[nxt][j] = *reinterpret_cast<const i8leaf::v4i*>(Bs + (wn * (TN * 16) + j * 16 + frag_r) * kStride + kb);
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) i8leaf::mma(acc[i][j], af[cur][i], bf[cur][j]);
      }
    }

    if (has_next) {
      __syncthreads();  // all warps finished reading the current LDS tile
      sa.store(As);
      sb.store(Bs);
      __syncthreads();
    }
  }

  // Each accumulator element e is output row 2*e + lane/16, column lane%16 on gfx11.
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int r = m0 + wm * (TM * 16) + i * 16 + 2 * e + (lane >> 4);
      if (r >= a.m) continue;
      const float xs = a.x_scale[r];
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int c = n0 + wn * (TN * 16) + j * 16 + (lane & 15);
        if (c >= a.n) continue;
        float v = static_cast<float>(acc[i][j][e]) * xs * a.w_scale[c];
        if (a.bias != nullptr) v += a.bias[c];
        if (a.lora_delta != nullptr) v += f(a.lora_delta[static_cast<std::int64_t>(r) * a.ldl + c]);
        const bf16_bits g = bf(v);  // the GEMM's own bf16 rounding, before the optional gate+residual epilogue
        const std::size_t at = static_cast<std::size_t>(r) * a.ldy + c;
        if (a.gate != nullptr) {
          // bf(x + bf(gemm_out)*gate): read the residual before storing to it (it may alias y).
          const float prod = f(g) * f(a.gate[c]);
          const float res = a.residual != nullptr ? f(a.residual[static_cast<std::int64_t>(r) * a.ldr + c]) : 0.0f;
          a.y[at] = bf(add(res, prod));
        } else {
          a.y[at] = g;
        }
      }
    }
  }
}

}  // namespace hip_block

namespace mi::hip {

// Launch the INT8 ConvRot GEMM: y [m, n] = bf16((x @ w^T) * xs * ws + bias). x [m, k] int8, w [n, k] int8.
void int8_gemm(hip_block::bf16_bits* y, std::int64_t ldy, const std::int8_t* x, std::int64_t ldx,
                       const float* x_scale, const std::int8_t* w, const float* w_scale, const float* bias, int m, int n,
                       int k, hipStream_t stream, const hip_block::bf16_bits* lora_delta, std::int64_t ldl,
                       const hip_block::bf16_bits* gate, const hip_block::bf16_bits* residual, std::int64_t ldr) {
  if (m == 0 || n == 0 || k == 0) return;
  hip_block::Int8GemmArgs a{x, ldx, x_scale, w, w_scale, bias, y, ldy, m, n, k, lora_delta, ldl,
                            gate, residual, ldr};
  // Chunk the grid's y dimension so a multi-megapixel image's token count cannot exceed the device's grid limit.
  constexpr int kChunk = 4096;
  const bool deep = k >= 4096;
  constexpr int bm = 128, bn = 128;
  const int gx = (m + bm - 1) / bm;
  const int gy = (n + bn - 1) / bn;
  for (int x0 = 0; x0 < gx; x0 += kChunk) {
    const int xn = std::min(kChunk, gx - x0);
    dim3 grid(gy, xn, 1);  // x = N blocks, y = M blocks (matches the kernel's grouped ordering)
    // Deep K: the 128-thread 4x4 fragment grid has the highest matrix density and the LDS reads are amortised. Shallow
    // K: the 512-thread 2x2 grid yields more blocks to interleave across the GPU's WGPs.
    if (deep) {
      hip_block::int8_gemm_tiled<128, 128, 128, 2, 2, 4, 4><<<grid, 128, 0, stream>>>(a, xn);
    } else {
      hip_block::int8_gemm_tiled<128, 128, 64, 4, 4, 2, 2><<<grid, 512, 0, stream>>>(a, xn);
    }
  }
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) throw std::runtime_error(std::string("hip_block::int8_gemm: ") + hipGetErrorString(e));
}

}  // namespace mi::hip

#endif
