// The LoRA side branch's and text encoder's bf16 GEMM. gfx11/gfx12 use the LDS-staged, software-pipelined 128x128
// tile of native bf16 WMMA (32 bytes of a row per 16-element K-step, LDS row pad 32). RDNA2 has neither matrix cores
// nor bf16 instructions, so there the operands are converted to fp16 on staging and a 4x4-per-thread register tile
// consumes them with V_DOT2_F32_F16, as in comfy-kitchen's gfx1035 fp16 VALU GEMM.
#if defined(ROCWMMA_I8_LEAF)

#include <hip/hip_runtime.h>
#if defined(__GFX10__)
#include <hip/hip_fp16.h>
#endif

#include <cstdint>
#include <stdexcept>
#include <string>

#include "bf16_gemm.h"
#include "device_math.h"
#include "hip_block.h"
#include "runtime.h"

namespace mi::hip {
namespace {

using bf16 = hip_block::bf16_bits;
using v4i = int __attribute__((ext_vector_type(4)));
using v16bf = __bf16 __attribute__((ext_vector_type(16)));
using v8f = float __attribute__((ext_vector_type(8)));

constexpr int kWave = 32;

#if defined(__GFX10__)

// RDNA2 tile: 64x64 output, 256 threads, each owning 4x4 outputs, BK=64 pairs converted from bf16 to fp16 on staging.
// Two LDS buffers are filled before the loop and then handed over with one barrier per K tile. STRIDE has eight spare
// halves so every 16-byte LDS read of eight halves starts 16-byte aligned (ds_read_b128 needs that on gfx10).
__device__ __forceinline__ __half bf16_to_half(bf16 b) {
  const float v = hip_block::detail::f(b);
  return __float2half_rn(fminf(fmaxf(v, -65504.0f), 65504.0f));
}

// Same template signature as the WMMA kernel below: host launchers cannot see __GFX10__ (a device-only macro), so they
// must reference a kernel name that exists in every architecture's code object. The RDNA2 body ignores the template's
// blocking arguments and uses the fixed tile above; the launcher picks the matching grid and thread count at runtime.
template <int, int, int, int, int, int, int>
__global__ __launch_bounds__(256) void gemm_tiled(const Bf16GemmArgs a) {
  constexpr int TM = 4, TN = 4, BM = 64, BN = 64, BK = 64, THREADS = 256;
  constexpr int STRIDE = BK + 8;
  static_assert((BM / TM) * (BN / TN) == THREADS, "the thread tile must cover the block");
  __shared__ __half As[2][BM * STRIDE];
  __shared__ __half Bs[2][BN * STRIDE];

  const int tid = static_cast<int>(threadIdx.x);
  const int trow = tid / (BN / TN), tcol = tid % (BN / TN);
  const int orow = trow * TM, ocol = tcol * TN;
  const int m0 = static_cast<int>(blockIdx.y) * BM, n0 = static_cast<int>(blockIdx.x) * BN;

  auto load_tile = [&](int buf, int k0) {
    for (int i = tid; i < BM * BK; i += THREADS) {
      const int r = i / BK, k = i % BK, gk = k0 + k;
      As[buf][r * STRIDE + k] = (m0 + r < a.m && gk < a.k)
                                    ? bf16_to_half(a.a[static_cast<std::int64_t>(m0 + r) * a.lda + gk])
                                    : __float2half_rn(0.0f);
    }
    for (int i = tid; i < BN * BK; i += THREADS) {
      const int r = i / BK, k = i % BK, gk = k0 + k;
      Bs[buf][r * STRIDE + k] = (n0 + r < a.n && gk < a.k)
                                    ? bf16_to_half(a.b[static_cast<std::int64_t>(n0 + r) * a.ldb + gk])
                                    : __float2half_rn(0.0f);
    }
  };

  float acc[TM][TN] = {};
  load_tile(0, 0);
  if (BK < a.k) load_tile(1, BK);
  __syncthreads();

  using v8h = _Float16 __attribute__((ext_vector_type(8)));
  using v2h = _Float16 __attribute__((ext_vector_type(2)));
  int buf = 0;
  for (int k0 = 0; k0 < a.k; k0 += BK) {
    const int nb = 1 - buf, kn = k0 + BK;
    if (kn < a.k) load_tile(nb, kn);  // the barrier below orders this against this tile's readers
    __syncthreads();

    const __half* ac = &As[buf][0];
    const __half* bc = &Bs[buf][0];
#pragma unroll
    for (int k = 0; k < BK; k += 8) {
      v8h av[TM], bv[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) av[i] = *reinterpret_cast<const v8h*>(&ac[(orow + i) * STRIDE + k]);
#pragma unroll
      for (int j = 0; j < TN; ++j) bv[j] = *reinterpret_cast<const v8h*>(&bc[(ocol + j) * STRIDE + k]);
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            acc[i][j] = __builtin_amdgcn_fdot2(reinterpret_cast<const v2h*>(&av[i])[q],
                                               reinterpret_cast<const v2h*>(&bv[j])[q], acc[i][j], true);
          }
        }
      }
    }
    __syncthreads();  // the next iteration's prefetch must not overwrite this tile's operands
    buf = nb;
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = m0 + orow + i;
    if (row >= a.m) continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int col = n0 + ocol + j;
      if (col >= a.n) continue;
      const std::size_t at = static_cast<std::size_t>(row) * a.ldc + col;
      const float add = a.residual != nullptr ? hip_block::detail::f(a.residual[at]) : 0.0f;
      a.c[at] = hip_block::detail::bf(acc[i][j] + add);
    }
  }
}

#else  // !__GFX10__

template <int ROWS, int BKB, int THREADS>
struct Stager {
  static constexpr int kPerRow = BKB / 16;
  static constexpr int kChunks = ROWS * kPerRow;
  static constexpr int kPerThread = kChunks / THREADS;
  static_assert(kChunks % THREADS == 0, "threads must divide the tile's chunks");
  static constexpr int kStride = BKB + 32;

  uint4 regs[kPerThread];

  __device__ __forceinline__ void load(const bf16* src, int row0, int rows_total, int kbyte0, int kbytes) {
    const int tid = static_cast<int>(threadIdx.x);
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) {
      const int c = tid * kPerThread + i;
      const int row = row0 + c / kPerRow;
      const int kbyte = kbyte0 + (c % kPerRow) * 16;
      regs[i] = (row < rows_total && kbyte < kbytes)
                    ? *reinterpret_cast<const uint4*>(reinterpret_cast<const char*>(src) +
                                                       static_cast<std::size_t>(row) * kbytes + kbyte)
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

template <int BM, int BN, int BKB, int WARPS_M, int WARPS_N, int TM, int TN>
__global__ __launch_bounds__(WARPS_M* WARPS_N* kWave) void gemm_tiled(const Bf16GemmArgs a) {
  constexpr int kThreads = WARPS_M * WARPS_N * kWave;
  constexpr int kStride = BKB + 32;
  constexpr int kSteps = BKB / 32;
  static_assert(BM == WARPS_M * TM * 16 && BN == WARPS_N * TN * 16, "warp grid must tile the block");
  static_assert(Stager<BM, BKB, kThreads>::kStride == kStride, "staging stride must match read stride");

  __shared__ __align__(16) std::int8_t As[BM * kStride];
  __shared__ __align__(16) std::int8_t Bs[BN * kStride];

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wm = warp / WARPS_N, wn = warp % WARPS_N;
  const int r = lane & 15;
  const int blocks_n = gridDim.x;
  const int blocks_m = gridDim.y;
  const int bid = blockIdx.y * blocks_n + blockIdx.x;
  constexpr int kGroupM = 4;
  const int per_group = kGroupM * blocks_n;
  const int group = bid / per_group;
  const int idx = bid - group * per_group;
  const int rows = blocks_m - group * kGroupM < kGroupM ? blocks_m - group * kGroupM : kGroupM;
  const int mb = group * kGroupM + idx % rows;
  const int nb = idx / rows;
  const int m0 = mb * BM, n0 = nb * BN;

  v8f acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  const int kbytes = 2 * a.k;
  Stager<BM, BKB, kThreads> sa, sb;
  sa.load(a.a, m0, a.m, 0, kbytes);
  sb.load(a.b, n0, a.n, 0, kbytes);
  sa.store(As);
  sb.store(Bs);
  __syncthreads();

  for (int k0 = 0; k0 < kbytes; k0 += BKB) {
    const int knext = k0 + BKB;
    const bool has_next = knext < kbytes;
    if (has_next) {
      sa.load(a.a, m0, a.m, knext, kbytes);
      sb.load(a.b, n0, a.n, knext, kbytes);
    }
    v16bf af[2][TM], bf[2][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
      af[0][i] = *reinterpret_cast<const v16bf*>(As + (wm * (TM * 16) + i * 16 + r) * kStride);
#pragma unroll
    for (int j = 0; j < TN; ++j)
      bf[0][j] = *reinterpret_cast<const v16bf*>(Bs + (wn * (TN * 16) + j * 16 + r) * kStride);
#pragma unroll
    for (int kk = 0; kk < kSteps; ++kk) {
      const int cur = kk & 1, nxt = cur ^ 1;
      const int kb = (kk + 1) * 32;
      if (kk + 1 < kSteps) {
#pragma unroll
        for (int i = 0; i < TM; ++i)
          af[nxt][i] = *reinterpret_cast<const v16bf*>(As + (wm * (TM * 16) + i * 16 + r) * kStride + kb);
#pragma unroll
        for (int j = 0; j < TN; ++j)
          bf[nxt][j] = *reinterpret_cast<const v16bf*>(Bs + (wn * (TN * 16) + j * 16 + r) * kStride + kb);
      }
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af[cur][i], bf[cur][j], acc[i][j]);
    }
    if (has_next) {
      __syncthreads();
      sa.store(As);
      sb.store(Bs);
      __syncthreads();
    }
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int row = m0 + wm * (TM * 16) + i * 16 + 2 * e + (lane >> 4);
      if (row >= a.m) continue;
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int col = n0 + wn * (TN * 16) + j * 16 + (lane & 15);
        if (col >= a.n) continue;
        const std::size_t at = static_cast<std::size_t>(row) * a.ldc + col;
        const float add = a.residual != nullptr ? hip_block::detail::f(a.residual[at]) : 0.0f;
        a.c[at] = hip_block::detail::bf(acc[i][j][e] + add);
      }
    }
  }
}

#endif  // !__GFX10__

}  // namespace

void bf16_gemm_tiled(bf16* c, std::int64_t ldc, const bf16* a, std::int64_t lda,
                     const bf16* b, std::int64_t ldb, int m, int n, int k, hipStream_t stream,
                     const bf16* residual) {
  if (m == 0 || n == 0 || k == 0) return;
  const Bf16GemmArgs args{a, b, c, residual, lda, ldb, ldc, m, n, k};
  constexpr int kChunk = 4096;
  // The same template instantiation is launched on both architectures so the mangled kernel name exists in every
  // object of the fat binary. Only the grid and the thread count differ: RDNA2's body uses a 64x64 tile / 256 threads.
  const bool rdna2 = device_is_rdna2();
  const int tile = rdna2 ? 64 : 128;
  const int gx = (m + tile - 1) / tile, gy = (n + tile - 1) / tile;
  for (int x0 = 0; x0 < gx; x0 += kChunk) {
    const int xn = x0 + kChunk < gx ? kChunk : gx - x0;
    dim3 grid(gy, xn, 1);
    if (rdna2) {
      gemm_tiled<128, 128, 128, 2, 2, 4, 4><<<grid, 256, 0, stream>>>(args);
    } else if (k >= 2048) {
      gemm_tiled<128, 128, 128, 2, 2, 4, 4><<<grid, 128, 0, stream>>>(args);
    } else {
      gemm_tiled<128, 128, 64, 4, 4, 2, 2><<<grid, 512, 0, stream>>>(args);
    }
  }
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) throw std::runtime_error(std::string("bf16_gemm_tiled: ") + hipGetErrorString(e));
}

}  // namespace mi::hip

#else  // !ROCWMMA_I8_LEAF

#include <stdexcept>
#include "bf16_gemm.h"
namespace mi::hip {
void bf16_gemm_tiled(hip_block::bf16_bits*, std::int64_t, const hip_block::bf16_bits*, std::int64_t,
                     const hip_block::bf16_bits*, std::int64_t, int, int, int, hipStream_t,
                     const hip_block::bf16_bits*) {
  throw std::runtime_error("bf16 tiled GEMM needs an AMD gfx11 build");
}
}  // namespace mi::hip

#endif
