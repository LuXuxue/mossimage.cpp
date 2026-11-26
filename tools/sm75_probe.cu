// Small, self-contained probes for the arithmetic that matters on sm_75 GPUs (GTX 16xx / TU116 / TU117 have no
// tensor cores). Each probe checks correctness against a host reference and times the device work with CUDA events.
// Built as `sm75_probe` with MOSSINFER_SM75=ON; a development tool, not installed.
//
// Usage: sm75_probe [info|bw|alu|gemm|dp4a|attn|all]
//   info  device properties and clock/memory limits
//   bw    H2D/D2H/D2D and vectorised copy bandwidth
//   alu   raw dependent-chain throughput: fp32 FMA, fp16x2 FMA, INT32 DP4A, INT32 MAD
//   gemm  cuBLASLt fp32 / fp16 / bf16 and the engine's bf16->fp16 GEMM, plus CUTLASS INT8 ConvRot GEMM
//   dp4a  hand-written tiled DP4A INT8 GEMM (native instruction) vs the CUTLASS m8n8k16 (emulated MMA) path
//   attn  the sm_75 exact attention kernel at engine shapes
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend/cuda_libs.h"
#include "core/cuda_check.h"
#include "core/tensor.h"
#include "kernels/basic/gemm.h"
#include "kernels/quant/quant.h"
#include "kernels/sm75/sm75.h"

namespace {

using mi::DType;
using mi::Device;
using mi::Tensor;

std::uint16_t float_to_bf16_bits(float f) {
  std::uint32_t u = 0;
  std::memcpy(&u, &f, sizeof u);
  const std::uint32_t r = u + 0x7fffu + ((u >> 16) & 1u);
  return static_cast<std::uint16_t>(r >> 16);
}
float bf16_to_float(std::uint16_t b) {
  std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
  float f = 0;
  std::memcpy(&f, &u, sizeof f);
  return f;
}
std::uint32_t hash_u32(std::uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdull;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ull;
  x ^= x >> 33;
  return static_cast<std::uint32_t>(x);
}
std::vector<float> normal(std::size_t n, std::uint64_t seed, float scale) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint64_t key = seed * 0x9E3779B97F4A7C15ull + i * 2;
    const float u1 = (static_cast<float>(hash_u32(key) >> 8) + 1.f) * (1.f / 16777217.f);
    const float u2 = static_cast<float>(hash_u32(key + 1) >> 8) * (1.f / 16777216.f);
    v[i] = scale * std::sqrt(-2.f * std::log(u1)) * std::cos(6.283185307179586f * u2);
  }
  return v;
}

class DevBuf {
 public:
  explicit DevBuf(std::size_t bytes) { MI_CUDA_CHECK(cudaMalloc(&p_, bytes == 0 ? 1 : bytes)); }
  ~DevBuf() { cudaFree(p_); }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  void* get() const { return p_; }
  template <class T>
  void upload(const std::vector<T>& v) {
    MI_CUDA_CHECK(cudaMemcpy(p_, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  template <class T>
  std::vector<T> download(std::size_t n) const {
    std::vector<T> v(n);
    MI_CUDA_CHECK(cudaMemcpy(v.data(), p_, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
  }

 private:
  void* p_ = nullptr;
};

// Times `fn` with CUDA events; returns ms per call. The first calls warm up caches and lazy library state.
template <class F>
double bench(F&& fn, int iters = 30, int warmup = 3) {
  for (int i = 0; i < warmup; ++i) fn();
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t a, b;
  MI_CUDA_CHECK(cudaEventCreate(&a));
  MI_CUDA_CHECK(cudaEventCreate(&b));
  MI_CUDA_CHECK(cudaEventRecord(a));
  for (int i = 0; i < iters; ++i) fn();
  MI_CUDA_CHECK(cudaEventRecord(b));
  MI_CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0.f;
  MI_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  MI_CUDA_CHECK(cudaEventDestroy(a));
  MI_CUDA_CHECK(cudaEventDestroy(b));
  return ms / iters;
}

struct Stats {
  double cosine = 0, max_abs = 0, max_rel = 0;
  bool finite = true;
};
Stats compare(const std::vector<float>& got, const std::vector<float>& want) {
  Stats s;
  double dot = 0, ng = 0, nw = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double g = got[i], w = want[i];
    s.finite &= std::isfinite(static_cast<float>(g));
    dot += g * w;
    ng += g * g;
    nw += w * w;
    s.max_abs = std::max(s.max_abs, std::fabs(g - w));
    s.max_rel = std::max(s.max_rel, std::fabs(g - w) / std::max(1e-3, std::fabs(w)));
  }
  s.cosine = dot / std::sqrt(ng * nw);
  return s;
}
void print_stats(const Stats& s) {
  std::printf("cosine %.9f max|diff| %.3g max|rel| %.3g%s", s.cosine, s.max_abs, s.max_rel,
              s.finite ? "" : " NON-FINITE");
}

void info() {
  int dev = 0;
  MI_CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp p{};
  MI_CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  int clock = 0, mem_clock = 0;
  cudaDeviceGetAttribute(&clock, cudaDevAttrClockRate, dev);
  cudaDeviceGetAttribute(&mem_clock, cudaDevAttrMemoryClockRate, dev);
  std::printf("device: %s  cc %d.%d  SMs %d  warp %d  max threads/SM %d\n", p.name, p.major, p.minor,
              p.multiProcessorCount, p.warpSize, p.maxThreadsPerMultiProcessor);
  std::printf("clocks: SM %d MHz  mem %d MHz  bus %d bit  L2 %d KiB  shared/block %d KiB\n", clock / 1000,
              mem_clock / 1000, p.memoryBusWidth, p.l2CacheSize / 1024, p.sharedMemPerBlock / 1024);
  std::printf("DRAM: %.2f GiB  theoretical bandwidth %.1f GB/s\n", p.totalGlobalMem / (1024.0 * 1024 * 1024),
              2.0 * mem_clock / 1000.0 * p.memoryBusWidth / 8.0 / 1000.0);
  std::printf("FP32 SIMT peak (clock x 32 FMA x FLOPs): %.2f TFLOPS\n",
              clock / 1000.0 * 1e6 * 32 * 2 * p.multiProcessorCount / 1e12);
  std::printf("tensor cores: %s\n", mi::kernels::sm75::no_tensor_cores() ? "NO (TU116/TU117) " : "yes");
}

// ---------------------------------------------------------------- bandwidth
__global__ void copy_f4(const float4* __restrict__ src, float4* __restrict__ dst, std::size_t n) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
    dst[i] = src[i];
}

void bandwidth() {
  constexpr std::size_t kBytes = 64u << 20;
  std::vector<float> h(kBytes / sizeof(float), 1.f);
  DevBuf a(kBytes), b(kBytes);
  a.upload(h);
  MI_CUDA_CHECK(cudaMemset(b.get(), 0, kBytes));
  const auto ms_h2d = bench([&] { MI_CUDA_CHECK(cudaMemcpy(a.get(), h.data(), kBytes, cudaMemcpyHostToDevice)); });
  const auto ms_d2h = bench([&] { MI_CUDA_CHECK(cudaMemcpy(h.data(), b.get(), kBytes, cudaMemcpyDeviceToHost)); });
  const auto ms_d2d = bench(
      [&] { MI_CUDA_CHECK(cudaMemcpyAsync(b.get(), a.get(), kBytes, cudaMemcpyDeviceToDevice, nullptr)); });
  const auto ms_kernel = bench([&] { copy_f4<<<1024, 256>>>(static_cast<const float4*>(a.get()),
                                                           static_cast<float4*>(b.get()), kBytes / sizeof(float4)); });
  const auto gb = [](double ms) { return kBytes / (ms * 1e-3) / 1e9; };
  std::printf("bw 64 MiB: H2D %6.2f GB/s  D2H %6.2f GB/s  cudaMemcpy D2D %6.2f GB/s  vectorised r+w %6.2f GB/s\n",
              gb(ms_h2d), gb(ms_d2h), gb(ms_d2d), 2 * gb(ms_kernel));
}

// ---------------------------------------------------------------- ALU peaks
// Volatile inline asm with a dependency on the previous iteration: the compiler cannot hoist or remove the work.
__global__ void alu_f32(float* out, int iters) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float a = 1.f + tid * 1e-6f, b = 1.0000001f, c = 0.9999999f;
#pragma unroll 1
  for (int i = 0; i < iters; ++i) asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(b), "f"(c));
  out[tid] = a;
}
__global__ void alu_f16(unsigned* out, int iters) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned h = 0x3c003c00u, k = 0x3c003800u;  // half2(1,1), half2(1,0.5)
#pragma unroll 1
  for (int i = 0; i < iters; ++i) asm volatile("fma.rn.f16x2 %0, %0, %1, %1;" : "+r"(h) : "r"(k));
  out[tid] = h;
}
__global__ void alu_dp4a(int* out, int iters) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int a = 0x01010101 + (tid & 7), b = 0x02020202 + (tid & 3);
  int c = 0;
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    int d = 0;
    asm volatile("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    c = d;
  }
  out[tid] = c;
}
__global__ void alu_i32(int* out, int iters) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int a = 1 + (tid & 7), b = 2;
#pragma unroll 1
  for (int i = 0; i < iters; ++i) asm volatile("mad.lo.s32 %0, %0, %1, %1;" : "+r"(a) : "r"(b));
  out[tid] = a;
}

void alu() {
  constexpr int blocks = 128, threads = 256, iters = 4096;
  DevBuf out(static_cast<std::size_t>(blocks) * threads * 4);
  const double ms_f32 = bench([&] { alu_f32<<<blocks, threads>>>(static_cast<float*>(out.get()), iters); }, 15);
  const double ms_f16 = bench([&] { alu_f16<<<blocks, threads>>>(static_cast<unsigned*>(out.get()), iters); }, 15);
  const double ms_dp4a = bench([&] { alu_dp4a<<<blocks, threads>>>(static_cast<int*>(out.get()), iters); }, 15);
  const double ms_i32 = bench([&] { alu_i32<<<blocks, threads>>>(static_cast<int*>(out.get()), iters); }, 15);
  const double n = static_cast<double>(blocks) * threads * iters;
  const auto rate = [&](double ms, double ops) { return ops / (ms * 1e-3) / 1e9; };
  std::printf("alu  %d blocks x %d threads x %d dependent instructions\n", blocks, threads, iters);
  std::printf("  fp32 fma      : %7.3f ms  %7.1f GFLOP/s  (%7.1f G instr/s)\n", ms_f32, rate(ms_f32, n * 2),
              rate(ms_f32, n));
  std::printf("  fp16x2 fma    : %7.3f ms  %7.1f GFLOP/s  (%7.1f G instr/s)\n", ms_f16, rate(ms_f16, n * 4),
              rate(ms_f16, n));
  std::printf("  int8 dp4a     : %7.3f ms  %7.1f GOP/s     (%7.1f G instr/s)\n", ms_dp4a, rate(ms_dp4a, n * 8),
              rate(ms_dp4a, n));
  std::printf("  int32 mad     : %7.3f ms  %7.1f GOP/s     (%7.1f G instr/s)\n", ms_i32, rate(ms_i32, n * 2),
              rate(ms_i32, n));
}

// ---------------------------------------------------------------- cuBLASLt GEMM
void check_lt(cublasStatus_t s, const char* what) {
  if (s != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string("cuBLASLt ") + what + ": " + cublasLtGetStatusString(s));
}
#define CHECK_LT(x) check_lt((x), #x)

// Row-major D[M,N] = X[M,K] * W[N,K]^T, read by cuBLASLt as column-major D^T = op_T(W) * X (the engine's TN layout).
class LtGemm {
 public:
  LtGemm(cudaDataType_t ab, cudaDataType_t out) : ab_(ab), out_(out) {
    CHECK_LT(cublasLtCreate(&h_));
    CHECK_LT(cublasLtMatmulDescCreate(&desc_, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    CHECK_LT(cublasLtMatmulDescSetAttribute(desc_, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
    CHECK_LT(cublasLtMatmulDescSetAttribute(desc_, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
  }
  ~LtGemm() {
    if (d_ != nullptr) cublasLtMatrixLayoutDestroy(d_);
    if (b_ != nullptr) cublasLtMatrixLayoutDestroy(b_);
    if (a_ != nullptr) cublasLtMatrixLayoutDestroy(a_);
    if (desc_ != nullptr) cublasLtMatmulDescDestroy(desc_);
    if (h_ != nullptr) cublasLtDestroy(h_);
  }
  LtGemm(const LtGemm&) = delete;
  LtGemm& operator=(const LtGemm&) = delete;

  void prepare(int m, int n, int k, std::size_t ws) {
    if (a_ != nullptr) {
      cublasLtMatrixLayoutDestroy(a_);
      cublasLtMatrixLayoutDestroy(b_);
      cublasLtMatrixLayoutDestroy(d_);
    }
    // W [N,K] row-major is read as column-major [K,N] with ld = K; X [M,K] likewise with ld = K.
    CHECK_LT(cublasLtMatrixLayoutCreate(&a_, ab_, k, n, k));
    CHECK_LT(cublasLtMatrixLayoutCreate(&b_, ab_, k, m, k));
    CHECK_LT(cublasLtMatrixLayoutCreate(&d_, out_, n, m, n));
    cublasLtMatmulPreference_t pref = nullptr;
    CHECK_LT(cublasLtMatmulPreferenceCreate(&pref));
    struct Guard {
      cublasLtMatmulPreference_t p;
      ~Guard() { cublasLtMatmulPreferenceDestroy(p); }
    } guard{pref};
    CHECK_LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws));
    cublasLtMatmulHeuristicResult_t r{};
    int found = 0;
    const cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(h_, desc_, a_, b_, d_, d_, pref, 1, &r, &found);
    if (st != CUBLAS_STATUS_SUCCESS || found == 0) {
      std::printf("    [cuBLASLt heuristic %s, found %d: using the default algorithm]\n",
                  cublasLtGetStatusString(st), found);
      algo_ = cublasLtMatmulAlgo_t{};
      ws_ = 0;
    } else {
      algo_ = r.algo;
      ws_ = r.workspaceSize;
    }
  }
  void run(const void* w, const void* x, void* d, void* ws, cudaStream_t stream) {
    const float alpha = 1.f, beta = 0.f;
    CHECK_LT(cublasLtMatmul(h_, desc_, &alpha, w, a_, x, b_, &beta, d, d_, d, d_, &algo_, ws, ws_, stream));
  }

 private:
  cudaDataType_t ab_, out_;
  cublasLtHandle_t h_ = nullptr;
  cublasLtMatmulDesc_t desc_ = nullptr;
  cublasLtMatrixLayout_t a_ = nullptr, b_ = nullptr, d_ = nullptr;
  cublasLtMatmulAlgo_t algo_{};
  std::size_t ws_ = 0;
};

template <bool F16>
__global__ void simt_gemm_tiled(const void* __restrict__ Ap, const void* __restrict__ BTp, float* __restrict__ C,
                                int M, int N, int K);

void gemm_shape(int M, int N, int K, bool check) {
  std::printf("gemm  M %d N %d K %d  (%.2f GFLOP)\n", M, N, K, 2.0 * M * N * K / 1e9);
  const auto xv = normal(static_cast<std::size_t>(M) * K, 51, 0.15f);
  const auto wv = normal(static_cast<std::size_t>(N) * K, 52, 0.05f);
  std::vector<float> host(static_cast<std::size_t>(M) * N);
  if (check) {
    for (int r = 0; r < M; ++r)
      for (int c = 0; c < N; ++c) {
        double acc = 0;
        for (int k = 0; k < K; ++k) acc += static_cast<double>(xv[r * K + k]) * wv[c * K + k];
        host[r * N + c] = static_cast<float>(acc);
      }
  }
  std::vector<std::uint16_t> xbf(xv.size()), xf16(xv.size()), wf16(wv.size());
  for (std::size_t i = 0; i < xv.size(); ++i) {
    xbf[i] = float_to_bf16_bits(xv[i]);
    const __half h = __float2half_rn(xv[i]);
    std::memcpy(&xf16[i], &h, 2);
  }
  for (std::size_t i = 0; i < wv.size(); ++i) {
    const __half h = __float2half_rn(wv[i]);
    std::memcpy(&wf16[i], &h, 2);
  }
  DevBuf dxf(xv.size() * 4), dwf(wv.size() * 4);  // fp32 operands
  DevBuf dxh(xf16.size() * 2), dwh(wf16.size() * 2);  // fp16 operands
  DevBuf dxb(xbf.size() * 2);                       // bf16 activations for the engine path
  dxf.upload(xv);
  dwf.upload(wv);
  dxh.upload(xf16);
  dwh.upload(wf16);
  dxb.upload(xbf);
  std::vector<float> wTf(static_cast<std::size_t>(K) * N);
  std::vector<std::uint16_t> wTh(wTf.size());
  for (int k = 0; k < K; ++k)
    for (int n = 0; n < N; ++n) {
      wTf[k * N + n] = wv[n * K + k];
      wTh[k * N + n] = wf16[n * K + k];
    }
  DevBuf dWTf(wTf.size() * 4), dWTh(wTh.size() * 2);
  dWTf.upload(wTf);
  dWTh.upload(wTh);
  const std::size_t ws_bytes = 32u << 20;
  DevBuf ws(ws_bytes);
  const double flops = 2.0 * M * N * K;
  const auto show = [&](const char* name, double ms, const std::vector<float>& got) {
    std::printf("  %-22s %8.3f ms  %6.2f TFLOPS  ", name, ms, flops / (ms * 1e-3) / 1e12);
    if (check) print_stats(compare(got, host));
    std::printf("\n");
  };

  // fp32 SGEMM through cuBLASLt.
  DevBuf d32(static_cast<std::size_t>(M) * N * 4);
  std::printf("  [fp32] preparing\n");
  {
    LtGemm g(CUDA_R_32F, CUDA_R_32F);
    g.prepare(M, N, K, ws_bytes);
    const auto ms = bench([&] { g.run(dwf.get(), dxf.get(), d32.get(), ws.get(), nullptr); }, 20);
    g.run(dwf.get(), dxf.get(), d32.get(), ws.get(), nullptr);
    show("fp32 (cuBLASLt F32)", ms, check ? d32.download<float>(host.size()) : std::vector<float>{});
  }
  // fp16 in/out. On TU117 cuBLASLt has no tensor cores and must use ordinary arithmetic.
  DevBuf dh(static_cast<std::size_t>(M) * N * 2);
  std::printf("  [fp16] preparing\n");
  {
    LtGemm g(CUDA_R_16F, CUDA_R_16F);
    g.prepare(M, N, K, ws_bytes);
    const auto ms = bench([&] { g.run(dwh.get(), dxh.get(), dh.get(), ws.get(), nullptr); }, 20);
    if (check) {
      g.run(dwh.get(), dxh.get(), dh.get(), ws.get(), nullptr);
      const auto bits = dh.download<std::uint16_t>(host.size());
      std::vector<float> got(bits.size());
      for (std::size_t i = 0; i < bits.size(); ++i) {
        __half h;
        std::memcpy(&h, &bits[i], 2);
        got[i] = __half2float(h);
      }
      show("fp16 (cuBLASLt F16)", ms, got);
    } else {
      show("fp16 (cuBLASLt F16)", ms, {});
    }
  }
  // Hand-written tiled SIMT GEMMs (fp32 operands, and fp16 operands converted to fp32 in the kernel).
  DevBuf dsim(static_cast<std::size_t>(M) * N * 4);
  {
    const dim3 grid((N + 63) / 64, (M + 63) / 64);
    const auto ms = bench(
        [&] { simt_gemm_tiled<false><<<grid, dim3(32, 8)>>>(dxf.get(), dWTf.get(), static_cast<float*>(dsim.get()), M, N, K); },
        20);
    if (check) {
      simt_gemm_tiled<false><<<grid, dim3(32, 8)>>>(dxf.get(), dWTf.get(), static_cast<float*>(dsim.get()), M, N, K);
      MI_CUDA_CHECK(cudaDeviceSynchronize());
      show("hand tiled fp32 SIMT", ms, dsim.download<float>(host.size()));
    } else {
      show("hand tiled fp32 SIMT", ms, {});
    }
    const auto ms16 = bench(
        [&] { simt_gemm_tiled<true><<<grid, dim3(32, 8)>>>(dxh.get(), dWTh.get(), static_cast<float*>(dsim.get()), M, N, K); },
        20);
    if (check) {
      simt_gemm_tiled<true><<<grid, dim3(32, 8)>>>(dxh.get(), dWTh.get(), static_cast<float*>(dsim.get()), M, N, K);
      MI_CUDA_CHECK(cudaDeviceSynchronize());
      show("hand tiled fp16->f32", ms16, dsim.download<float>(host.size()));
    } else {
      show("hand tiled fp16->f32", ms16, {});
    }
  }
  // The engine's actual bf16-linear path: bf16 x + fp16 w, fp16 GEMM with fp32 acc and a SIMT epilogue.
  std::printf("  [engine bf16->fp16] preparing\n");
  {
    DevBuf dbf(static_cast<std::size_t>(M) * N * 2);
    mi::kernels::Gemm g;
    Tensor tx = Tensor::view(dxb.get(), DType::kBF16, Device::kCUDA, {M, K});
    Tensor tw = Tensor::view(dwh.get(), DType::kF16, Device::kCUDA, {N, K});
    Tensor td = Tensor::view(dbf.get(), DType::kBF16, Device::kCUDA, {M, N});
    const auto ms = bench([&] { g.linear(td, tx, tw, nullptr, ws.get(), ws_bytes, nullptr); }, 20);
    if (check) {
      g.linear(td, tx, tw, nullptr, ws.get(), ws_bytes, nullptr);
      const auto bits = dbf.download<std::uint16_t>(host.size());
      std::vector<float> got(bits.size());
      for (std::size_t i = 0; i < bits.size(); ++i) got[i] = bf16_to_float(bits[i]);
      show("engine bf16->fp16 GEMM", ms, got);
    } else {
      show("engine bf16->fp16 GEMM", ms, {});
    }
  }
}

// Hand-written tiled GEMM with fp32 accumulation. F16=true loads fp16 operands (the engine's sm75 weight format) and
// converts them to fp32 in the kernel; the cuBLASLt fp16 route on TU117 reaches only ~0.3 TFLOPS.
template <bool F16>
__global__ void simt_gemm_tiled(const void* __restrict__ Ap, const void* __restrict__ BTp, float* __restrict__ C,
                                int M, int N, int K) {
  constexpr int BM = 64, BN = 64, BK = 8;
  __shared__ float As[BM][BK], Bs[BK][BN];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int r0 = blockIdx.y * BM, c0 = blockIdx.x * BN;
  float acc[BM / 8][BN / 32];
#pragma unroll
  for (int i = 0; i < BM / 8; ++i)
#pragma unroll
    for (int j = 0; j < BN / 32; ++j) acc[i][j] = 0.f;
  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int i = ty; i < BM; i += 8)
      for (int j = 0; j < 8; ++j) {
        const int r = r0 + i, k = k0 + j;
        if (r >= M || k >= K) continue;
        if constexpr (F16) {
          const __half* a = static_cast<const __half*>(Ap);
          As[i][j] = __half2float(a[r * K + k]);
        } else {
          const float* a = static_cast<const float*>(Ap);
          As[i][j] = a[r * K + k];
        }
      }
    for (int i = 0; i < BK; ++i)
      for (int j = tx; j < BN; j += 32) {
        const int k = k0 + i, c = c0 + j;
        if (k >= K || c >= N) continue;
        if constexpr (F16) {
          const __half* b = static_cast<const __half*>(BTp);
          Bs[i][j] = __half2float(b[k * N + c]);
        } else {
          const float* b = static_cast<const float*>(BTp);
          Bs[i][j] = b[k * N + c];
        }
      }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; ++k)
#pragma unroll
      for (int i = 0; i < BM / 8; ++i)
#pragma unroll
        for (int j = 0; j < BN / 32; ++j) acc[i][j] = fmaf(As[ty + i * 8][k], Bs[k][tx + j * 32], acc[i][j]);
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < BM / 8; ++i)
#pragma unroll
    for (int j = 0; j < BN / 32; ++j) {
      const int r = r0 + ty + i * 8, c = c0 + tx + j * 32;
      if (r < M && c < N) C[r * N + c] = acc[i][j];
    }
}

// ---------------------------------------------------------------- hand-written DP4A GEMM
// B is stored [K,N] so threads along x read contiguous bytes; A is [M,K]. BM rows per block, BN columns.
template <int BM, int BN, int BK>
__global__ void dp4a_gemm(const std::int8_t* __restrict__ A, const std::int8_t* __restrict__ BT,
                          const float* __restrict__ as, const float* __restrict__ bs,
                          __nv_bfloat16* __restrict__ C, int M, int N, int K) {
  __shared__ std::int8_t As[BM][BK];
  __shared__ std::int8_t Bs[BK][BN];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * BM + ty, col = blockIdx.x * BN + tx;
  int acc = 0;
  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int i = ty; i < BM; i += blockDim.y)
      for (int j = tx; j < BK; j += blockDim.x)
        As[i][j] = (blockIdx.y * BM + i >= M) || (k0 + j >= K) ? 0 : A[(i + blockIdx.y * BM) * K + k0 + j];
    for (int i = ty; i < BK; i += blockDim.y)
      for (int j = tx; j < BN; j += blockDim.x)
        Bs[i][j] = (k0 + i >= K) || (blockIdx.x * BN + j >= N) ? 0
                                                               : BT[(k0 + i) * N + j + blockIdx.x * BN];
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; k += 4) {
      const int av = *reinterpret_cast<const int*>(&As[ty][k]);
      // B tile is [BK][BN]: four consecutive K bytes for one column are strided by BN, so pack them by hand.
      const int bv = static_cast<unsigned char>(Bs[k][tx]) | (static_cast<unsigned char>(Bs[k + 1][tx]) << 8) |
                     (static_cast<unsigned char>(Bs[k + 2][tx]) << 16) |
                     (static_cast<unsigned char>(Bs[k + 3][tx]) << 24);
      acc = __dp4a(av, bv, acc);
    }
    __syncthreads();
  }
  if (row < M && col < N) {
    float v = static_cast<float>(acc) * as[row];
    v = v * bs[col];
    C[row * N + col] = __float2bfloat16_rn(v);
  }
}

// A bigger DP4A tile with real reuse: 64x64 output tile, 128-deep K, eight rows x two columns per thread.
__global__ void dp4a_gemm_tiled(const std::int8_t* __restrict__ A, const std::int8_t* __restrict__ BT,
                                const float* __restrict__ as, const float* __restrict__ bs,
                                __nv_bfloat16* __restrict__ C, int M, int N, int K) {
  constexpr int BM = 64, BN = 64, BK = 128;
  __shared__ std::int8_t As[BM][BK];
  __shared__ std::int8_t Bs[BK][BN];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int r0 = blockIdx.y * BM, c0 = blockIdx.x * BN;
  int acc[BM / 8][BN / 32];
#pragma unroll
  for (int i = 0; i < BM / 8; ++i)
#pragma unroll
    for (int j = 0; j < BN / 32; ++j) acc[i][j] = 0;
  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int i = ty; i < BM; i += 8)
      for (int j = tx; j < BK; j += 32)
        As[i][j] = (r0 + i < M) && (k0 + j < K) ? A[(r0 + i) * K + k0 + j] : 0;
    for (int i = ty; i < BK; i += 8)
      for (int j = tx; j < BN; j += 32)
        Bs[i][j] = (k0 + i < K) && (c0 + j < N) ? BT[(k0 + i) * N + c0 + j] : 0;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; k += 4) {
      int av[BM / 8], bv[BN / 32];
#pragma unroll
      for (int i = 0; i < BM / 8; ++i) av[i] = *reinterpret_cast<const int*>(&As[ty + i * 8][k]);
#pragma unroll
      for (int j = 0; j < BN / 32; ++j) {
        const int cc = tx + j * 32;
        bv[j] = static_cast<unsigned char>(Bs[k][cc]) | (static_cast<unsigned char>(Bs[k + 1][cc]) << 8) |
                (static_cast<unsigned char>(Bs[k + 2][cc]) << 16) |
                (static_cast<unsigned char>(Bs[k + 3][cc]) << 24);
      }
#pragma unroll
      for (int i = 0; i < BM / 8; ++i)
#pragma unroll
        for (int j = 0; j < BN / 32; ++j) acc[i][j] = __dp4a(av[i], bv[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < BM / 8; ++i)
#pragma unroll
    for (int j = 0; j < BN / 32; ++j) {
      const int r = r0 + ty + i * 8, c = c0 + tx + j * 32;
      if (r < M && c < N) {
        float v = static_cast<float>(acc[i][j]) * as[r];
        v = v * bs[c];
        C[r * N + c] = __float2bfloat16_rn(v);
      }
    }
}

// The same arithmetic with the engine's [N,K] weight layout, one output per thread (uncoalesced weight reads).
__global__ void dp4a_gemm_plain(const std::int8_t* __restrict__ A, const std::int8_t* __restrict__ B,
                                const float* __restrict__ as, const float* __restrict__ bs,
                                __nv_bfloat16* __restrict__ C, int M, int N, int K) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= M || col >= N) return;
  int acc = 0;
  const std::int8_t* a = A + row * K;
  const std::int8_t* b = B + col * K;
  for (int k = 0; k < K; k += 4) acc = __dp4a(*reinterpret_cast<const int*>(a + k), *reinterpret_cast<const int*>(b + k), acc);
  float v = static_cast<float>(acc) * as[row];
  v = v * bs[col];
  C[row * N + col] = __float2bfloat16_rn(v);
}

void dp4a_shape(int M, int N, int K, bool check) {
  std::printf("dp4a  M %d N %d K %d  (%.2f GFLOP)\n", M, N, K, 2.0 * M * N * K / 1e9);
  const auto xv = normal(static_cast<std::size_t>(M) * K, 51, 0.15f);
  const auto wv = normal(static_cast<std::size_t>(N) * K, 52, 0.05f);
  std::vector<float> want(static_cast<std::size_t>(M) * N);
  if (check) {
    for (int r = 0; r < M; ++r)
      for (int c = 0; c < N; ++c) {
        int acc = 0;
        for (int k = 0; k < K; ++k) {
          const int a = static_cast<int>(static_cast<std::int8_t>(std::min(127.f, std::max(-127.f, xv[r * K + k] * 127.f))));
          const int b = static_cast<int>(static_cast<std::int8_t>(std::min(127.f, std::max(-127.f, wv[c * K + k] * 127.f))));
          acc += a * b;
        }
        want[r * N + c] = static_cast<float>(acc) / (127.f * 127.f);
      }
  }
  std::vector<std::int8_t> axv(xv.size()), awv(wv.size()), btv(static_cast<std::size_t>(K) * N);
  for (std::size_t i = 0; i < xv.size(); ++i)
    axv[i] = static_cast<std::int8_t>(std::min(127.f, std::max(-127.f, xv[i] * 127.f)));
  for (std::size_t i = 0; i < wv.size(); ++i)
    awv[i] = static_cast<std::int8_t>(std::min(127.f, std::max(-127.f, wv[i] * 127.f)));
  for (int k = 0; k < K; ++k)
    for (int n = 0; n < N; ++n) btv[k * N + n] = awv[n * K + k];
  const std::vector<float> as(M, 1.f / 127.f), bs(N, 1.f / 127.f);
  DevBuf dA(axv.size()), dB(awv.size()), dBT(btv.size()), das(as.size() * 4), dbs(bs.size() * 4);
  DevBuf dC(static_cast<std::size_t>(M) * N * 2);
  dA.upload(axv);
  dB.upload(awv);
  dBT.upload(btv);
  das.upload(as);
  dbs.upload(bs);
  const double flops = 2.0 * M * N * K;
  auto* a = static_cast<const std::int8_t*>(dA.get());
  auto* wt = static_cast<const std::int8_t*>(dBT.get());
  auto* w = static_cast<const std::int8_t*>(dB.get());
  auto* sa = static_cast<const float*>(das.get());
  auto* sb = static_cast<const float*>(dbs.get());
  auto* c = static_cast<__nv_bfloat16*>(dC.get());
  constexpr int BM = 8, BN = 32, BK = 128;
  const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  const auto ms = bench([&] { dp4a_gemm<BM, BN, BK><<<grid, dim3(BN, BM)>>>(a, wt, sa, sb, c, M, N, K); }, 20);
  const dim3 grid2((N + 31) / 32, (M + 7) / 8);
  const auto ms_plain = bench(
      [&] { dp4a_gemm_plain<<<grid2, dim3(32, 8)>>>(a, w, sa, sb, c, M, N, K); }, 20);
  const dim3 grid3((N + 63) / 64, (M + 63) / 64);
  const auto ms_tiled = bench(
      [&] { dp4a_gemm_tiled<<<grid3, dim3(32, 8)>>>(a, wt, sa, sb, c, M, N, K); }, 20);
  std::printf("  DP4A 8x32 tile    %8.3f ms  %6.2f TFLOPS\n", ms, flops / (ms * 1e-3) / 1e12);
  std::printf("  DP4A 64x64 tile   %8.3f ms  %6.2f TFLOPS\n", ms_tiled, flops / (ms_tiled * 1e-3) / 1e12);
  std::printf("  plain DP4A [N,K]  %8.3f ms  %6.2f TFLOPS\n", ms_plain, flops / (ms_plain * 1e-3) / 1e12);
  if (check) {
    const auto verify = [&](const char* name, auto launch) {
      launch();
      MI_CUDA_CHECK(cudaDeviceSynchronize());
      const auto bits = dC.download<std::uint16_t>(want.size());
      std::vector<float> got(bits.size());
      for (std::size_t i = 0; i < bits.size(); ++i) got[i] = bf16_to_float(bits[i]);
      std::printf("  %-18s correctness: ", name);
      print_stats(compare(got, want));
      std::printf("\n");
    };
    verify("8x32 tile", [&] { dp4a_gemm<BM, BN, BK><<<grid, dim3(BN, BM)>>>(a, wt, sa, sb, c, M, N, K); });
    verify("64x64 tile", [&] { dp4a_gemm_tiled<<<grid3, dim3(32, 8)>>>(a, wt, sa, sb, c, M, N, K); });
    verify("plain [N,K]", [&] { dp4a_gemm_plain<<<grid2, dim3(32, 8)>>>(a, w, sa, sb, c, M, N, K); });
  }
  // The engine's INT8 ConvRot GEMM (CUTLASS m8n8k16 MMA, emulated on TU116/TU117) with the same operands.
  mi::kernels::quant::Int8Operands op;
  op.x_q = a;
  op.ldx = K;
  op.x_scale = sa;
  op.w = w;
  op.w_scale = sb;
  op.m = M;
  op.n = N;
  op.k = K;
  const auto ms_int8 = bench([&] { mi::kernels::quant::int8_gemm(op, nullptr, c, N, nullptr); }, 20);
  std::printf("  CUTLASS INT8 MMA %8.3f ms  %6.2f TFLOPS  (vs best hand DP4A %.2fx)\n", ms_int8,
              flops / (ms_int8 * 1e-3) / 1e12, std::min(ms, ms_tiled) / ms_int8);
}

// ---------------------------------------------------------------- attention timing
void attention() {
  const auto run = [](int len, int heads, int kv, int dim, bool causal, int iters) {
    const auto q = normal(static_cast<std::size_t>(len) * heads * dim, 71, 1.f);
    const auto k = normal(static_cast<std::size_t>(len) * kv * dim, 72, 1.f);
    const auto v = normal(static_cast<std::size_t>(len) * kv * dim, 73, 1.f);
    std::vector<std::uint16_t> qb(q.size()), kb(k.size()), vb(v.size());
    for (std::size_t i = 0; i < q.size(); ++i) qb[i] = float_to_bf16_bits(q[i]);
    for (std::size_t i = 0; i < k.size(); ++i) kb[i] = float_to_bf16_bits(k[i]);
    for (std::size_t i = 0; i < v.size(); ++i) vb[i] = float_to_bf16_bits(v[i]);
    DevBuf dq(qb.size() * 2), dk(kb.size() * 2), dv(vb.size() * 2), do_(qb.size() * 2), dcu(2 * 4);
    dq.upload(qb);
    dk.upload(kb);
    dv.upload(vb);
    const std::vector<int> cu{0, len};
    dcu.upload(cu);
    const std::size_t ws = static_cast<std::size_t>(len) * heads * sizeof(float);
    DevBuf dl(ws);
    auto t = [](void* p, DType dt, std::initializer_list<std::int64_t> s) {
      return Tensor::view(p, dt, Device::kCUDA, s);
    };
    const Tensor tq = t(dq.get(), DType::kBF16, {len, heads, dim});
    const Tensor tk = t(dk.get(), DType::kBF16, {len, kv, dim});
    const Tensor tv = t(dv.get(), DType::kBF16, {len, kv, dim});
    const Tensor to = t(do_.get(), DType::kBF16, {len, heads, dim});
    const Tensor tc = t(dcu.get(), DType::kI32, {2});
    const float scale = 1.f / std::sqrt(static_cast<float>(dim));
    const auto ms = bench(
        [&] {
          mi::kernels::sm75::flash_fwd(tq, tk, tv, to, tc, tc, len, causal, scale, dl.get(), ws, nullptr);
        },
        iters, 2);
    const double flops = (causal ? 0.5 : 1.0) * 2.0 * len * len * dim * heads;
    std::printf("attn L %5d heads %2d kv %2d dim %3d causal %d: %8.3f ms  %6.2f TFLOPS\n", len, heads, kv, dim,
                causal ? 1 : 0, ms, flops / (ms * 1e-3) / 1e12);
  };
  run(266, 24, 24, 128, false, 30);   // 256^2 DiT
  run(4113, 24, 24, 128, false, 5);   // 1024^2 DiT
  run(4096, 32, 8, 256, true, 5);     // text encoder
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::string what = argc > 1 ? argv[1] : "all";
  const std::string shape = argc > 2 ? argv[2] : "";
  mi::backend::load_cublaslt();
  try {
    if (what == "info" || what == "all") info();
    if (what == "bw" || what == "all") bandwidth();
    if (what == "alu" || what == "all") alu();
    if (what == "gemm" || what == "all") {
      mi::kernels::Gemm warm;  // creates the cuBLASLt handle once, outside the timings
      (void)warm;
      if (shape == "check") {
        gemm_shape(128, 512, 1024, true);
      } else {
        gemm_shape(256, 3072, 3072, false);  // one DiT attention/MLP linear
        gemm_shape(1024, 3072, 8192, false); // the ffn_down linear at a larger token count
        gemm_shape(256, 3072, 64, false);    // runtime LoRA delta, small image
        gemm_shape(4096, 3072, 64, false);   // runtime LoRA delta, 1024^2
      }
    }
    if (what == "dp4a" || what == "all") {
      if (shape == "check") {
        dp4a_shape(128, 512, 1024, true);
        dp4a_shape(130, 520, 1024, true);  // tails that are not multiples of the 64x64 tile
      } else {
        dp4a_shape(256, 3072, 3072, false);
        dp4a_shape(1024, 3072, 8192, false);
      }
    }
    if (what == "attn" || what == "all") attention();
  } catch (const std::exception& e) {
    std::printf("EXCEPTION: %s\n", e.what());
    return 2;
  }
  return 0;
}
