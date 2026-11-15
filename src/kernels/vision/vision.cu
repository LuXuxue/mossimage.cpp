#include "kernels/vision/vision.h"

#include <cuda_bf16.h>

#include <stdexcept>
#include <string>

#include "core/cuda_check.h"

namespace mi::kernels::vision {
namespace {

using bf16 = __nv_bfloat16;

void require(bool ok, const char* op, const std::string& what) {
  if (!ok) throw std::invalid_argument(std::string("kernels::vision::") + op + ": " + what);
}

void check_bf16(const char* op, const char* name, const Tensor& t) {
  require(t.dtype == DType::kBF16 && t.device == Device::kCUDA, op, std::string(name) + " must be a CUDA bf16 tensor");
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// One block of 256 threads per row; two passes over the row (mean, then variance), as torch's fp32 accumulation does.
__global__ void layer_norm_kernel(const bf16* x, const bf16* w, const bf16* b, bf16* out, int cols, float eps) {
  __shared__ float partial[8];
  const std::int64_t row = blockIdx.x;
  const bf16* xr = x + row * cols;
  bf16* yr = out + row * cols;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;

  float s = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) s += __bfloat162float(xr[c]);
  s = warp_sum(s);
  if (lane == 0) partial[warp] = s;
  __syncthreads();
  float total = 0.0f;
  for (int i = 0; i < 8; ++i) total += partial[i];
  const float mean = total / static_cast<float>(cols);
  __syncthreads();

  float v = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float d = __bfloat162float(xr[c]) - mean;
    v += d * d;
  }
  v = warp_sum(v);
  if (lane == 0) partial[warp] = v;
  __syncthreads();
  float var = 0.0f;
  for (int i = 0; i < 8; ++i) var += partial[i];
  const float rstd = rsqrtf(var / static_cast<float>(cols) + eps);

  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float y = (__bfloat162float(xr[c]) - mean) * rstd * __bfloat162float(w[c]) + __bfloat162float(b[c]);
    yr[c] = __float2bfloat16(y);
  }
}

template <bool kTanh>
__global__ void gelu_kernel(bf16* x, std::int64_t n) {
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const float v = __bfloat162float(x[i]);
    float y;
    if constexpr (kTanh) {
      constexpr float kBeta = 0.7978845608028654f;  // sqrt(2 / pi)
      y = 0.5f * v * (1.0f + tanhf(kBeta * (v + 0.044715f * v * v * v)));
    } else {
      y = 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));
    }
    x[i] = __float2bfloat16(y);
  }
}

// grid: one block per token; block: (128 columns, 16 heads / 2) covering 3 tensors x 16 heads x 128 columns.
__global__ void qkv_rope_pad_kernel(const bf16* qkv, const float* cos, const float* sin, bf16* q, bf16* k, bf16* v) {
  constexpr int kDim = kHeads * kHeadDim;  // 1152
  constexpr int kHalf = kHeadDim / 2;      // 36
  const std::int64_t t = blockIdx.x;
  const int col = threadIdx.x;  // 0..127
  const float* ct = cos + t * kHeadDim;
  const float* st = sin + t * kHeadDim;
  for (int which = 0; which < 3; ++which) {
    bf16* dst = which == 0 ? q : which == 1 ? k : v;
    for (int h = threadIdx.y; h < kHeads; h += blockDim.y) {
      bf16* row = dst + (t * kHeads + h) * kPaddedHeadDim;
      if (col >= kHeadDim) {
        row[col] = __float2bfloat16(0.0f);
        continue;
      }
      const bf16* src = qkv + t * 3 * kDim + which * kDim + h * kHeadDim;
      const float xv = __bfloat162float(src[col]);
      if (which == 2) {
        row[col] = src[col];
        continue;
      }
      // rotate_half: cat(-x[36:], x[:36])
      const float rot = col < kHalf ? -__bfloat162float(src[col + kHalf]) : __bfloat162float(src[col - kHalf]);
      row[col] = __float2bfloat16(xv * ct[col] + rot * st[col]);
    }
  }
}

__global__ void unpad_kernel(const bf16* attn, bf16* out, std::int64_t tokens) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t n = tokens * kHeads * kHeadDim;
  if (i >= n) return;
  const std::int64_t t = i / (kHeads * kHeadDim);
  const int rem = static_cast<int>(i % (kHeads * kHeadDim));
  const int h = rem / kHeadDim, c = rem % kHeadDim;
  out[i] = attn[(t * kHeads + h) * kPaddedHeadDim + c];
}

// kron^n(H4)[i][j] is the product over base-4 digits of H4[a][b], which is -1 exactly when a + b == 3.
__device__ __forceinline__ float hadamard_sign(int i, int j, int digits) {
  int neg = 0;
  for (int d = 0; d < digits; ++d, i >>= 2, j >>= 2) neg ^= ((i & 3) + (j & 3) == 3) ? 1 : 0;
  return neg ? -1.0f : 1.0f;
}

__global__ void dequant_kernel(const std::int8_t* q, const float* scale, bf16* out, std::int64_t rows, std::int64_t k, int g,
                               int digits, float inv_sqrt_g) {
  const std::int64_t n = rows * k;
  for (std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const std::int64_t r = idx / k;
    const std::int64_t c = idx % k;
    const float s = scale[r];
    if (g == 0) {
      out[idx] = __float2bfloat16(static_cast<float>(q[idx]) * s);
      continue;
    }
    const std::int64_t base = r * k + (c / g) * g;
    const int j = static_cast<int>(c % g);
    float acc = 0.0f;
    for (int i = 0; i < g; ++i) acc = fmaf(static_cast<float>(q[base + i]) * s, hadamard_sign(i, j, digits) * inv_sqrt_g, acc);
    out[idx] = __float2bfloat16(acc);
  }
}

unsigned grid_for(std::int64_t n, int block) {
  const std::int64_t g = (n + block - 1) / block;
  return static_cast<unsigned>(g < 65535 * 8 ? g : 65535 * 8);
}

}  // namespace

void layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps, const Tensor& out,
                cudaStream_t stream) {
  constexpr const char* op = "layer_norm";
  check_bf16(op, "x", x);
  check_bf16(op, "weight", weight);
  check_bf16(op, "bias", bias);
  check_bf16(op, "out", out);
  require(x.rank == 2 && out.rank == 2 && out.dim(0) == x.dim(0) && out.dim(1) == x.dim(1), op, "expected x, out [N, C]");
  require(weight.numel() == x.dim(1) && bias.numel() == x.dim(1), op, "weight and bias must have C elements");
  if (x.dim(0) == 0) return;
  layer_norm_kernel<<<static_cast<unsigned>(x.dim(0)), 256, 0, stream>>>(
      x.ptr<const bf16>(), weight.ptr<const bf16>(), bias.ptr<const bf16>(), out.ptr<bf16>(), static_cast<int>(x.dim(1)), eps);
  MI_CUDA_CHECK_LAUNCH();
}

void gelu_tanh(const Tensor& x, cudaStream_t stream) {
  check_bf16("gelu_tanh", "x", x);
  if (x.numel() == 0) return;
  gelu_kernel<true><<<grid_for(x.numel(), 256), 256, 0, stream>>>(x.ptr<bf16>(), x.numel());
  MI_CUDA_CHECK_LAUNCH();
}

void gelu_erf(const Tensor& x, cudaStream_t stream) {
  check_bf16("gelu_erf", "x", x);
  if (x.numel() == 0) return;
  gelu_kernel<false><<<grid_for(x.numel(), 256), 256, 0, stream>>>(x.ptr<bf16>(), x.numel());
  MI_CUDA_CHECK_LAUNCH();
}

void qkv_rope_pad(const Tensor& qkv, const Tensor& cos, const Tensor& sin, const Tensor& q_out, const Tensor& k_out,
                  const Tensor& v_out, cudaStream_t stream) {
  constexpr const char* op = "qkv_rope_pad";
  check_bf16(op, "qkv", qkv);
  const std::int64_t n = qkv.dim(0);
  require(qkv.rank == 2 && qkv.dim(1) == 3 * kHeads * kHeadDim, op, "expected qkv [N, 3456]");
  require(cos.dtype == DType::kF32 && sin.dtype == DType::kF32 && cos.numel() == n * kHeadDim && sin.numel() == n * kHeadDim,
          op, "expected cos, sin f32 [N, 72]");
  for (const Tensor* t : {&q_out, &k_out, &v_out}) {
    check_bf16(op, "out", *t);
    require(t->numel() == n * kHeads * kPaddedHeadDim, op, "expected outputs [N, 16, 128]");
  }
  if (n == 0) return;
  qkv_rope_pad_kernel<<<static_cast<unsigned>(n), dim3(kPaddedHeadDim, 8), 0, stream>>>(
      qkv.ptr<const bf16>(), cos.ptr<const float>(), sin.ptr<const float>(), q_out.ptr<bf16>(), k_out.ptr<bf16>(),
      v_out.ptr<bf16>());
  MI_CUDA_CHECK_LAUNCH();
}

void unpad_heads(const Tensor& attn, const Tensor& out, cudaStream_t stream) {
  constexpr const char* op = "unpad_heads";
  check_bf16(op, "attn", attn);
  check_bf16(op, "out", out);
  const std::int64_t n = attn.dim(0);
  require(attn.numel() == n * kHeads * kPaddedHeadDim && out.numel() == n * kHeads * kHeadDim, op,
          "expected attn [N, 16, 128] and out [N, 1152]");
  if (n == 0) return;
  const std::int64_t total = n * kHeads * kHeadDim;
  unpad_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(attn.ptr<const bf16>(), out.ptr<bf16>(), n);
  MI_CUDA_CHECK_LAUNCH();
}

void dequantize_convrot(const Tensor& q, const Tensor& scale, int g, const Tensor& out, cudaStream_t stream) {
  constexpr const char* op = "dequantize_convrot";
  require(q.dtype == DType::kI8 && q.rank == 2, op, "expected q i8 [N, K]");
  require(scale.dtype == DType::kF32 && scale.numel() == q.dim(0), op, "expected scale f32 [N]");
  check_bf16(op, "out", out);
  require(out.numel() == q.numel(), op, "out must have N * K elements");
  int digits = 0;
  if (g != 0) {
    for (int v = g; v > 1; v >>= 2) ++digits;
    require((1 << (2 * digits)) == g && q.dim(1) % g == 0, op, "group must be a power of 4 dividing K");
  }
  if (q.numel() == 0) return;
  dequant_kernel<<<grid_for(q.numel(), 256), 256, 0, stream>>>(q.ptr<const std::int8_t>(), scale.ptr<const float>(),
                                                               out.ptr<bf16>(), q.dim(0), q.dim(1), g, digits,
                                                               g ? 1.0f / static_cast<float>(1 << digits) : 1.0f);
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace mi::kernels::vision
