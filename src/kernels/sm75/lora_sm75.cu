// sm_75 LoRA epilogue: out = bf16(bf16(out) + bf16(delta)) in fp32, on ordinary CUDA cores.
// Used by the runtime side branch; the INT8 base GEMM and the fp16 cuBLASLt delta GEMM stay unchanged.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "core/tensor.h"
#include "kernels/sm75/sm75.h"

namespace mi::kernels::sm75 {
namespace {

__global__ void add_delta_kernel(__nv_bfloat16* __restrict__ out, std::int64_t ldo,
                                 const __nv_bfloat16* __restrict__ delta, std::int64_t ldd, std::int64_t rows,
                                 std::int64_t cols) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * cols) return;
  const std::int64_t r = i / cols, c = i % cols;
  const float v = __bfloat162float(out[r * ldo + c]) + __bfloat162float(delta[r * ldd + c]);
  out[r * ldo + c] = __float2bfloat16_rn(v);
}

}  // namespace

void add_delta_bf16(const Tensor& out, const Tensor& delta, cudaStream_t stream) {
  const std::int64_t rows = out.dim(0), cols = out.dim(1);
  if (rows == 0 || cols == 0) return;
  const std::int64_t total = rows * cols;
  const int blocks = static_cast<int>((total + 255) / 256);
  add_delta_kernel<<<blocks, 256, 0, stream>>>(
      out.ptr<__nv_bfloat16>(), out.stride[0], delta.ptr<const __nv_bfloat16>(), delta.stride[0], rows, cols);
}

}  // namespace mi::kernels::sm75
