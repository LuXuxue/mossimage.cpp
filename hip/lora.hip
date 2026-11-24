// Small elementwise kernels for the LoRA side branch: add a bf16 delta to a base GEMM's bf16 output, with the same
// round-to-nearest-even and fp32 add as the rest of the backend.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "device_math.h"
#include "lora.h"

namespace mi::hip {
namespace {

using bf16 = hip_block::bf16_bits;
using hip_block::detail::add;
using hip_block::detail::bf;
using hip_block::detail::f;

__global__ void add_kernel(bf16* dst, const bf16* src, std::uint64_t count) {
  const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) dst[i] = bf(add(f(dst[i]), f(src[i])));
}

}  // namespace

void lora_add(bf16* dst, const bf16* src, std::uint64_t count, hipStream_t stream) {
  if (count == 0) return;
  constexpr int kThreads = 256;
  const auto blocks = static_cast<unsigned>((count + kThreads - 1) / kThreads);
  add_kernel<<<blocks, kThreads, 0, stream>>>(dst, src, count);
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) throw std::runtime_error(std::string("lora_add: ") + hipGetErrorString(e));
}

}  // namespace mi::hip
