#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "device_math.h"
#include "te_dequant.h"

namespace mi::hip::te_dequant {
namespace {

using namespace hip_block::detail;

constexpr int kWave = 32;
constexpr int kGroup = 256;  // ConvRot's Hadamard group

struct DequantArgs {
  const std::int8_t* q;
  const float* scale;
  std::uint16_t* out;
  int k, groups;
};

// One wave per 256-element group. The four stages read a disjoint butterfly's four values before writing them, so no
// barriers are needed inside a stage; the group's scale multiply is the same order as dequant_fast.h's host code.
__global__ __launch_bounds__(kWave) void dequant_i8_kernel(const DequantArgs a) {
  __shared__ float v[kGroup];
  const int t = static_cast<int>(threadIdx.x);
  const int groups_per_row = a.k / kGroup;
  for (int g = static_cast<int>(blockIdx.x); g < a.groups; g += static_cast<int>(gridDim.x)) {
    const float sc = a.scale[g / groups_per_row];
    const std::int64_t at = static_cast<std::int64_t>(g) * kGroup;
    HIP_BLOCK_UNROLL
    for (int i = 0; i < kGroup / kWave; ++i) {
      const int j = t * (kGroup / kWave) + i;
      v[j] = static_cast<float>(a.q[at + j]) * sc;
    }
    __syncthreads();
    for (int step = 1; step < kGroup; step <<= 2) {
      // 64 butterflies per stage, two per lane.
      HIP_BLOCK_UNROLL
      for (int b = t * (kGroup / (4 * kWave)); b < t * (kGroup / (4 * kWave)) + kGroup / (4 * kWave); ++b) {
        const int base = (b / step) * 4 * step, off = b % step;
        const float w0 = v[base + off], w1 = v[base + step + off];
        const float w2 = v[base + 2 * step + off], w3 = v[base + 3 * step + off];
        const float p = w0 - w3, q = w0 + w3, r = w1 + w2, s = w1 - w2;
        v[base + off] = 0.5f * (p + r);
        v[base + step + off] = 0.5f * (q + s);
        v[base + 2 * step + off] = 0.5f * (q - s);
        v[base + 3 * step + off] = 0.5f * (r - p);
      }
      __syncthreads();
    }
    HIP_BLOCK_UNROLL
    for (int i = 0; i < kGroup / kWave; ++i) {
      const int j = t * (kGroup / kWave) + i;
      a.out[at + j] = bf(v[j]);
    }
    __syncthreads();
  }
}

}  // namespace

void dequant_i8(const std::int8_t* q, const float* scale, std::uint16_t* out, int n, int k, hipStream_t stream) {
  if (n <= 0 || k <= 0) return;
  const int groups = n * (k / kGroup);
  const DequantArgs a{q, scale, out, k, groups};
  dequant_i8_kernel<<<dim3(4096), dim3(kWave), 0, stream>>>(a);
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) throw std::runtime_error(std::string("te_dequant: ") + hipGetErrorString(e));
}

}  // namespace mi::hip::te_dequant
