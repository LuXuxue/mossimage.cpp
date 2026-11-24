// Host entry point of the high-tier INT8 ConvRot GEMM (int8_gemm.hip). Kept in a header so tools can call it without
// compiling the HIP kernel source into their own translation unit.
#pragma once

#include <hip/hip_runtime_api.h>

#include <cstdint>

#include "hip_block.h"

namespace mi::hip {

void int8_gemm(hip_block::bf16_bits* y, std::int64_t ldy, const std::int8_t* x, std::int64_t ldx, const float* x_scale,
               const std::int8_t* w, const float* w_scale, const float* bias, int m, int n, int k, hipStream_t stream,
               const hip_block::bf16_bits* lora_delta = nullptr, std::int64_t ldl = 0);

}  // namespace mi::hip
