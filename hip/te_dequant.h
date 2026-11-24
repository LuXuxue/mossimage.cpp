// Device-side dequantization of the text encoder's int8 ConvRot weights.
//
// Each row of g=256 int8 values is scaled and multiplied by H_256 = kron^4(H4) as four radix-4 butterfly stages,
// reproducing dequant_fast.h's host transform operation for operation and rounded once to bf16 at the end. The int8
// weights stay on the device (no 2x host bf16 image and no 18 GiB host-to-device copy of dequantized weights); only the
// scales are uploaded once at construction.
#pragma once

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime_api.h>

namespace mi::hip::te_dequant {

// q: [n, k] int8, scale: [n] fp32, out: [n, k] bf16 bits; k a multiple of 256.
void dequant_i8(const std::int8_t* q, const float* scale, std::uint16_t* out, int n, int k, hipStream_t stream);

}  // namespace mi::hip::te_dequant
