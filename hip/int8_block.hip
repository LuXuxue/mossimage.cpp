// The high tier's INT8 ConvRot DiT block (int8_block.h): the four linears (qkv, o, up_gate, down) as int8 x int8 ->
// int32 -> bf16 with per-row activation and weight scales, the activations rotated+quantized per row before each linear,
// and the norms, modulation, RoPE, attention and SiTU-GLU in bf16. The op order and the bf16 rounding points follow the
// engine's all-INT8 block (models/moss_dit/dit_block_quant, no LoRA), which is hip_block::run_block_op with the four
// GEMMs swapped for int8 GEMMs plus a bf16 epilogue.
#include "int8_block.h"

#include <hip/hip_runtime.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "device_math.h"
#include "dit_kernels.h"
#include "runtime.h"
#include "int8_attn.h"
#include "int8_gemm.h"
#include "rotate_quantize.hip"

namespace mi::hip {
namespace {

using namespace hip_block;
using detail::add;
using detail::bf;
using detail::f;
using detail::mul;

constexpr std::int64_t kDim = 3072;
constexpr std::int64_t kQkv = 3 * 3072;
constexpr std::int64_t kFfn = 8192;
constexpr int kHeads = 24;
constexpr std::int64_t kHead = 128;

// x[m, n] += bf16(out[m, n] * gate[n]) in place (the gated residual after an int8 GEMM).
__global__ void gate_residual_kernel(bf16* x, const bf16* out, const bf16* gate, std::int64_t m, int n) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= m * n) return;
  const int c = static_cast<int>(i % n);
  x[i] = bf(add(f(x[i]), mul(f(out[i]), f(gate[c]))));
}

}  // namespace

void int8_gate_residual(bf16* residual, std::int64_t ldr, const bf16* out, const bf16* gate, std::int64_t m, int n,
                        hipStream_t stream) {
  const std::int64_t total = m * n;
  const int kThreads = 256;
  gate_residual_kernel<<<dim3((total + kThreads - 1) / kThreads), dim3(kThreads), 0, stream>>>(residual, out, gate, m, n);
  check_launch("int8_gate_residual");
}

void run_int8_block_op(int op, const Int8BlockWeights& w, const BlockBuffers& b, const Int8Buffers& ib, hipStream_t stream,
                       const bf16* lora_delta, std::int64_t ldl) {
  const std::int64_t tokens = b.tokens;
  const auto mod = [&](int i) { return b.mod + static_cast<std::int64_t>(i) * kDim; };
  switch (op) {
    case 0:  // norm1 + modulate -> y (bf16)
      rms_norm_modulate(b.x, kDim, w.norm1, mod(1), mod(0), b.y, kDim, tokens, kDim, stream);
      break;
    case 1:  // qkv = int8_gemm(rotate_quantize(y))
      rotate_quantize(b.y, kDim, tokens, kDim, ib.i8_y, kDim, ib.scale, stream);
      int8_gemm(b.qkv, kQkv, ib.i8_y, kDim, ib.scale, w.qkv.q, w.qkv.scale, nullptr,
                static_cast<int>(tokens), static_cast<int>(kQkv), static_cast<int>(kDim), stream, lora_delta, ldl);
      break;
    case 2:  // qk norm + rope (in place on qkv)
      qk_norm_rope(b.qkv, kQkv, tokens, kHeads, w.q_norm, w.k_norm, b.rope, kHead, stream);
      break;
    case 3:  // attention: int8 QK + int8 PV, comfy-kitchen's gfx1103 SageAttention path
      int8_attn_quantize(b.qkv, kQkv, ib.attn, static_cast<int>(tokens),
                         static_cast<std::size_t>(ib.attn.padded) * kHeads * kHeadDim, stream);
      int8_attn_run(ib.attn, b.attn, kDim, b.max_seqlen,
                    static_cast<float>(1.0 / std::sqrt(static_cast<double>(kHead))), stream);
      break;
    case 4:  // x += int8_gemm(rotate_quantize(attn)) * gate_msa
      rotate_quantize(b.attn, kDim, tokens, kDim, ib.i8_y, kDim, ib.scale, stream);
      int8_gemm(b.x, kDim, ib.i8_y, kDim, ib.scale, w.o.q, w.o.scale, nullptr,
                static_cast<int>(tokens), static_cast<int>(kDim), static_cast<int>(kDim), stream, lora_delta, ldl,
                mod(2), b.x, kDim);  // gate+residual fused: no gemm_out round trip
      break;
    case 5:  // norm2 + modulate -> y (bf16)
      rms_norm_modulate(b.x, kDim, w.norm2, mod(4), mod(3), b.y, kDim, tokens, kDim, stream);
      break;
    case 6:  // ffn = situ_glu(int8_gemm(rotate_quantize(y)))
      rotate_quantize(b.y, kDim, tokens, kDim, ib.i8_y, kDim, ib.scale, stream);
      int8_gemm(ib.ug, kFfn * 2, ib.i8_y, kDim, ib.scale, w.up_gate.q, w.up_gate.scale, nullptr,
                static_cast<int>(tokens), static_cast<int>(kFfn * 2), static_cast<int>(kDim), stream, lora_delta, ldl);
      situ_glu(ib.ug, kFfn * 2, b.ffn, kFfn, tokens, kFfn, stream);
      break;
    default:  // x += int8_gemm(rotate_quantize(ffn)) * gate_mlp
      rotate_quantize(b.ffn, kFfn, tokens, kFfn, ib.i8_ffn, kFfn, ib.scale, stream);
      int8_gemm(b.x, kDim, ib.i8_ffn, kFfn, ib.scale, w.down.q, w.down.scale, nullptr,
                static_cast<int>(tokens), static_cast<int>(kDim), static_cast<int>(kFfn), stream, lora_delta, ldl,
                mod(5), b.x, kDim);  // gate+residual fused: no gemm_out round trip
      break;
  }
}

}  // namespace mi::hip
