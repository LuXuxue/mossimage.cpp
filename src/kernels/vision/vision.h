// Kernels of the Qwen3.5 vision tower (Qwen3_5VisionModel) that the rest of the engine has no use for: LayerNorm
// with a bias, the two GELUs, and the attention's rotary embedding fused with the head padding FlashAttention needs.
//
// All tensors are contiguous bf16 unless noted. Each kernel computes in fp32 and rounds once to bf16, as torch's bf16
// ops do; they are checked against torch on the same inputs (tests/unit/test_vision_kernels.cu), not bit for bit.
#pragma once

#include <cuda_runtime.h>

#include "core/tensor.h"

namespace mi::kernels::vision {

// The vision tower's attention: 16 heads of 72 dims. FlashAttention has no 72-dim kernel, so q, k and v are padded
// with zeros to 128 dims (zeros add nothing to q·k and give zero output columns); the softmax scale stays 1/sqrt(72).
inline constexpr int kHeads = 16;
inline constexpr int kHeadDim = 72;
inline constexpr int kPaddedHeadDim = 128;

// nn.LayerNorm(eps) over the last dim: out = (x - mean) / sqrt(var + eps) * weight + bias, biased variance.
// x, out: [N, C]; weight, bias: [C]. out may alias x.
void layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps, const Tensor& out,
                cudaStream_t stream);

// In place: x = gelu(x, approximate="tanh") (ACT2FN["gelu_pytorch_tanh"]), any shape.
void gelu_tanh(const Tensor& x, cudaStream_t stream);
// In place: x = gelu(x) with erf (nn.GELU()), any shape.
void gelu_erf(const Tensor& x, cudaStream_t stream);

// qkv: [N, 3 * 1152], the qkv projection (bias included) viewed as [N, 3, 16, 72]. cos, sin: f32 [N, 72].
// q_out, k_out, v_out: [N, 16, 128]: apply_rotary_pos_emb_vision to q and k (fp32: x * cos + rotate_half(x) * sin,
// rounded to bf16), v copied, columns 72..127 zero.
void qkv_rope_pad(const Tensor& qkv, const Tensor& cos, const Tensor& sin, const Tensor& q_out, const Tensor& k_out,
                  const Tensor& v_out, cudaStream_t stream);

// attn: [N, 16, 128] -> out: [N, 16 * 72], dropping the padding.
void unpad_heads(const Tensor& attn, const Tensor& out, cudaStream_t stream);

// INT8 ConvRot weights to bf16 on the device: out[r, :] = (q[r, :] * scale[r]) . H_g per group of g columns (H_g the
// normalized Hadamard kron^n(H4) / sqrt(g), its own inverse; quant::dequantize_rows on the GPU). q: i8 [N, K];
// scale: f32 [N]; out: bf16 [N, K]; g a power of 4 dividing K, or 0 for no rotation.
void dequantize_convrot(const Tensor& q, const Tensor& scale, int g, const Tensor& out, cudaStream_t stream);

}  // namespace mi::kernels::vision
