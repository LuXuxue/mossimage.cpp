# GTX 1650 arithmetic probes

`tools/sm75_probe.cu` (target `sm75_probe`, built with `MOSSINFER_SM75=ON`, not installed) measures the
arithmetic that the sm_75 backend actually uses. Every probe checks its result against a host reference
and times device work with CUDA events after warmup. Run `sm75_probe [info|bw|alu|gemm|dp4a|attn|all]`;
`sm75_probe gemm check` / `sm75_probe dp4a check` run the correctness variants (including tile tails).

Measured on a GTX 1650 (TU117, sm_75, 14 SMs, 4 GiB GDDR5, **no tensor cores**). Clocks drift under
sustained load, so the raw ALU rates vary by ~25 % between runs; the ratios are stable.

## Hardware ceiling

| Quantity | Measured | Note |
|---|---|---|
| SM clock / DRAM clock | 1665 / 4001 MHz | boost; power limit 75 W |
| DRAM bandwidth (vectorised read+write) | ~106 GB/s | 128-bit bus, 128 GB/s theoretical |
| `cudaMemcpy` D2D | ~52 GB/s | H2D/D2H ~7 GB/s |
| FP32 FMA (dependent chain) | 0.86–1.12 TFLOPS | predicted peak 1.49 TFLOPS |
| FP16x2 FMA (`fma.rn.f16x2`) | 1.83–2.36 TFLOPS | ~2x FP32 despite having no tensor cores |
| INT8 `__dp4a` | 3.33–4.30 TOPS | 4 MACs per instruction, native on Turing |
| INT32 `mad` | 0.63–0.93 TOPS | |

## GEMM throughput (TFLOPS equivalent; larger is better)

| Shape (M,N,K) | cuBLASLt fp32 | cuBLASLt fp16 | engine bf16→fp16 | CUTLASS INT8 MMA | hand DP4A 64x64 |
|---|---:|---:|---:|---:|---:|
| 256 × 3072 × 3072 | 0.88 | 0.31 | 0.31 | 1.30 | **1.80** |
| 1024 × 3072 × 8192 | 0.65 | 0.35 | 0.31 | 1.30 | **1.93** |
| 256 × 3072 × 64 (LoRA) | **1.59** | 0.30 | 0.24 | – | – |
| 4096 × 3072 × 64 (LoRA) | **1.46** | 0.33 | 0.29 | – | – |

Correctness (128 × 512 × 1024, host reference):
- cuBLASLt fp32: cosine 1.000000000, max abs 4.8e-7.
- cuBLASLt fp16: cosine 0.99999994, max relative 0.21 only where the fp32 reference is ~1e-3 (fp16 input rounding).
- engine bf16→fp16: cosine 0.9999972.
- all three hand-written DP4A variants: cosine 0.9999986, max relative 0.0039 — acceptable, with the exact same
  int32 accumulation and dequantisation order `(acc * x_scale) * w_scale` as the engine's INT8 epilogue.
  Verified for tile tails (M=130, N=520) as well.

## Attention (the sm_75 exact kernel, `mma.sync m16n8k8` executed on CUDA cores)

| Shape | Time | Throughput |
|---|---:|---:|
| 266 tokens, 24 heads, D128 | 2.0 ms | 0.21 TFLOPS |
| 4113 tokens, 24 heads, D128 (1024² DiT) | 352 ms | 0.30 TFLOPS |
| 4096 tokens, 32/8 GQA, D256 causal (text encoder) | 505 ms | 0.27 TFLOPS |

## Answers

**Can hand-written instructions beat the library here? Yes, measurably.** The user-reported AMD/iGPU pattern
reproduces on NVIDIA's no-tensor-core Turing:

- The CUTLASS INT8 ConvRot GEMM uses Turing's `m8n8k16` INT8 MMA. On TU117 the SM decodes `mma.sync` on ordinary
  arithmetic units. A hand-written tile using the native `__dp4a` instruction with a 64x64x128 tile and real
  weight reuse is **1.4–1.5x faster** than CUTLASS (1.80–1.93 vs 1.30 TFLOPS). A naive 8x32 tile is 4x *slower*,
  so the win comes from both the instruction and the tiling, not from `__dp4a` alone.
- Conversely, a naive hand-written fp32 SIMT GEMM (0.27–0.29 TFLOPS) is much slower than cuBLASLt fp32
  (0.65–0.88 TFLOPS): cuBLASLt's fp32 kernels are well optimised for this architecture. So "hand-written is
  always faster" is false; it wins specifically where the library's path depends on an emulated instruction.

**Where the remaining speedups are:**
1. **INT8 ConvRot GEMMs → DP4A.** Largest and most certain win: 1.4–1.5x on the DiT linears, with up to 4.3 TOPS
   of DP4A headroom (currently ~45 % of it). Implementation note: the probe stores weights as [K,N] for coalesced
   loads; the engine stores [N,K], so either transpose INT8 blocks once when they are loaded/streamed or write a
   shared-memory-transposing DP4A kernel. The epilogues (bias, gated residual, SiTU-GLU) can stay byte-identical.
2. **Runtime LoRA Δ → fp32 cuBLASLt.** The side branch currently computes its factors with fp16 cuBLASLt at
   0.29–0.33 TFLOPS. With K = rank (32/64), cuBLASLt fp32 reaches 1.46–1.59 TFLOPS — a **~4.5x** speedup for
   the same fp32 accumulation, since fp16→fp32 conversion of the tiny A/B factors is exact and essentially free.
3. **8-bit attention → DP4A QK.** The existing sm75 `q8_fp16` attention uses an emulated `m8n8k16` MMA for QKᵀ,
   which is why it is refused on GTX 16xx. Replacing just that QK MMA with `__dp4a` removes the emulation penalty
   (at 1024², QKᵀ alone is ~104 GFLOP, i.e. ~58 ms at 1.8 TFLOPS instead of a large fraction of the current
   352 ms attention call), and could make 8-bit attention viable again on TU116/TU117.
4. Exact attention is at 0.21–0.30 TFLOPS against a 1.83–2.36 TFLOPS fp16x2 ALU ceiling. Rewriting it as chunked
   fp32 cuBLASLt matmuls plus a SIMT softmax is plausible, but a naive hand-written fp32 GEMM did *not* beat
   cuBLASLt here; this direction needs a carefully vectorised kernel and is the least certain of the four.

The fp16 cuBLASLt path being 2.8–4.7x slower than fp32 on this GPU is a library/architecture mismatch, not a
numerical necessity: fp32 uses the full CUDA-core rate while the fp16 kernels here run at ~0.3 TFLOPS despite an
fp16x2 ALU ceiling above 1.8 TFLOPS.
