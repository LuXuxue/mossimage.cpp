# sm_75 full-pipeline precision and headroom (GTX 1650 / TU117)

Measured on a GTX 1650 4 GiB (14 SMs, no tensor cores). Rates come from `sm75_probe`, end-to-end times from
`moss-infer generate`; the methodology and raw numbers are in [sm75-probes.md](sm75-probes.md).

## Hardware ceilings

| Operation | Peak on this GPU | Note |
|---|---:|---|
| FP32 FMA | 1.49 TFLOPS | dependent-chain measurement 0.86–1.12 |
| FP16×2 FMA | 1.83–2.36 TFLOPS | ordinary CUDA cores, no tensor cores |
| INT8 DP4A | 3.33–4.30 TOPS = 6.7–8.6 TFLOPS-equivalent | 4 MACs/instruction |
| DRAM / pinned H2D / page cache / cold disk | 106 / 11.0 / 6.8 / 0.2 GB/s | |

"Percent of peak" below compares the measured equivalent FLOP/s to the relevant column; DP4A numbers are doubled
from MAC/s to FLOP/s for an apples-to-apples comparison.

## Stage-by-stage

| Stage | Arithmetic | Measured rate | % of relevant peak | Time share |
|---|---|---:|---:|---|
| DiT qkv / o / up_gate / down (42 blocks) | INT8 ConvRot → native DP4A, fp32 int32 accumulation, bf16 output | 1.80–1.93 TFLOPS | **21–29 %** of DP4A | 55–70 % of denoise |
| Runtime LoRA A/B/t/Δ | fp32 cuBLASLt (K = rank 32/64), fp32 Δ added to bf16 base | 1.46–1.59 TFLOPS | **98–107 %** of FP32 | small (saturated) |
| DiT exact attention (opt-in) | fp32 cuBLASLt GEMMs + parallel SIMT softmax | 0.99 TFLOPS @4113 tok (211 ms) | 66 % of FP32 | ~15 % of denoise |
| q8 attention QK (default) | int8 × int8 → int32 DP4A | ~1.8 TFLOPS | 21–27 % of DP4A | ~30 % of q8 attention |
| q8 attention P·V (default) | emulated `m16n8k8` fp16 MMA, fp32 accum | ~0.8 TFLOPS | 35–44 % of FP16×2 | ~70 % of q8 attention |
| Text encoder GEMMs | INT8 ConvRot DP4A, same as DiT | 1.80–1.93 TFLOPS | 21–29 % of DP4A | ~2.2 s when not cached |
| Text encoder attention | fp32 GEMMs + SIMT softmax, D256 causal | 0.35 TFLOPS (390 ms @4096 tok) | 23 % of FP32 | part of the 2.2 s |
| VAE convolutions (decode) | fp32 SIMT, NHWC, 8×16×32 tile, 3×3 | 0.23 TFLOPS @1024² (11.7 s) | **15 %** of FP32 | 20–25 % of a 1024² run |
| VAE mid attention | fp32 SIMT online softmax, D384 | not separately timed | low | small |
| Weight streaming | pinned H2D / page cache / disk | 11.0 / 6.8 / 0.2 GB/s | ~100 % of achieved H2D | 4.76 GB/step, 0.43 s when pinned |
| Norm / RoPE / quantise / elementwise | SIMT, memory bound | ~106 GB/s ceiling | n/a | few percent |

Work per DiT step: 0.2265 GFLOP/token/block × 266 tokens × 42 blocks = 2.53 TFLOP at 256²;
× 6144 tokens = 58.4 TFLOP at 1536². Attention adds 208 GFLOP/call at 4113 tokens (464 GFLOP at 6144).
The decoder is ~2680 GFLOP at 1024² (~1340 GMAC); `up_blocks.1` alone is ~54 % of conv MACs and tiled decoding
recomputes 1.4–2.4× at large sizes.

## Where the time actually goes

- **256², 4 steps, `max_speed`:** ~50 s total — load 38 s (76 %), denoise 8.1 s (16 %), text 2.2 s (4 %),
  decode 0.45 s (1 %). Within a 2.03 s denoise step the DP4A GEMMs are ~1.4 s, attention ~0.05 s, the rest is
  LoRA/elementwise. The first generation is I/O bound, not compute bound.
- **1536², 1 step, soft VRAM, page-cache weights:** ~85 s total — load 5 s, denoise 58–63 s (70 %),
  decode 20 s (23 %), text 2.2 s (3 %). Within denoise the GEMMs are ~32 s, q8 attention ~17 s, LoRA and
  elementwise ~9 s. This is the regime where kernel efficiency matters.

## Acceleration potential, largest first

1. **VAE convolutions** — 15 % of FP32 peak and ~20 s at 1536²; the single biggest remaining pool. A lossy fp16×2
   path has a ~2× ALU ceiling, and Winograd F(2×2,3×3) removes ~2.25× of the MACs, but both change output bytes and
   need a separate quality gate. A bit-exact tiling/occupancy pass is safer but smaller. Reducing tiled-decode
   recompute (1.4–2.4×) would save up to ~30 % of decode.
2. **DiT INT8 GEMMs** — 21–29 % of DP4A peak, i.e. 3.4–4.8× headroom in principle. The current 64×32×128 tile
   already beats CUTLASS's emulated MMA 1.4–1.5×; a naive 8×32 tile was 4× slower, so further gains require careful
   shared-memory/occupancy tuning, not just `__dp4a`.
3. **q8 P·V** — a native DP4A loop is 2.7× faster than the emulated fp16 MMA on the matmul alone, but needs scalar
   per-head V quantisation plus cross-lane P exchange (the fragment-layout trap from the first QK attempt) and
   changes q8 scores. At 1024² this is ~2.7 s/step of a 35 s step, so the accuracy risk currently outweighs it.
4. **Text encoder attention** — 0.35 vs ~1.0 TFLOPS for the DiT equivalent; ~~1.3–2× possible with better
   chunking, but only when the embedding cache misses.
5. **Exact attention** — already at 66 % of FP32 peak; little headroom.
6. **LoRA / weight streaming** — LoRA is saturated at the FP32 FMA peak and the H2D copy is at the measured link
   rate. Only algorithmic changes (fewer bytes, fewer calls) help here.

## Possible algorithmic fusions

| Fusion | What it removes | Estimated gain |
|---|---|---:|
| QKV GEMM epilogue → qk_norm + RoPE + int8 quantisation | two passes over `[M,3D]` qkv per block (~9.5 GB at 1536²) | ~0.1 s/step at 1536², more launch overhead saved at 256² |
| Compute LoRA Δ first, fold Δ add into the base DP4A GEMM epilogue | one elementwise pass over `[M,N]` per site (~9.5 GB at 1536²) | ~0.1 s/step at 1536² |
| Fuse the exact-attention softmax into the QK GEMM epilogue | score slab traffic (3.6 GB at 6144 tokens) | ~0.03–0.1 s/step |
| Fuse SiLU into VAE conv epilogues; keep GroupNorm separate (needs a reduction) | one pass over each activation tensor (~30 × 1.6 GB at 1024²) | ~0.4 s of an 11.7 s decode |
| Replace VAE mid attention with chunked fp32 GEMMs + SIMT softmax (the sm70 approach) | redundant K/V reads | small; only 16k-token mid block |
| `--step-cache fbcache` | whole DiT calls when the residual barely changes | up to 25–50 % of denoise on 4-step SFT, with a small image change |

Fusions mostly save memory traffic; at 256² the run is dominated by a one-time weight load, and at 1536² by DP4A
GEMM and VAE arithmetic. So the two changes with real end-to-end leverage remain **VAE convolution arithmetic** and
**DiT GEMM tiling**, not more kernel fusion.
