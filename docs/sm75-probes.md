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

## Implemented: fp32 exact attention on GTX 16xx

`kernels/sm75/attention_f32_sm75.cu` replaces the emulated fp16 MMA attention with chunked cuBLASLt fp32 GEMMs
(QKᵀ and P·V) and a parallel SIMT softmax. All `sm75_check` attention cases (attn128/attn256/big, including GQA,
varlen and bottom-right causal) pass the cosine ≥ 0.9999 and LSE ≤ 2e-4 gates. Measured with `sm75_probe attn`:

| Shape | fp16 MMA (emulated) | fp32 GEMMs | Speedup |
|---|---:|---:|---:|
| 266 tokens, 24 heads, D128 | 2.6 ms | 2.4 ms | 1.09x |
| 4113 tokens, 24 heads, D128 | 350 ms | 211 ms | **1.66x** |
| 4096 tokens, 32/8 GQA, D256 causal | 502 ms | 390 ms | 1.29x |

End-to-end at 1024² (1 step, cached text embedding) the gain is only ~3 % (42.0 -> 40.9 s/step): the DiT weight
streaming and the 12 s tiled VAE decode dominate. `MI_SM75_FP32_ATTN=0` selects the old kernel for A/B measurement.
It needs a larger workspace (`attention::flash_fwd_workspace_bytes(total_q, heads, max_seqlen)`, ~40 MiB at 256²),
and cuBLASLt plan objects are cached for the lifetime of the process because captured CUDA graphs may replay them.

## Implemented: DP4A for q8 attention's QK

`attention_q8_sm75.cu` now has a `kDp4a` path. PTX's m8n8k16 fragment layout is
A[row=group][K=tig*4+i], B[K=tig*4+i][N=group], C[row=group][N=2*tig+e]. The DP4A path therefore stages the whole
query tile in shared memory and each thread sums all 128 K dims of its two query rows against key rows
`2*tig` and `2*tig+1`, with the existing fp16 P·V epilogue unchanged. `MI_Q8_DP4A=0` selects the original MMA path.

| Shape | MMA QK | DP4A QK | Speedup |
|---|---:|---:|---:|
| 4113 tokens, 24 heads, D128 | 256 ms | 183 ms | **1.4x** |
| 266 tokens, 24 heads, D128 | 1.63 ms | 1.24 ms | 1.31x |

The 256² 4-step SFT generation produces **bit-identical latents** to the MMA path and the expected correct image.
An earlier attempt produced a corrupted image: its row/key indexing happened to match the MMA accumulator layout
(`q8map` confirmed `row=group`, `key=2*tig+e`) but its operand staging did not follow the PTX fragment layout, so
scores only matched on iid-random data. The `q8cmp` probe records that mismatch for reference. Because q8 now works
on GTX 16xx, an unset attention resolves to `q8_fp16` there, as on other architectures.

## Implemented: DP4A for the fused INT8 epilogues too

`sm75_probe gemmepi` compares each CUTLASS epilogue (`m8n8k16` emulated MMA) with a DP4A plain GEMM into a temporary
bf16 matrix followed by the existing SIMT `gated_residual` / `situ_glu` kernel, at M=256, N=1536, K=3072:

| Epilogue | CUTLASS | DP4A + SIMT | Speedup | Output differences |
|---|---:|---:|---:|---:|
| gated_residual | 2.690 ms | 1.917 ms | 1.40x | 0 |
| situ_glu | 2.667 ms | 1.718 ms | 1.55x | 0 |

`kernels/sm75/int8_gemm_sm75.cu` now uses that split path for both epilogues (one scratch buffer per stream).
A no-LoRA 256² 1-step run with pinned host weights went from 2.30 to 1.87 s/step, with a **bit-identical latent**.
The runtime-LoRA side branch already used the same split path.

## End-to-end speedups beyond the kernels

At 256², 4 steps, seed 42, the same prompt and a warm embedding cache:

| Configuration | s/step | Note |
|---|---:|---|
| original (page cache, merged LoRA, exact) | 3.54 | before this branch |
| q8 default, `balanced` (page cache) | 2.51 | bit-identical PNG to explicit q8 |
| q8 default, `max_speed` (pinned host) | 2.07 | bit-identical PNG to `balanced` |

`--mode max_speed` keeps 5 of 42 DiT blocks resident in VRAM and copies the rest from a pinned host copy; the
one-time load grows (39 s vs 9 s) and host RAM rises to ~7 GiB, so it pays off only for repeated generations.
At 1024² (1 step, with the SFT LoRA) `max_speed` gives
34.93 s/step versus 40.82 s/step for the earlier balanced q8-DP4A run (~17 %); the 11.7 s tiled fp32 VAE decode is
now the largest single stage.

On a 4 GiB card the q8 pre-pass does **not** fit above ~5000 latent tokens: at 1536x1024 (6144 tokens) its plan needs
a 1.77 GiB arena against 1.74 GiB available, while the chunked exact attention needs 1.68 GiB. An unset attention
therefore falls back to `exact` there (verified: 1536x1024 completes in 71 s/step with exact, versus 71 s/step when
`exact` was requested explicitly); the QK DP4A path is still the default at 1024² and below.

Forcing it with `--vram-reserve 256` does let the q8 plan start, but the run then spills into WDDM shared GPU memory:
the pinned-host copy rate collapses (the report showed 0.1 GB/s and 164 stalls even when the copy was nominally
hidden), peak device memory reaches 2.94 GiB, and the process **crashes with 0xC0000005 during teardown after writing
a correct PNG**. The same crash occurs with `exact` at that reserve, so it is the memory oversubscription, not the q8
kernel. Keeping the default 384 MiB reserve caps the peak at 2.83 GiB and exits cleanly. The planner's dedicated-VRAM
cap and `--vram-reserve` are therefore deliberate: shared GPU memory is system RAM, not free VRAM.

### Diagnostic soft-VRAM mode

`MI_SOFT_VRAM_BUDGET=1` is an opt-in experiment: it disables the large-latent exact fallback and makes `planner::plan`
return the leanest plan with a warning instead of throwing `BudgetError`. It is never the default. Measured at
1536x1024, 1 step, seed 42, unset attention (resolves to `q8_fp16`), pinned/page-cache weights as chosen by the plan:

| Reserve | Weight source | s/step | Device peak | Exit | Note |
|---:|---|---:|---:|---:|---|
| default 384 | page cache | 58.93 | 2.89 GiB | 0 | warning; 0.1 GB/s stream, 41 stalls (1.09 s); total 85.6 s |
| 320 | pinned host | 58.23 | 2.91 GiB | 0 | q8 fits the 1.79 GiB cap, no warning |
| 288 | pinned host | 58.30 | 2.94 GiB | 2 of 3 runs crash | `0xC0000005` at teardown after a correct PNG |
| 256 | pinned host | 57.94 | 2.94 GiB | crash | same failure as the earlier explicit-q8 run |

The safe exact fallback at this resolution is 71.4 s/step with a 2.83 GiB peak and exits cleanly, so the soft q8 path
is **~18 % faster per step**. Step time is almost independent of the reserve; the spill boundary is a device peak of
about 2.9 GiB, above which WDDM shared-memory teardown becomes flaky (2.94 GiB crashed in 2/3 runs, once exited 0).
The 384 MiB default also changes the planner's choice to page-cache weights (5 s load vs 39 s for the pinned copy),
making its end-to-end total the lowest of these runs despite the slow stream. The switch is for measurement only: it
trades a small speedup for shared-memory stalls and a nondeterministic crash on exit.

## Explored but not kept

- **DP4A for q8 attention's P·V.** `sm75_probe pvbench` times the core P[64,32]·V[32,128] matmul: 0.024 ms for the
  emulated fp16 `m16n8k8` MMA versus 0.009 ms for a native DP4A loop (**2.7x** on the matmul alone). Integrating it
  needs V quantised to int8 with one scalar scale per head (the dequant product must stay a single multiply) plus a
  cross-lane exchange of P quartets — the same fragment-layout trap that broke the first QK attempt — and the q8
  scores would change. Since P·V is only part of the attention call and attention is a small share of end-to-end
  time (2.7 s/step of 35 s at 1024², <0.1 s/step at 256²), the accuracy risk is not worth it. QK stays DP4A and P·V
  stays fp16 MMA.
- **Lower register pressure in the VAE convs.** A 256-thread variant of `conv2d_f32_sm75` (1 pixel × 16 channels per
  thread instead of 2 × 16, so 64 accumulators instead of 128) was bit-exact once its tile mapping was fixed, but the
  lost weight reuse made the 1024² tiled decode **slower**: 14.70 s vs 11.70 s. Reverted; the default 128-thread tile
  stays. A 256-thread variant that assigned 8 channels per thread initially dropped the tile's upper 8 rows (black
  image) — the new `vae_conv` check with a multi-block 8x16 tile in `sm75_check` would have caught it, but the
  variant is not faster, so neither it nor the extra case is kept.
- **Fused DiT blocks / CUDA graphs.** Unavailable on sm_75: the fused bf16 CUTLASS GEMMs need sm_80+ tensor cores, and
  `--dit-blocks fused` / `fused_graphs` fall back to `op_by_op`.
- **VAE fp16x2 convs.** The hardware ceiling is ~2x the fp32 rate, but rounding operands to fp16 changes many output
  bytes. The current decoder is a reference-precision path; making it lossy needs a separate quality gate and is not
  part of this branch.
