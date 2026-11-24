# MOSS-Image HIP backend

Native HIP inference for MOSS-Image on AMD RDNA3 GPUs (gfx11xx), built with the
TheRock SDK. It is the AMD counterpart of the CUDA and Vulkan backends.

## Features

- `--tier high`: INT8 ConvRot DiT with native gfx11 WMMA intrinsics
- Runtime LoRA on all four DiT GEMMs; the delta is fused into the base GEMM's
  fp32 accumulator before bf16 rounding
- INT8 SageAttention-style attention (QK and PV, u8 probabilities)
- Full `moss-infer generate` CLI surface, UTF-8 prompts, single-file
  `--dit`/`--te`/`--vae` overrides, varlen batches and high-resolution support

## Build (Windows / TheRock)

```bat
hip\build-win.bat
```

The script loads the MSVC environment, discovers the SDK through
`rocm-sdk path --root` (or `ROCM_PATH`/`ROCM_HOME`), configures CMake with Ninja,
and copies the executable plus the runtime DLLs into `hip\dist\`.
No machine-specific SDK path is stored. Intermediate files stay in the
git-ignored `hip\build-win\`.

Relevant CMake inputs: the TheRock `clang-cl.exe`, `HIP_TARGET=gfx1103`,
`ROCM_PATH` pointing at the `_rocm_sdk_devel` directory, and
`--rocm-device-lib-path=<ROCM_PATH>\lib\llvm\amdgcn\bitcode`. The runtime DLLs
`amdhip64_7.dll`, `rocm_kpack.dll` and `amd_comgr.dll` must sit next to the exe.

## Run

```bat
moss-infer-hip generate --model <bundle> --dit <file> --te <file> --vae <file> ^
  --prompt "..." --tier high --steps 30 --cfg 4 ^
  [--lora <file> --lora-scale 1.0] -o out.png
```

`--model` supplies the tokenizer and configs; the weight files may be single
`.safetensors` files. Flags requiring CUDA-only machinery are accepted but print
an explicit notice instead of being silently ignored. Launcher scripts are kept
outside the repository.

## Implementation notes

- `int8_leaf.h` uses raw `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`
  intrinsics, not rocWMMA: each lane holds a whole 16-byte K-step, and
  accumulator element `e` is row `2*e + lane/16`, column `lane % 16`.
- The INT8 and bf16 GEMMs share an LDS-staged, software-pipelined 128x128 tile;
  shallow K uses a 512-thread 2x2 fragment grid, deep K a 128-thread 4x4 grid.
- Attention evaluates `Sᵀ = K·Qᵀ` and `Oᵀ = Vᵀ·Pᵀ` in 128-query / 64-key tiles,
  keeping u8 probabilities in registers so the online softmax is scalar per
  lane. Q/K/V quantizers derive all varlen offsets from `cu_seqlens` on device.
  The V padding is not zeroed: out-of-range keys get zero probability, and a
  large `hipMemsetAsync` was far more expensive than all quantization kernels.
- The text encoder's int8 ConvRot weights are dequantized to host bf16 once at
  construction. The reference dense Hadamard multiply is O(g^2) and dominates
  startup; `dequant_fast.h` applies the same H4-based transform as radix-4
  butterfly stages (O(g log g)), writes bf16 directly and processes tensors on
  several threads. This cuts construction from ~165 s to ~6 s and `encode()` is
  ~1.5 s. The fast transform differs from the reference at ~0.1% of bf16 weight
  values; set `MI_HIP_TE_EXACT_DEQUANT=1` for bit-exact (but much slower) weights.
  The reported "text encoder load" and "text encoder" timings are separate.
- The DiT attention output head stride is the head dimension (128), not the
  full residual width; using the latter corrupts adjacent token rows.
- The LoRA delta must be added before the base GEMM's bf16 rounding, not to the
  rounded output, to match the CUDA side branch's magnitude.
- Grids that scale with token count are chunked so large images stay within
  driver limits; partial M tiles are padded and launch errors are checked.

## Checks and benchmarks

`build-win.bat` also builds these standalone tools:

- `hip_int8_gemm_check`, `hip_bf16_gemm_check`, `hip_int8_attn_probe`,
  `hip_attn_probe` — compare kernels against host references
- `hip_int8_gemm_bench`, `hip_bf16_gemm_bench`, `hip_rotate_quantize_bench`,
  `hip_lora_info`, `hip_te_profile`, `hip_dequant_fast_check` — report kernel
  throughput, LoRA branch shapes and text-encoder timing/accuracy
- `MI_HIP_OP_TIME=1` prints a per-op DiT timing breakdown
