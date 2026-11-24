# MOSS-Image HIP backend

Native HIP inference for MOSS-Image on AMD RDNA3/RDNA4 GPUs (`HIP_TARGET`: gfx1100, gfx1103,
gfx1200, gfx1201), built with ROCm 7 (TheRock on Windows). It is the AMD counterpart of the CUDA
and Vulkan backends.

## Features

- `--tier high`: INT8 ConvRot DiT with native gfx11 WMMA intrinsics
- Runtime LoRA on all four DiT GEMMs; the delta is fused into the base GEMM's
  fp32 accumulator before bf16 rounding
- INT8 SageAttention-style attention (QK and PV, u8 probabilities)
- Accepts the `moss-infer generate` flag set, UTF-8 prompts, single-file
  `--dit`/`--te`/`--vae` overrides, varlen batches and high-resolution support.
  Flags that need CUDA-only machinery are accepted but print a no-effect notice.

## Build (Windows / TheRock)

```bat
hip\build-win.bat
```

The script loads the MSVC environment (via the VS installer), discovers the SDK through
`rocm-sdk path --root` (or `ROCM_PATH`), configures CMake with Ninja, and copies the
executable plus the runtime DLLs into `hip\dist\`. No machine-specific SDK path is
stored. `HIP_TARGET` defaults to the development GPU's single architecture and can be
overridden for a release build. Intermediate files stay in the git-ignored `hip\build-win\`.

Relevant CMake inputs: the TheRock `clang-cl.exe`, `HIP_TARGET`, `ROCM_PATH` pointing at
the SDK's devel root, and `--rocm-device-lib-path=<ROCM_PATH>\lib\llvm\amdgcn\bitcode`.
The runtime DLLs `amdhip64*.dll`, `rocm_kpack.dll` and `amd_comgr.dll` from the SDK must
sit next to the exe; the versioned `amdhip64` name depends on the SDK.

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
  The V padding is left uninitialized: out-of-range keys get zero probability,
  so those bytes never contribute to PV.
- The text encoder's int8 ConvRot weights are dequantized to host bf16 once at
  construction. The reference dense Hadamard multiply is O(g^2) and dominates
  startup; `dequant_fast.h` applies the same H4-based transform as radix-4
  butterfly stages (O(g log g)), writes bf16 directly and processes tensors on
  several threads. The fast transform differs from the reference at a small
  fraction of bf16 weight values; set `MI_HIP_TE_EXACT_DEQUANT` (any value) for
  bit-exact but much slower weights. Projection GEMMs use the LDS-staged tiled
  kernel on row-major weights by default; `MI_HIP_TE_PACKED_GEMM` (any value)
  selects the original packed rocWMMA path. "text encoder load" and "text
  encoder" are reported separately.
  The int8 ConvRot layer weights default to staying int8 on the device:
  `te_dequant.hip` runs the same radix-4 transform there as the host path and
  writes bf16 directly into the per-layer buffer, avoiding the second host bf16
  image and its copy. This is bit-identical to the host dequantization;
  `MI_HIP_TE_DEVICE_DEQUANT=0` restores the host path and uses one fewer
  layer-sized device buffer.
- The VAE decoder's mid attention packs Q/K/V/P into 2 TF32 bf16 pieces by
  default (3 GEMM products instead of 6), with a small decoded-image difference
  from the fp32-accurate path. Set `MI_HIP_VAE_ATTN_PIECES=3` for the exact
  3-piece path; `MI_HIP_VAE_PROFILE` (any value) prints per-op VAE times.
- The DiT attention output head stride is the head dimension (128), not the
  full residual width; using the latter corrupts adjacent token rows.
- The LoRA delta must be added before the base GEMM's bf16 rounding, not to the
  rounded output, to match the CUDA side branch's magnitude.
- Grids that scale with token count are chunked so large images stay within
  driver limits; partial M tiles are padded and launch errors are checked.

## Checks and benchmarks

`CMakeLists.txt` builds the standalone checks and benchmarks by default
(`MOSSINFER_HIP_TOOLS=ON`); the Windows `build-win.bat` currently configures with
them off. On a build with them enabled:

- `hip_int8_gemm_check`, `hip_bf16_gemm_check`, `hip_int8_attn_probe`,
  `hip_attn_probe` — compare kernels against host references
- `hip_int8_gemm_bench`, `hip_bf16_gemm_bench`, `hip_rotate_quantize_bench`,
  `hip_lora_info`, `hip_te_profile`, `hip_dequant_fast_check`,
  `hip_bf16_small_bench` — report kernel throughput, LoRA branch shapes and
  text-encoder timing/accuracy
- `MI_HIP_OP_TIME` (any value) prints a per-op DiT timing breakdown
