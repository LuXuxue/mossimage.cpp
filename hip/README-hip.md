# HIP backend for AMD (gfx1103 / 780M)

Native-HIP inference for MOSS-Image on AMD GPUs, built from the **TheRock** SDK
(no full ROCm install required). This is the AMD counterpart of the CUDA
(`moss-infer`) and Vulkan (`moss-infer-vulkan`) backends — it runs on the AMD
iGPU through HIP, not Vulkan.

## Status

| Item | State |
| --- | --- |
| Windows / TheRock build (`moss-infer-hip.exe`) | **Done** — `build-win.bat` configures, builds and copies the runtime DLLs (`amdhip64_7`, `rocm_kpack`, `amd_comgr`) into `dist/` |
| 780M (gfx1103) device detection | **Done** — `--list-devices` shows the device |
| `--dit`/`--te`/`--vae` single-file weights | **Done** — same model files as the CUDA launcher |
| `--tier high` (INT8 ConvRot DiT) | **Done** — INT8 GEMM uses the native gfx11 `v_wmma_i32_16x16x16_iu8`; quantization and every block op verified against CPU references |
| Runtime LoRA (`--lora`, `--lora-scale`) | **Done** — low-rank side branch on all four DiT block GEMMs, with q/k/v adapters merged into the fused QKV site and the delta fused into the base GEMM's fp32 accumulator before bf16 rounding, as on CUDA |
| Image fidelity vs CUDA | **Matches** — for the same prompt/seed/steps, the final latent correlates 0.99 (no LoRA) / 0.88 (with the 4-step LoRA) with the CUDA engine, and the decoded images agree visually |
| CLI parity with `moss-infer generate` | **Done** — every CUDA generate flag is accepted; `--params`, `--json-report`, schedulers, flow multiplier, noise layout, sampler selection and the `generate` subcommand prefix work; backend-specific flags print a clear notice |
| Chinese / UTF-8 prompts | **Done** — the Windows entry point converts UTF-16 argv to UTF-8 (`wmain`), as the CUDA and Vulkan CLIs do |
| High resolution (1536x1024) | **Done** — the INT8 GEMM pads its partial M tile and checks launches, so a 6157-token joint sequence no longer faults |

The local models under `E:\ComfyUI\models` are **INT8 ConvRot** (the `--tier high`
layout). The reference tier needs **bf16** DiT/TE weights, which are not present
on this machine.

## Build (Windows / TheRock)

```bat
hip\build-win.bat
```

This loads the MSVC environment, locates the SDK with
`rocm-sdk path --root` (or the `ROCM_PATH`/`ROCM_HOME` override), configures CMake
(Ninja) and builds `moss-infer-hip`. No machine-specific SDK path is stored in
the script. All compile-time intermediates stay in the git-ignored
`hip\build-win\`; the final artifacts (exe + the SDK's `amdhip64*.dll`,
`rocm_kpack.dll`, `amd_comgr.dll`) are copied to the git-ignored `hip\dist\`.
Key CMake inputs:

- `CMAKE_CXX_COMPILER` = TheRock `lib\llvm\bin\clang-cl.exe`
- `ROCM_PATH` = the `_rocm_sdk_devel` directory
- `HIP_TARGET` = `gfx1103`
- `MI_HIP_WERROR=OFF` (TheRock emits MSVC-mode warnings)

The TheRock SDK ships the HIP device library at
`<ROCM_PATH>\lib\llvm\amdgcn\bitcode` (not the default search path), so
`CMakeLists.txt` passes `--rocm-device-lib-path` to it. `amdhip64_7.dll` imports
`rocm_kpack.dll` and `amd_comgr.dll`; all three must sit next to the exe.

## Run

Use the same launcher as the CUDA build (e.g.
`E:\ComfyUI\output\moss_image_sft_int8_convrot-4step-hip.bat`), or call the binary
directly. Launcher scripts are intentionally kept outside the repository because
they are edited per experiment.

```bat
moss-infer-hip generate --model <bundle> --dit <file> --te <file> --vae <file> ^
  --prompt "..." --tier high --steps 30 --cfg 4 ^
  [--lora <file> --lora-scale 1.0] -o out.png
```

`--model` supplies the tokenizer and configs; the three weight files may be
single `.safetensors` files. The CLI accepts the full `moss-infer generate` flag
set: the `generate` prefix, `--params`/`--json-report`, all sampler/scheduler and
SLG/CFG-accelerator flags, memory flags and I/O flags. Flags that need CUDA-only
machinery (planner modes, GGUF cache, image conditioning, LoRA merging, samplers
other than euler) are accepted but print a clear `no effect on this backend` /
`not implemented` notice instead of being silently ignored.

## Implementation notes

- `int8_leaf.h` uses the gfx11-native 16x16x16 int8 WMMA. The 16x16x32 shape
  used by the first draft maps to the gfx9 MFMA intrinsic and is a no-op on
  gfx11 (it silently produced zeros).
- The INT8 block's bf16 norms live at byte offsets
  `norm1=0`, `norm2=d*2`, `q_norm=2*d*2`, `k_norm=(2*d+128)*2`, before the qkv
  weight at `i8_norms`.
- Grid dimensions that scale with the token count are chunked so 4.2 MP images
  do not exceed driver limits.
- The INT8 GEMM stages a partial M tile through shared memory with zero rows:
  `load_a` always fetches 16 rows, so without padding the last tile reads past
  the activation buffer at large token counts and faults asynchronously (the
  error surfacing much later as `hipMemcpy to the host: unspecified launch
  failure`). The kernel now also checks launch errors at its call site.
- One INT8 GEMM workgroup computes a 128x32 output tile so eight M fragments
  reuse the same weight fragments. `hip_int8_gemm_check` checks it bit-for-bit
  (with a fused LoRA delta) against a host replay on several tail shapes.
- The attention key step is 64 (was 32), halving the number of softmax passes
  and barriers per head. `hip_attn_probe` checks both packed and unpacked paths
  against a CPU reference.
- At 1536x1024 with the 4-step LoRA, a step takes about 54 s on the 780M (down
  from 150 s before the attention and GEMM tiling changes). This is still well
  above ComfyUI's ~9.4 s/it: reaching it needs the software-pipelined 128x128
  multi-wave LDS GEMM from the ROCm-optimised ComfyUI kernels, which is not
  ported here.
- The DiT block's attention output head stride is the head dimension (128), not
  the full residual width; using the latter silently overwrites adjacent token
  rows and produces grey noise.
- The LoRA delta must be added to the base GEMM's fp32 accumulator *before* the
  bf16 rounding, not to the rounded output; fusing it into `int8_gemm` matches
  the CUDA side branch's magnitude.
