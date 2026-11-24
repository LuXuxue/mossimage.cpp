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

The local models under `E:\ComfyUI\models` are **INT8 ConvRot** (the `--tier high`
layout). The reference tier needs **bf16** DiT/TE weights, which are not present
on this machine.

## Build (Windows / TheRock)

```bat
hip\build-win.bat
```

This loads the MSVC environment, configures CMake (Ninja) against the TheRock
SDK, and builds `moss-infer-hip`. All compile-time intermediates stay in the
git-ignored `hip\build-win\`; the final artifacts (exe + the SDK's `amdhip64*.dll`,
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

```bat
hip\moss_infer_hip_gfx1103.bat
```

The launcher uses the same bundle/weight files as the CUDA launcher. With
`LORA_FILE` empty it runs the base model with **steps 30 / cfg 4**; setting
`LORA_FILE` to the 4-step distill LoRA switches to **steps 6 / cfg 1** and adds
`--lora`.

```bat
moss-infer-hip --model <bundle> --dit <file> --te <file> --vae <file> ^
  --prompt "..." --tier high --steps 30 --cfg 4 ^
  [--lora <file> --lora-scale 1.0] -o out.png
```

`--model` supplies the tokenizer and configs; the three weight files may be
single `.safetensors` files. CUDA-only knobs (`--mode`, `--attention`,
`--sampler`, `--scheduler`, `--dit-blocks`, step-cache flags) are accepted but
reported as ignored.

## Implementation notes

- `int8_leaf.h` uses the gfx11-native 16x16x16 int8 WMMA. The 16x16x32 shape
  used by the first draft maps to the gfx9 MFMA intrinsic and is a no-op on
  gfx11 (it silently produced zeros).
- The INT8 block's bf16 norms live at byte offsets
  `norm1=0`, `norm2=d*2`, `q_norm=2*d*2`, `k_norm=(2*d+128)*2`, before the qkv
  weight at `i8_norms`.
- Grid dimensions that scale with the token count are chunked so 4.2 MP images
  do not exceed the device grid limits.
- The DiT block's attention output head stride is the head dimension (128), not
  the full residual width; using the latter silently overwrites adjacent token
  rows and produces grey noise.
- The LoRA delta must be added to the base GEMM's fp32 accumulator *before* the
  bf16 rounding, not to the rounded output; fusing it into `int8_gemm` matches
  the CUDA side branch's magnitude.
