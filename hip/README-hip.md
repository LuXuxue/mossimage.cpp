# HIP backend for AMD (gfx1103 / 780M)

Native-HIP inference for MOSS-Image on AMD GPUs, built from the **TheRock** SDK
(no full ROCm install required). This is the AMD counterpart of the CUDA
(`moss-infer`) and Vulkan (`moss-infer-vulkan`) backends — it runs on the AMD
iGPU through HIP, not Vulkan.

## Status

| Item | State |
| --- | --- |
| Windows / TheRock build (`moss-infer-hip.exe`) | **Done** — compiles and links |
| 780M (gfx1103) device detection | **Done** — `--list-devices` shows the device + rocWMMA leaf |
| Reference tier (bf16 GEMMs) | **Built** — runs, needs a bf16 DiT/TE bundle |
| `--tier high` (INT8 ConvRot DiT + 8-bit attention) | **In progress** — the INT8 ConvRot kernels are the remaining port |
| End-to-end image test | Blocked until `--tier high` is complete (the local models are INT8) |

The local models under `E:\ComfyUI\models` are **INT8 ConvRot** (the `--tier high`
layout). The reference tier needs **bf16** DiT/TE weights, which are not present
on this machine. So an end-to-end image test requires the `--tier high` port.

## Build (Windows / TheRock)

```bat
hip\build-win.bat
```

This loads the MSVC environment, configures CMake (Ninja) against the TheRock
SDK, and builds `moss-infer-hip`. All compile-time intermediates (CMake cache,
Ninja state, object files) stay in the git-ignored temp folder `hip\build-win\`;
the final artifacts (exe + the SDK's `amdhip64*.dll`) are copied to the
git-ignored `hip\dist\` folder after a successful build. Key CMake inputs:

- `CMAKE_CXX_COMPILER` = TheRock `lib\llvm\bin\clang-cl.exe`
- `ROCM_PATH` = the `_rocm_sdk_devel` directory
- `HIP_TARGET` = `gfx1103`
- `MI_HIP_WERROR=OFF` (TheRock emits MSVC-mode warnings)

The TheRock SDK ships the HIP device library at
`<ROCM_PATH>\lib\llvm\amdgcn\bitcode` (not the default search path), so
`CMakeLists.txt` passes `--rocm-device-lib-path` to it. The runtime needs the
SDK's `bin\*.dll` (notably `amdhip64_7.dll`) next to the exe — `build-win.bat`
copies them after the build.

## Run

```bat
hip\moss_infer_hip_gfx1103.bat
```

This uses the bundle `E:\mossinfer\models\MOSS-Image-hip` (tokenizer + symbolic
links to the INT8 DiT/TE and bf16 VAE under `E:\ComfyUI\models`) and runs
`--tier high`.

## `--tier high` port (remaining work)

The CUDA engine's `--tier high` runs the DiT's GEMMs as INT8 ConvRot and the
attention as 8-bit. Porting to HIP requires, in `hip/`:

1. **INT8 weight loading** — accept `int8_convrot(256)` DiT/TE tensors in
   `dit.cpp` / `text_encoder.cpp` (packed i8 + per-1x32/1x128 f32 scales), and
   expose them to the kernels.
2. **Quantize kernels** — HIP ports of `rotate_quantize` and
   `rms_norm_modulate_rotate_quantize` (bf16 in → i8 + scale out), in
   `block_kernels.hip`.
3. **INT8 ConvRot GEMM** — a rocWMMA `i8` mma walk (the leaf
   `v_wmma_i8_16x16x32_i8` is already compiled) with the ConvRot dequant epilogue,
   for the `qkv`/`out`/`mlp` linears.
4. **8-bit attention** — the Q8 attention kernels (q8_fp8 / q8_fp16) for the
   42 DiT blocks.
5. **Wiring** — select the INT8 path in `generate.cpp` / `dit.cpp` when the
   loaded DiT is INT8 and `--tier high`, mirroring the CUDA planner.

The reference (`comfy-kitchen-igpu`) gfx1103 optimizations and the CUDA
`kernels/quant/` sources are the implementation references.
