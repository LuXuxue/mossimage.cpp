@echo off
rem ---------------------------------------------------------------------------
rem moss-infer-hip launcher for the AMD 780M (gfx1103), built from TheRock SDK.
rem
rem Native-HIP counterpart of moss_image_sft_int8_convrot-4step.bat. It runs the
rem MOSS-Image pipeline on the AMD iGPU through HIP (not Vulkan), at --tier high
rem (INT8 ConvRot DiT + 8-bit attention) with the same models as the CUDA bat.
rem
rem Model bundle (E:\mossinfer\models\MOSS-Image-hip) holds:
rem   tokenizer\tokenizer.json
rem   transformer\diffusion_pytorch_model.safetensors   (INT8 ConvRot DiT)
rem   text_encoder\diffusion_pytorch_model.safetensors   (INT8 ConvRot TE)
rem   vae\diffusion_pytorch_model.safetensors            (bf16 FLUX.2 VAE)
rem The weight files are symbolic links to the copies under E:\ComfyUI\models.
rem ---------------------------------------------------------------------------
setlocal

set "MOSS_INFER_HIP=%~dp0dist\moss-infer-hip.exe"
set "MODEL_DIR=E:\mossinfer\models\MOSS-Image-hip"
set "OUTPUT_DIR=E:\mossinfer\output\hip"

if not exist "%MOSS_INFER_HIP%" (
  echo [moss-infer-hip] binary not found: %MOSS_INFER_HIP%
  echo [moss-infer-hip] build it first: hip\build-win.bat (see README-hip.md)
  exit /b 1
)
if not exist "%MODEL_DIR%\tokenizer\tokenizer.json" (
  echo [moss-infer-hip] model bundle not found: %MODEL_DIR%
  exit /b 1
)

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

set "PROMPT=a portrait of a young woman, soft studio light, detailed skin, 85mm"
set "SIZE=1024x1024"

echo [moss-infer-hip] prompt : %PROMPT%
echo [moss-infer-hip] model  : %MODEL_DIR%
echo [moss-infer-hip] output : %OUTPUT_DIR%

"%MOSS_INFER_HIP%" ^
  --model "%MODEL_DIR%" ^
  --prompt "%PROMPT%" ^
  --width  %SIZE:~0,4% ^
  --height %SIZE:~5,4% ^
  --steps 30 ^
  --cfg 5 ^
  --seed 42 ^
  --max-seq 512 ^
  --tier high ^
  -o "%OUTPUT_DIR%\hip_%RANDOM%.png"

echo [moss-infer-hip] exit %errorlevel%
exit /b %errorlevel%
