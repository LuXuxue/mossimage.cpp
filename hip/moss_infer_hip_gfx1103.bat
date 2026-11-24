@echo off
rem ---------------------------------------------------------------------------
rem moss-infer-hip launcher for the AMD 780M (gfx1103), built from TheRock SDK.
rem
rem Native-HIP counterpart of moss_image_sft_int8_convrot-4step.bat. It runs the
rem MOSS-Image pipeline through HIP (not Vulkan) at --tier high (INT8 ConvRot DiT
rem + 8-bit attention). The bundle supplies tokenizer/configs; the weights are
rem passed as single files, as in the CUDA launcher.
rem ---------------------------------------------------------------------------
setlocal EnableExtensions

set "MOSS_INFER_HIP=%~dp0dist\moss-infer-hip.exe"
set "MODEL_DIR=E:\mossinfer\models\MOSS-Image"
set "OUTPUT_DIR=E:\mossinfer\output\hip"

set "DIT_FILE=E:\ComfyUI\models\diffusion_models\MOSS\moss_image_sft_int8_convrot.safetensors"
set "TE_FILE=E:\ComfyUI\models\text_encoders\Qwen3.5-9B-heretic-v2_int8_convrot.safetensors"
set "VAE_FILE=E:\ComfyUI\models\vae\flux2-vae_small.safetensors"

rem Leave LORA_FILE empty to test the base model (steps 30, cfg 4). Set it to the
rem 4-step distill LoRA to generate with the adapter (steps 6, cfg 1).
set "LORA_FILE="
set "LORA_SCALE=1.0"
set "LORA_PLACEMENT=side_branch"

if defined LORA_FILE (
  set "STEPS=6"
  set "CFG=1"
) else (
  set "STEPS=30"
  set "CFG=4"
)

if not exist "%MOSS_INFER_HIP%" (
  echo [moss-infer-hip] binary not found: %MOSS_INFER_HIP%
  echo [moss-infer-hip] build it first: hip\build-win.bat ^(see README-hip.md^)
  exit /b 1
)
if not exist "%MODEL_DIR%\tokenizer\tokenizer.json" (
  echo [moss-infer-hip] model bundle not found: %MODEL_DIR%
  exit /b 1
)
for %%F in ("%DIT_FILE%" "%TE_FILE%" "%VAE_FILE%") do if not exist %%~F (
  echo [moss-infer-hip] weight file not found: %%~F
  exit /b 1
)
if defined LORA_FILE if not exist "%LORA_FILE%" (
  echo [moss-infer-hip] LoRA not found: %LORA_FILE%
  exit /b 1
)
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

set "PROMPT=a portrait of a young woman, soft studio light, detailed skin, 85mm"
set "SIZE=1024x1024"
set "NEGATIVE=worst quality, bad anatomy, watermark, jpeg artifact"

set "LORA_ARGS="
if defined LORA_FILE set LORA_ARGS=--lora "%LORA_FILE%" --lora-scale %LORA_SCALE% --lora-placement %LORA_PLACEMENT%

echo [moss-infer-hip] prompt : %PROMPT%
echo [moss-infer-hip] model  : %MODEL_DIR%
echo [moss-infer-hip] steps  : %STEPS%  cfg: %CFG%
echo [moss-infer-hip] output : %OUTPUT_DIR%

"%MOSS_INFER_HIP%" ^
  --model "%MODEL_DIR%" ^
  --dit "%DIT_FILE%" ^
  --te "%TE_FILE%" ^
  --vae "%VAE_FILE%" ^
  --prompt "%PROMPT%" ^
  --negative "%NEGATIVE%" ^
  --width  %SIZE:~0,4% ^
  --height %SIZE:~5,4% ^
  --steps %STEPS% ^
  --cfg %CFG% ^
  --seed 42 ^
  --max-seq 512 ^
  --skip-layers "22,24,26" ^
  --tier high ^
  %LORA_ARGS% ^
  -o "%OUTPUT_DIR%\hip_%RANDOM%.png"

echo [moss-infer-hip] exit %errorlevel%
exit /b %errorlevel%
