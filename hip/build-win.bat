@echo off
rem Windows build driver for the HIP backend (TheRock SDK): loads the MSVC environment, then configures and builds.
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1

set "ROCM_SDK=C:\Users\LuXuxue\AppData\Local\Programs\Python\Python314\Lib\site-packages\_rocm_sdk_devel"
set "SRC=%~dp0"
if "%SRC:~-1%"=="\" set "SRC=%SRC:~0,-1%"

rem build-win/ holds all compile-time intermediates (CMake, Ninja, object files); it is git-ignored.
set "BUILD=%SRC%\build-win"
set "CFG=%BUILD%\CMakeCache.txt"
rem dist/ holds the final compiled artifacts; it is git-ignored.
set "DIST=%SRC%\dist"

if not exist "%CFG%" (
  cmake -G Ninja -S "%SRC%" -B "%BUILD%" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CXX_COMPILER="%ROCM_SDK%\lib\llvm\bin\clang-cl.exe" ^
    -DHIP_TARGET=gfx1103 ^
    -DROCM_PATH="%ROCM_SDK%" ^
    -DMI_HIP_WERROR=OFF ^
    -DMOSSINFER_HIP_TOOLS=OFF
  if errorlevel 1 exit /b 1
)

cmake --build "%BUILD%" -j
if errorlevel 1 exit /b 1

if not exist "%DIST%" mkdir "%DIST%"
copy /y "%BUILD%\moss-infer-hip.exe" "%DIST%\" >nul
copy /y "%BUILD%\amdhip64*.dll" "%DIST%\" >nul
exit /b 0
