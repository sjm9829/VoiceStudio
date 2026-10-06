@echo off
REM P13 on-machine GPU validation (P12.3 build/GPU split): run on the target PC
REM with an NVIDIA GPU.
REM Usage: scripts\validate_gpu_windows.bat
REM For an installed build run directly:
REM   "C:\Program Files\VoiceStudio\VoiceStudio.exe" --smoke-test --require-gpu
setlocal
cd /d "%~dp0.."

python scripts\check_cuda.py || exit /b 1
dist\VoiceStudio\VoiceStudio.exe --smoke-test --require-gpu || exit /b 1
echo GPU_VALIDATION_OK
