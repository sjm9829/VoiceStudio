@echo off
REM P13 실기 GPU validation(P12.3 역할 분리): NVIDIA GPU가 있는 대상 PC에서 실행한다.
REM 사용법: scripts\validate_gpu_windows.bat  (설치본 검증은
REM "C:\Program Files\VoiceStudio\VoiceStudio.exe" --smoke-test --require-gpu 를 직접 실행)
setlocal
cd /d "%~dp0.."

python scripts\check_cuda.py || exit /b 1
dist\VoiceStudio\VoiceStudio.exe --smoke-test --require-gpu || exit /b 1
echo GPU_VALIDATION_OK
