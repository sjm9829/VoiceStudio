@echo off
REM P13 runtime GPU validation (Windows, RTX GPU machine only).
REM Usage:
REM   scripts\validate_runtime_gpu_windows.bat --audio "<user audio>" --ref-text "..." --generate-text "..."
REM Audio/ref-text are passed through to p13_runtime_e2e.py (no hardcoded user data).
setlocal
cd /d "%~dp0.."

echo === P13 runtime GPU validation ===
echo Python / torch / torchaudio / CUDA combination record:
python -c "import sys; print('python', sys.version)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
python -c "import torchaudio; print('torchaudio', torchaudio.__version__)"
if errorlevel 1 goto :err

echo === check_cuda ===
python scripts\check_cuda.py
if errorlevel 1 goto :err

echo === GPU pytest (tests\test_gpu_real.py -m gpu) ===
python -m pytest -q tests\test_gpu_real.py -m gpu
if errorlevel 1 goto :err

echo === runtime E2E ===
if "%~1"=="" (
  echo [FAIL] missing arguments. Pass --audio / --ref-text / --generate-text.
  goto :err
)
python scripts\p13_runtime_e2e.py %*
if errorlevel 1 goto :err

echo.
echo GPU_VALIDATION_OK
exit /b 0

:err
echo GPU_VALIDATION_FAILED
exit /b 1
