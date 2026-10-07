@echo off
REM P13 runtime GPU validation gate (Windows, RTX GPU machine only).
REM Usage:
REM   scripts\validate_runtime_gpu_windows.bat --audio "<user audio>" --ref-text "..." --generate-text "..."
REM Optional passthrough: --skip-stt, --model-dir, --app-exe.
REM Source E2E and frozen E2E both run: source success does not prove the
REM PyInstaller bundled runtime (P13 section 22).
setlocal
cd /d "%~dp0.."

REM Split user args: SOURCE_ARGS must never contain --app-exe, FROZEN_ARGS keeps it.
set "SOURCE_ARGS="
set "FROZEN_ARGS="
:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--app-exe" goto :take_appexe
set "SOURCE_ARGS=%SOURCE_ARGS% %1"
set "FROZEN_ARGS=%FROZEN_ARGS% %1"
shift
goto :parse_args
:take_appexe
set "FROZEN_ARGS=%FROZEN_ARGS% %1 %~2"
shift
shift
goto :parse_args
:args_done

echo === P13 runtime GPU validation ===
echo Python / torch / torchaudio / CUDA combination record:
python -c "import sys; print('python', sys.version)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
python -c "import torchaudio; print('torchaudio', torchaudio.__version__)"
if errorlevel 1 goto :err

echo === prepare_ffmpeg (fixed LGPL artifact, SHA-256 verified) ===
python scripts\prepare_ffmpeg.py
if errorlevel 1 goto :err

echo === check_ffmpeg (license/buildconf/libmp3lame) ===
python scripts\check_ffmpeg.py
if errorlevel 1 goto :err

echo === check_cuda ===
python scripts\check_cuda.py
if errorlevel 1 goto :err

echo === GPU pytest (tests\test_gpu_real.py -m gpu) ===
python -m pytest -q tests\test_gpu_real.py -m gpu
if errorlevel 1 goto :err

echo === SOURCE E2E ===
if not defined SOURCE_ARGS (
  echo [FAIL] missing arguments. Pass --audio / --ref-text / --generate-text.
  goto :err
)
python scripts\p13_runtime_e2e.py %SOURCE_ARGS%
if errorlevel 1 goto :err
echo SOURCE_E2E_OK

echo === FROZEN prerequisite ===
if not exist dist\VoiceStudio\VoiceStudio.exe (
  echo [FAIL] build frozen app first: scripts\build_windows.bat
  goto :err
)

echo === FROZEN strict smoke (--require-gpu) ===
dist\VoiceStudio\VoiceStudio.exe --smoke-test --require-gpu
if errorlevel 1 goto :err

echo === FROZEN E2E (bundled ffmpeg, frozen workers, frozen STT smoke) ===
python scripts\p13_runtime_e2e.py --app-exe "dist\VoiceStudio\VoiceStudio.exe" %FROZEN_ARGS%
if errorlevel 1 goto :err
echo FROZEN_E2E_OK

echo.
echo GPU_VALIDATION_OK
exit /b 0

:err
echo GPU_VALIDATION_FAILED
exit /b 1
