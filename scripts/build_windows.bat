@echo off
REM Voice Studio Windows build (PyInstaller onedir, packaging\VoiceStudio.spec)
setlocal
cd /d "%~dp0.."

REM P17-C/D: GGUF engine (llama.cpp b11540 CUDA 12.4) bundle prerequisite.
REM Fetched from the pinned release only; skipped silently when already prepared.
python scripts\fetch_gguf_engine.py || goto :err

REM FFmpeg build prerequisite (P13): binaries are not committed to the repository.
REM Prepare them automatically from a fixed LGPL release (no manual download and
REM no system PATH dependency), then verify license/buildconf (P12.3-04).
python scripts\prepare_ffmpeg.py || goto :err
python scripts\check_ffmpeg.py || goto :err

python -m venv .venv || goto :err
call .venv\Scripts\activate

python -m pip install --upgrade pip || goto :err

REM Matches pyproject.toml extras: dev (pytest/pytest-qt). UI/audio/transcribe
REM dependencies were moved into base dependencies (P12.1-10).
pip install -e ".[dev]" || goto :err

REM Runtime package integrity check (build/GPU validation split): no GPU needed
REM on the build machine. Real CUDA/GPU validation runs on the P13 target PC via
REM scripts\check_cuda.py and the --require-gpu smoke.
python scripts\check_runtime_packages.py || goto :err

REM P13: non-GPU pytest gate before packaging. gpu/stt markers are opt-in
REM (real CUDA model / real faster-whisper download) and excluded here.
set QT_QPA_PLATFORM=offscreen
python -m pytest -q -m "not gpu and not stt" tests || goto :err
set QT_QPA_PLATFORM=

pip install pyinstaller || goto :err
pyinstaller --noconfirm --clean packaging\VoiceStudio.spec || goto :err

REM Frozen smoke: CPU-safe (packaging integrity). CUDA unavailable is a SKIP, not a failure.
dist\VoiceStudio\VoiceStudio.exe --smoke-test || goto :err
REM Verify FFmpeg is actually bundled in the build output (P12.3-05)
python scripts\check_dist.py || goto :err

echo BUILD_OK dist\VoiceStudio
goto :eof
:err
echo BUILD_FAILED
exit /b 1
