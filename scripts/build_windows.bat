@echo off
REM 보이스 스튜디오 Windows 빌드 (PyInstaller onedir, packaging\VoiceStudio.spec)
setlocal
cd /d "%~dp0.."

python -m venv .venv || goto :err
call .venv\Scripts\activate

python -m pip install --upgrade pip || goto :err

REM pyproject.toml extras와 일치: dev(pytest/pytest-qt). UI/audio/transcribe 의존성은
REM base dependencies로 이동되어 있다(P12.1-10).
pip install -e ".[dev]" || goto :err

REM GPU/TTS 런타임: qwen-tts는 PyPI에서 설치하되, torch CUDA wheel은 공식 권장 방식
REM (https://pytorch.org 의 CUDA 설치 명령)으로 먼저 설치한다. 버전은 임의 고정하지 않는다.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124 || goto :err
pip install qwen-tts || goto :err

python scripts\check_cuda.py || goto :err

pip install pyinstaller || goto :err
pyinstaller --noconfirm --clean packaging\VoiceStudio.spec || goto :err

echo BUILD_OK dist\VoiceStudio
goto :eof
:err
echo BUILD_FAILED
exit /b 1
