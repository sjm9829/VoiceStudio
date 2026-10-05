@echo off
REM 보이스 스튜디오 Windows 빌드 (PyInstaller onedir, packaging\VoiceStudio.spec)
setlocal
cd /d "%~dp0.."

REM CUDA wheel 태그(P12.2-12): cu124처럼 낡은 조합을 하드코딩하지 않는다.
REM CUDA_TAG를 빌드 시점에 지정할 수 있고, 기본은 현재 권장 cu126
REM (torch CUDA wheel에서 RTX 2070 SUPER/CC 7.5 지원). 다른 조합은 환경변수로 교체.
if "%CUDA_TAG%"=="" set CUDA_TAG=cu126

python -m venv .venv || goto :err
call .venv\Scripts\activate

python -m pip install --upgrade pip || goto :err

REM pyproject.toml extras와 일치: dev(pytest/pytest-qt). UI/audio/transcribe 의존성은
REM base dependencies로 이동되어 있다(P12.1-10).
pip install -e ".[dev]" || goto :err

REM GPU/TTS 런타임: torch CUDA wheel을 공식 index에서 먼저 설치하고, qwen-tts는 PyPI.
REM qwen-tts 설치 전후로 torch 버전을 기록해 qwen-tts가 torch를 교체/강등하는지 확인한다.
python -m pip install torch --index-url https://download.pytorch.org/whl/%CUDA_TAG% || goto :err
echo TORCH_BEFORE_QWEN_TTS > build_torch_version.txt
python -c "import torch; print(torch.__version__)" >> build_torch_version.txt || goto :err
pip install qwen-tts || goto :err
echo TORCH_AFTER_QWEN_TTS >> build_torch_version.txt
python -c "import torch; print(torch.__version__)" >> build_torch_version.txt || goto :err
type build_torch_version.txt
REM pip 의존성 무결성 확인(qwen-tts가 torch/transformers를 충돌 버전으로 격상/강등하지 않았는지)
python -m pip check || goto :err

REM CUDA/모델 런타임 진단: 실패 시 빌드를 중단한다(P12.2-13).
python scripts\check_cuda.py || goto :err

pip install pyinstaller || goto :err
pyinstaller --noconfirm --clean packaging\VoiceStudio.spec || goto :err

REM frozen 스모크(옵션, 실기 필수 아님): --smoke-test 내부 모드 실행(P12.2-21)
dist\VoiceStudio\VoiceStudio.exe --smoke-test || goto :err

echo BUILD_OK dist\VoiceStudio
goto :eof
:err
echo BUILD_FAILED
exit /b 1
