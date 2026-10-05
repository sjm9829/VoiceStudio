@echo off
REM 보이스 스튜디오 Windows 빌드 (PyInstaller onedir, packaging\VoiceStudio.spec)
setlocal
cd /d "%~dp0.."

REM CUDA wheel 태그(P12.2-12): cu124처럼 낡은 조합을 하드코딩하지 않는다.
REM CUDA_TAG를 빌드 시점에 지정할 수 있고, 기본은 현재 권장 cu126
REM (torch CUDA wheel에서 RTX 2070 SUPER/CC 7.5 지원). 다른 조합은 환경변수로 교체.
if "%CUDA_TAG%"=="" set CUDA_TAG=cu126

REM FFmpeg build prerequisite(P12.3-03): binary는 repository에 commit하지 않고
REM 빌드 전에 third_party\bin에 준비한다. 없으면 BUILD_OK가 나오지 않도록 중단한다.
if not exist third_party\bin\ffmpeg.exe (
    echo [FAIL] third_party\bin\ffmpeg.exe missing
    echo        Prepare the LGPL ffmpeg.exe build prerequisite in third_party\bin first.
    goto :err
)
if not exist third_party\bin\ffprobe.exe (
    echo [FAIL] third_party\bin\ffprobe.exe missing
    goto :err
)
REM 공급 binary의 license/buildconf를 실제로 확인한다(P12.3-04). GPL 구성이면 배포 전 재검토.
python scripts\check_ffmpeg.py || goto :err

python -m venv .venv || goto :err
call .venv\Scripts\activate

python -m pip install --upgrade pip || goto :err

REM pyproject.toml extras와 일치: dev(pytest/pytest-qt). UI/audio/transcribe 의존성은
REM base dependencies로 이동되어 있다(P12.1-10).
pip install -e ".[dev]" || goto :err

REM GPU/TTS 런타임: qwen-tts는 PyPI에서 설치하고, 설치 전후로 torch/torchaudio 버전을
REM 기록해 qwen-tts가 CUDA wheel을 교체/강등하는지 확인한다(P12.3-12).
REM P12.3-11: torch와 torchaudio를 동일 CUDA wheel index에서 함께 설치해
REM qwen-tts가 torchaudio를 CPU wheel로 강등/격상하지 않도록 먼저 고정한다.
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/%CUDA_TAG% || goto :err
echo TORCH_BEFORE_QWEN_TTS > build_torch_version.txt
python -c "import torch, torchaudio; print('torch', torch.__version__); print('torchaudio', torchaudio.__version__); print('cuda', torch.version.cuda)" >> build_torch_version.txt || goto :err
pip install qwen-tts || goto :err
echo TORCH_AFTER_QWEN_TTS >> build_torch_version.txt
echo TORCHAUDIO_BEFORE_QWEN_TTS >> build_torch_version.txt
echo TORCHAUDIO_AFTER_QWEN_TTS >> build_torch_version.txt
python -c "import torch, torchaudio; print('torch', torch.__version__); print('torchaudio', torchaudio.__version__); print('cuda', torch.version.cuda); assert torch.version.cuda is not None" >> build_torch_version.txt || goto :err
type build_torch_version.txt
REM pip 의존성 무결성 확인(qwen-tts가 torch/transformers를 충돌 버전으로 격상/강등하지 않았는지)
python -m pip check || goto :err

REM CUDA/모델 런타임 진단: 실패 시 빌드를 중단한다(P12.2-13).
python scripts\check_cuda.py || goto :err

pip install pyinstaller || goto :err
pyinstaller --noconfirm --clean packaging\VoiceStudio.spec || goto :err

REM frozen 스모크: heavy import/FFmpeg/CUDA/dtype까지 확인(P12.2-21/P12.3-17)
dist\VoiceStudio\VoiceStudio.exe --smoke-test || goto :err
REM 빌드 결과물의 FFmpeg 실제 포함 검증(P12.3-05)
python scripts\check_dist.py || goto :err

echo BUILD_OK dist\VoiceStudio
goto :eof
:err
echo BUILD_FAILED
exit /b 1
