# 07. 패키징과 릴리스

## 방식
- PyInstaller `onedir` (거대 onefile 미사용) + Inno Setup 설치 프로그램.
- 모델 가중치는 설치 프로그램에 포함하지 않고 첫 필요 시 공식 Hugging Face에서 내려받음.
- FFmpeg 바이너리는 설치 프로그램에 포함하거나, 없으면 첫 실행 안내로 제공.

## 스크립트
- `scripts/build_windows.bat`: venv 생성 → 의존성 설치 → PyInstaller onedir 빌드
- `scripts/make_installer.bat`: Inno Setup(iscc) 실행
- `scripts/check_cuda.py`: CUDA/그래픽 카드 진단(개발 모드)

## Build 머신과 런타임의 GPU 요구사항(역할 분리)

- Build machine: NVIDIA GPU not required. `scripts\build_windows.bat`는
  `scripts\check_runtime_packages.py`(패키지 무결성만 검사)를 사용하며
  CUDA wheel 설치와 패키징은 GPU 없이 가능하다.
- Runtime inference: NVIDIA CUDA GPU required. 앱은 여전히 CUDA GPU 없이 TTS 실행을
  지원하지 않는다(CPU fallback 없음).
- P13 validation target: RTX 2070 SUPER 8GB (Turing, CC 7.5, dtype float16,
  FlashAttention-2 OFF). 실기 검증은 `scripts\validate_gpu_windows.bat` 또는
  `dist\VoiceStudio\VoiceStudio.exe --smoke-test --require-gpu`로 수행한다.

## 라이선스/고지
- `third_party/LICENSES.md`: FFmpeg(libmp3lame 포함), PySide6, faster-whisper, PyInstaller 등
- `third_party/MODEL_NOTICE.md`: Qwen 모델 Apache-2.0 + Qwen 고지
- **FFmpeg 배포 라이선스 결정(P12.2-06)**: LGPL 빌드(--enable-lgpl, libmp3lame 포함)를 배포한다.
  GPL 빌드는 배포하지 않는다. 고지 문서는 `third_party/FFMPEG_NOTICE.txt`이며 PyInstaller spec이
  이를 설치 폴더 `bin/`에 함께 넣는다.

## Python 3.12 의존성 호환성 확인(P12.2-27, 2026-10 기준 PyPI 조사)
다음 조합이 모두 cp312 wheel(또는 py3-universal)을 제공하므로 Python 3.12를 유지한다:
- qwen-tts (universal wheel, 플랫폼 무관)
- torch (Windows cp312 wheel 제공)
- ctranslate2 (Windows cp312 wheel 제공)
- faster-whisper (pure python, universal)
- PyInstaller (Windows cp312 wheel 제공)
- PySide6 (Windows cp312 wheel 제공)
따라서 Python 3.11로 내릴 이유가 없다. 실제 Windows 설치에서 wheel 설치 오류가 나면
이 섹션을 먼저 갱신하고 재판정한다.

## frozen 스모크(P12.2-21)
- `VoiceStudio.exe --smoke-test`: 내부 개발용 모드. PySide6/worker 스키마/UI 모듈 import와
  AppContext 경로를 확인하고 종료한다(결과는 logs/smoke-test.log에도 기록).
- `scripts/smoke_frozen.bat <설치 폴더>`: 설치 후 `--smoke-test`와 `--worker invalid.json`
  진입을 확인한다. 일반 UI에는 노출하지 않는다.

## 설치 후 Self-Diagnosis(P12.2-22)
- 설정 화면의 `그래픽 카드 상태 확인`이 요약(그래픽 카드/CUDA/음성 모델/오디오 구성 요소/
  자동 받아쓰기)을 표시한다.
- 상세(torch 버전, CUDA 런타임, GPU 이름, bf16 지원, ffmpeg/ffprobe 경로, qwen_tts import,
  model_path)는 `%LOCALAPPDATA%\VoiceStudio\logs\diagnosis.log`에 기록된다.
- UI는 무거운 런타임을 import하지 않도록 `voice_studio.diagnostics`를 자식 인터프리터로 실행한다.

## worker 실패 진단(P12.2-23)
- worker 실패 시 traceback은 UI에 노출하지 않고 `%LOCALAPPDATA%\VoiceStudio\logs\worker-failure.log`에
  job_id/mode/exception/traceback을 남긴다.
- 취소(QProcess terminate→kill) 후 ffmpeg 자식 프로세스 잔존 여부는 **실기 미검증** 항목이며,
  P13 실기 체크리스트에서 확인한다(P12.2-19).

## GGUF 음성 엔진 번들(P17-C/D)

- 음성 백엔드는 설정에서 선택한다: `official`(기존 0.6B HF 스냅샷) 또는
  `gguf`(Qwen3-TTS 1.7B Q8_0 GGUF + llama.cpp).
- GGUF 모델 가중치는 설치 프로그램에 넣지 않고, 공식 repo의 고정 revision에서
  SHA-256 검증 후 내려받는다(기존 0.6B 정책과 동일).
- llama.cpp 실행 엔진은 고정 release `b11540`의 `win-cuda-12.4-x64` + `cudart`만
  사용한다. `scripts/build_windows.bat`가 `scripts/fetch_gguf_engine.py`로
  `packaging/engine-src/llama-cuda`를 준비하고 PyInstaller spec이 `<app>/engine/llama-cuda`로
  번들한다(저장소 미포함, 빌드 시 재생성).
- frozen에서 GGUF 캐시/엔진 경로는 각각 `%LOCALAPPDATA%\VoiceStudio\models\gguf`,
  `<app>\engine\llama-cuda`다. `VOICE_STUDIO_ENGINE_DIR`로 override 가능.
- GGUF 백엔드는 CUDA GPU가 없으면 명시적으로 실패한다(CPU fallback 없음, 기존 정책 동일).

## 릴리스 체크리스트
1. `pytest -m "not gpu"` 전부 통과
2. Windows 실기 빌드 + 설치 + 실행 + 생성 E2E (Windows validation pending 관리)
3. 한글 경로/공백 경로/긴 파일명 실기 확인
4. 설치 후 재실행 시 profile 유지 확인
5. 제거 시 사용자 데이터 정책 문서화(기본 유지)
