# 보이스 스튜디오

`등록한 목소리로 자연스러운 나레이션을 만들어 주는 음성 생성 프로그램`

Windows 일반 사용자를 위한 로컬 전용 음성 복제·나레이션 생성 데스크톱 애플리케이션입니다.
브라우저, 웹서버, Docker가 필요하지 않고, 목소리·대본·프로필이 외부 서버로 업로드되지 않습니다.

## 핵심 사용 흐름
1. `목소리 관리`에서 참조 음성 파일(MP3/M4A/WAV/FLAC)과 그 대사를 등록합니다.
2. 메인 `나레이션` 화면에서 목소리를 선택하고 대본을 입력합니다.
3. `음성 생성`을 누르면 구간별 생성 후 하나의 MP3로 완성됩니다.
4. 내장 재생으로 들어보고 `MP3 저장`으로 내보냅니다.

## 기술 스택
- Python 3.12, PySide6 (Windows 데스크톱 UI)
- 음성 복제: Qwen 공식 `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (ICL 모드, `x_vector_only_mode=False`)
- 오디오 입출력: FFmpeg/ffprobe (최종 출력은 MP3, mono, 48kHz)
- 자동 받아쓰기: faster-whisper `small`, `compute_type=int8`, CPU 전용
- 배포: PyInstaller `onedir` + Inno Setup

## 설치(개발)
```bash
cd voice-studio
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"   # Windows: .venv\Scripts\pip install -e ".[dev]"
```

## 개발 실행
```bash
.venv/bin/voice-studio            # 콘솔 엔트리
# 또는
.venv/bin/python -m voice_studio.main
```

## 테스트
```bash
.venv/bin/pytest -m "not gpu"
.venv/bin/pytest -m gpu           # 실제 Qwen 모델이 필요한 opt-in 테스트
```

## 빌드(Windows)
```bash
scripts\build_windows.bat        # PyInstaller onedir
scripts\make_installer.bat       # Inno Setup (설치 프로그램)
```
모델 가중치는 설치 프로그램에 포함하지 않고 첫 필요 시 Hugging Face에서 내려받습니다.

## 디렉터리
```
voice-studio/
  src/voice_studio/       애플리케이션 소스
    core/                 경로/설정/로그/오류
    domain/               VoiceProfile, GenerationJob 등 도메인
    services/             AudioService, ProfileService, ModelManager 등
    infra/                ffmpeg 어댑터, qwen 어댑터, 저장소
    ui/                   PySide6 화면
    workers/              Qwen worker 프로세스
  tests/                  단위 테스트
  docs/                   개발 문서(한국어)
  scripts/                빌드/패키징 스크립트
  third_party/            라이선스/고지
```

## 개인정보 로컬 처리
- 참조 음성, 대사, 생성 결과는 `%LOCALAPPDATA%\VoiceStudio\` 아래에만 저장됩니다.
- 모델 다운로드(Hugging Face) 시에만 네트워크를 사용하며, 그 외 동작은 로컬입니다.

## 라이선스
- 애플리케이션 코드: MIT (`LICENSE`)
- 모델: Apache-2.0 (`third_party/MODEL_NOTICE.md` 참조)
