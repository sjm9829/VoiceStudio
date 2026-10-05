# 구현 상태

## 현재 Phase
P01~P11 소스·테스트 1차 구현 완료. P09~P11 일부(설정 UX/통합/패키징 스크립트) 포함, 실기 검증 항목은 미검증.

## 로그
### P00
- 목표: 기술 스택 사실확인
- 결과: 개발 환경은 Linux(x86_64), Python 3.12.3, git 2.43 확인. ffmpeg 미설치 → AudioService 테스트는 fake adapter로 진행.
  Qwen 공식 API(ICL, `create_voice_clone_prompt`/`generate_voice_clone`, ref_text 필수, VoiceClonePromptItem list 전달)는
  공식 HF 모델 카드와 GitHub 소스 기준 문서로 고정했고, 실제 모델 접근은 이 환경에서 불가하므로 **미검증**으로 남김.
  Windows CUDA/PyInstaller/Inno Setup 실기 검증도 **미검증**(Windows validation pending).
- 남은 문제: 없음(기록된 미검증 항목은 각 Phase에서 opt-in 검증 필요)
- 다음: P01

### P01~P04 (scaffold/domain/오디오/프로필/Qwen 어댑터)
- 완료: core(paths/config/logging/errors), domain(VoiceProfile/GenerationJob), infra(FfmpegAdapter 실/가짜, SettingsRepository, ProfileRepository, QwenAdapter 실/가짜), services(AudioService/ProfileService/NarrationService/TextSegmenter/ModelManager/TranscriptionService), workers(worker_main + job/event 스키마) 구현.
- 핵심 설계: ref_code/ref_spk_embedding/x_vector_only_mode/icl_mode/ref_text를 safetensors+metadata에 영속화하고,
  생성 시 공식 VoiceClonePromptItem을 재구성해 list로 전달하는 경로를 QwenAdapter에 캡슐화(공식 dict 경로의 ref_text 누락 함정 대응).
- 프로필: 24kHz mono FLAC 참조 보관, metadata schema_version, dtype/shape 검증, v0 legacy migration 훅.

### P05~P08 (UI/받아쓰기/긴 대본)
- 완료: PySide6 UI shell(MainWindow/VoiceManagerDialog/VoiceEditorDialog/SettingsDialog/WaveformWidget),
  QProcess 기반 worker 실행/진행률/취소, 목소리 등록 동의 체크/이름 중복 차단, faster-whisper CPU 받아쓰기 서비스(지연 로드),
  Korean-first TextSegmenter(문단/문장부호 우선, target 120~180/hard max 240 상수화), chunk 무음 이어붙이기(180/320ms) + MP3 1회 인코딩(48kHz mono).
- 메인 프로세스는 qwen_tts/torch/faster_whisper를 import하지 않는다는 구조 테스트로 검증(test_settings_repository.py).

### P09~P11 (설정/통합/패키징)
- 설정 다이얼로그(저장 폴더/음질/GPU 상태/모델 받기), ModelManager(snapshot_download + .complete 마커 + 오프라인 오류), 패키징 스크립트(scripts/build_windows.bat, scripts/make_installer.bat, installer/voice-studio.iss) 작성.

## 검증
- `.venv/bin/pytest -m "not gpu" -p no:pytest-qt`: **59 passed** (fake adapter 기반 단위 테스트 전체).
- 전체 소스 ast 파싱(SYNTAX_OK). UI 모듈은 libGL 부재 환경 특성상 pytest-qt 플러그인 로드가 불가 →
  Qt 위젯 실행 테스트는 **미검증**(GUI 환경 필요), 소스 문법/구조 검사로 대체.
- 실제 Qwen 모델 생성, GPU, ffmpeg 실기, Windows 빌드: **미검증**.

## 남은 문제
- libGL이 있는 환경에서 pytest-qt UI 테스트 실행 필요.
- 실모델/GPU/Windows 검증은 opt-in(`gpu` marker) 환경에서 수행.
- worker_main의 실제 qwen_tts 클래스 시그니처는 공식 저장소 기준 문서 고정 상태 → 실기 smoke에서 재확인 필요.

## 다음
- GUI 가능 환경에서 P05~P07 UI 테스트 및 실기 검증(gpu marker), 이후 Windows 빌드 실기 확인.
