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

## P12 Real Runtime Integration (이번 단계, 코드 레벨 완료)

### 완료 내용
- **공식 Qwen3-TTS API 재대조**: QwenLM/Qwen3-TTS main 기준 `create_voice_clone_prompt(ref_audio, ref_text=None, x_vector_only_mode=False) -> List[VoiceClonePromptItem]`,
  `generate_voice_clone(text, ..., voice_clone_prompt=None, ...) -> Tuple[List[np.ndarray], int]`,
  `VoiceClonePromptItem`는 `from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem` public export로 확인.
  RealQwenAdapter는 list 반환의 첫 item을 사용하고, 생성 시 공식 `VoiceClonePromptItem`을 재구성해 **list**로 전달한다.
- **worker 계약 통합(schema v1)**: `workers/job_schema.py`(parse_job/build_narrate_payload/build_register_payload,
  E_JOB_MISSING_FIELD/E_JOB_BAD_VALUE/E_JOB_MODE/E_JOB_VERSION 사용자 오류 코드),
  `workers/launcher.py`(개발 `python -m voice_studio.main --worker job.json` / frozen `VoiceStudio.exe --worker job.json`),
  `workers/linebuffer.py`(partial JSONL 버퍼링), `workers/protocol.py` 이벤트 정비.
- **UI → worker 연결**: MainWindow 생성 버튼이 완성된 narrate job으로 QProcess worker를 실행하고 JSONL 진행/결과를 버퍼링 수신.
  등록 화면은 `profile_service.register(fake)`를 호출하지 않고 register worker(QProcess)를 실행하며,
  받아쓰기(`_TranscribeThread`)와 구간 미리 듣기(`_PreviewThread`)를 UI 스레드 밖에서 실행.
- **production fake 제거**: AppContext는 ProfileService(프로필 CRUD 전용, qwen=None) + 실제 FasterWhisperTranscriber(cpu/int8 지연 로드).
  FakeQwenAdapter/FakeTranscriber/FakeFfmpegAdapter는 dev_context와 테스트에만 존재.
- **모델 로컬 경로/오프라인**: ModelManager가 핵심 파일(config/tokenizer/vocab 등) 존재까지 검증한 로컬 snapshot 경로를 job에 전달하고,
  worker의 RealQwenAdapter가 `from_pretrained(로컬 경로)`를 사용. 다운로드 후 오프라인 동작 가능.
- **ref_code tensor 직렬화**: torch.Tensor/ndarray는 원래 dtype(정수 code는 정수 dtype)/shape로 safetensors 저장,
  embedding은 float dtype. 로드 시 dtype/ndim 검증, legacy JSON blob은 마이그레이션. torch 미설치 환경(UI 프로세스)에서도 안전.
- **worker 진입점**: `main.py --worker` 분기(QApplication 없이 worker 실행), worker_main.py 직접 실행도 동일 계약.
- **pyproject**: production runtime dependencies(PySide6/numpy/safetensors/huggingface_hub/faster-whisper) 명시,
  GPU(qwen-tts+torch CUDA wheel)는 설치 스크립트에서 공식 권장 방식으로 별도 설치.

### 테스트(정확한 숫자)
- `.venv/bin/pytest -m "not gpu"`: **75 passed, 5 skipped** (기존 59 유지 + 신규 계약/통합 16, ffmpeg 실기 5는 이 환경에 ffmpeg 없어 skip).
- 신규: job 스키마 round-trip/오류 코드, dev/frozen worker command, partial JSONL 버퍼, production 컨텍스트 fake 주입 없음,
  모듈 레벨 heavy import 금지(정적), MainWindow job 빌더 필수 필드, 등록 화면 worker 위임, 마커+핵심 파일 모델 검증.
- GPU opt-in(`pytest -m gpu`): 실모델 로드→ICL prompt→생성, 프로필 저장→새 프로세스 load round-trip, GPU 메모리 반환. 이 환경에서 **미검증**.

## P13 Windows GPU E2E Validation — 미검증
실제 Windows + NVIDIA GPU + Qwen 모델 + FFmpeg + packaged exe에서의 E2E(등록→재시작→대본→생성→MP3)는 이 Linux 환경에서 실행 불가 → **미검증**.
검증 절차는 README의 P13 체크리스트 및 `pytest -m gpu` 참조.

