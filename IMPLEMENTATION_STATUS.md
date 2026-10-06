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

## P12.1 Runtime Blocker Fix (2차: 실행 차단 버그 제거, 코드 레벨 완료)

### 코드 수준 완료 (실제 코드/테스트로 검증한 것)
- **P12.1-01** `create_context()`를 `app_context.py`에 명시 추가. production 전용, fake 어댑터 0건. `test_main_context_factory.py`로 검증.
- **P12.1-02** worker `_make_services()`가 `RealFfmpegAdapter`를 생성(Protocol 인스턴스화 제거). worker 계약 테스트로 검증.
- **P12.1-03** 등록 결과 이벤트가 존재하지 않는 `profile.profile_dir`를 참조하지 않는다. repository `path_for()` 사용, 이벤트는 profile_uuid/name 중심.
- **P12.1-04** 긴 파일 로드 시 기본 선택을 라벨이 아니라 실제 `wave.set_selection(0, min(duration, REFERENCE_TARGET_SECONDS))`로 반영. 10분 파일 regression 테스트(start=0, end=15) 통과. 사용자는 15초 이상도 자유 선택 가능.
- **P12.1-05** `REFERENCE_APP_MIN_SECONDS = 3.0`로 명칭/주석 정리(Qwen 공식 하드 제한 아님을 명시), UI 안내 문구 일치. 30초 이상은 경고 1회만 표시(강제 제한 없음).
- **P12.1-06** 등록 전 UI 검증: 파일 존재/이름 비어 있지 않음/중복 이름/start<end/앱 최소 길이/대사/권한/모델 다운로드/FFmpeg 사용 가능. 모델·FFmpeg 누락 시 traceback 대신 사용자 메시지.
- **P12.1-07** `_build_job()`/등록 시작의 `VoiceStudioError`/`OSError`를 `QMessageBox`로 처리. 등록 화면도 동일.
- **P12.1-08** CUDA 로드 시 `dtype=bfloat16` 명시. flash-attention은 optional로만 시도(import 실패 시 표준 attention fallback).
- **P12.1-11** FFmpeg 탐색 우선순위: frozen 시 `sys.executable` 옆 `bin/ffmpeg.exe` → 시스템 PATH.
- **P12.1-14** 미리듣기 임시 WAV를 `%LOCALAPPDATA%\VoiceStudio\cache\preview` 관리 경로로 이동, 시작 시 24시간 경과 파일 정리(삭제 실패는 다음 시작에 재시도). 미리듣기 스레드의 `preview_cache_dir` import 누락(NameError)도 수정.
- **P12.1-15** 프로필 atomic 저장(`.tmp-` → rename) 유지, `list_profiles()`는 손상/미완료 프로필을 `last_skipped`로 기록.
- **P12.1-10** `build_windows.bat`가 현재 pyproject와 일치(`.[dev]` + torch CUDA wheel 공식 경로 + qwen-tts + check_cuda.py). 존재하지 않는 `ui,audio,transcribe` extras 참조 제거.
- **P12.1-12** `packaging/VoiceStudio.spec` 추가(hiddenimports: qwen_tts/torch/transformers/tokenizers/safetensors/huggingface_hub/faster_whisper/ctranslate2/PySide6.QtMultimedia + collect_submodules/data).
- **P12.1-18** 선택 구간 regression 테스트: 600초 파일 기본 0~15초, 핸들 75~87초 등록 시 job에 실제 값 전달, 검증 차단 케이스(이름/길이/중복/모델) 전부 테스트.

### 미검증 (실기 확인 필요)
- 실제 Qwen Windows CUDA(bfloat16/flash-attn fallback), 실제 RTX GPU, worker 종료 후 nvidia-smi VRAM 반환.
- P12.1-09 모델 다운로드: 코드는 `.complete` 마커 + 핵심 파일 검증 순서로 설계되었으나 실제 snapshot_download/from_pretrained 일치는 실기 필요.
- FFmpeg 통합 테스트(실기 WAV→MP3→probe 등 5건): 이 환경에 ffmpeg가 없어 skip.
- frozen PyInstaller 빌드(spec 실제 빌드), Inno installer, bundled ffmpeg 라이선스(앱 배포용 LGPL build 선택) 확인.
- Windows startup smoke test(`pytest -m gpu` / P13 시나리오 전체).

### 테스트(정확한 숫자)
- `uv run pytest` (LD_LIBRARY_PATH에 로컬 libGL, offscreen): **84 passed, 5 skipped** (ffmpeg 실기 5 + GPU opt-in skip).
- 이 환경 한정: pytest-qt UI 실행 테스트가 libGL 없이는 불가했으나 로컬 libGL 경로로 해소.

## P12.2 Pre-Windows Validation (2026-10-06, 코드 레벨 완료 / 실기 미검증)

### 코드 레벨 완료(테스트로 검증)
- P12.2-01 AppContext.save_settings 복구(저장 후 context.settings 갱신).
- P12.2-02 MainWindow VoiceStudioError import 수정(모델 없음 → 사용자 안내, NameError 없음).
- P12.2-03 기본/사용자 지정 MP3 출력 폴더 저장·job build 시점 자동 생성(한글 경로 포함).
- P12.2-04 worker 결과를 jobs/<job_id>/result.mp3 캐시에 저장, 성공 시 결과 보존/실패·취소 시 폴더 정리.
- P12.2-05 frozen ffmpeg 탐색이 exe 옆 bin/과 _internal/bin/ 모두를 커버. spec이 ffmpeg.exe/ffprobe.exe를 bin/에 번들.
- P12.2-06 FFmpeg 배포 라이선스 결정: LGPL 빌드 배포, third_party/FFMPEG_NOTICE.txt 고지 + spec 번들.
- P12.2-07/08 spec: qwen_tts collect_data_files + ctranslate2/tokenizers/safetensors collect_all 보강.
- P12.2-09/10 dtype 객체 전달 + capability guard: RTX 2070 SUPER(Turing) 기본 torch.float16, bf16 지원 GPU만 bfloat16, CPU 강제 없음.
- P12.2-11 flash-attention 실패 시 표준 attention 1회 fallback(목표 PC 기본 OFF).
- P12.2-12/13 build_windows.bat CUDA_TAG 파라미터화, qwen-tts 설치 전후 torch 버전 기록, pip check, check_cuda 실패 시 빌드 중단 + qwen_tts import 검증.
- P12.2-14 모델 다운로드 별도 스레드(QThread)로 UI freeze 방지.
- P12.2-16 설정 다이얼로그가 저장된 bitrate를 복원.
- P12.2-17 worker result 이벤트에 result_type(audio/profile) 명시, register/narrate의 output_path 혼용 제거.
- P12.2-18 worker 성공/실패 무관 GPU cleanup을 finally로 보장.
- P12.2-21 VoiceStudio.exe --smoke-test 내부 모드 + scripts/smoke_frozen.bat(결과는 logs/smoke-test.log).
- P12.2-22 설정 화면 Self-Diagnosis 요약 + 상세는 logs/diagnosis.log(메인 프로세스 heavy import 회피, subprocess 방식).
- P12.2-23 worker 실패 진단을 logs/worker-failure.log에 기록(UI traceback 노출 금지).
- P12.2-24 REQUIRED_MODEL_FILES를 공식 HF repo 트리와 대조 검증(네트워크 가능 시, 테스트로 강제).
- P12.2-25 다운로드 atomicity: 부분 상태는 complete로 보지 않고, force 실패 시 기존 .complete 모델 보존(테스트).
- P12.2-26 production worker model_path 필수(E_MODEL_NOT_DOWNLOADED, HF 자동 다운로드 금지).
- P12.2-20 실제 사용자 경로 regression 5종(A 설정 round-trip, B 모델 없음 생성, C output dir 자동 생성, D FFmpeg 없음 안내, E 설정 재실행).
- P12.2-27 Python 3.12 호환성: PyPI cp312/universal wheel 확인 완료 → 3.12 유지(docs/07 기록).

### 실기 미검증(Windows + RTX 2070 SUPER 필요)
- frozen PyInstaller 빌드/Inno 설치 자체, bundled ffmpeg 실기 탐색.
- FP16 실제 생성 품질·속도, worker 종료 후 nvidia-smi VRAM 반환.
- P12.2-19 취소 후 ffmpeg 자식 프로세스 잔존 여부(terminate→kill fallback 존재, 실기 확인 필요).
- 모델 다운로드 실패 재시도 UX, 실제 HF 다운로드.

### 테스트
- non-GPU 전체: 106 passed / 6 skipped / 0 failed (P12.1 기준 85+5에서 증가).

## P12.3 Final Packaging Blockers

### Final Hotfix
- PyInstaller collection timing fix: `VoiceStudio.spec`은 `collect_submodules`/`collect_data_files`/`collect_all`(`spec_helpers.merge_collect`) 결과를 Analysis 생성 전에 전부 준비해 constructor에 전달한다. Analysis 이후 `a.hiddenimports/a.datas/a.binaries` 수정은 없음(AST regression 검증).
- frozen FFprobe smoke fix: `run_smoke_test()`의 FFmpeg/FFprobe 검사가 어댑터 인스턴스의 `ffmpeg`/`ffprobe` 속성과 `probe_version()`으로 실제 binary 실행 가능 여부까지 확인. 잘못된 인스턴스 `_resolve_binary` 호출 제거.
- worker slot ownership fix: MainWindow와 VoiceEditorDialog 모두 `_job_slot_acquired` ownership 상태로 자신이 acquire한 slot만 release. register 시작 거절 메시지, acquire 이후 startup 예외 시 즉시 반납, `_on_register_finished`의 try/finally 반납, closeEvent는 worker 실행 중에만 terminate(+kill fallback) 요청하고 즉시 release하지 않음.
- concurrency regression tests: narrate busy 중 register 거절, 빈 dialog close 시 slot 유지, register busy 중 narrate 거절, register finish 반납, startup 예외 반납, register 실행 중 close의 premature release 없음, MainWindow ownership.
- build script: torch/torchaudio before/after 기록을 `=== BEFORE/AFTER QWEN-TTS ===` 라벨+실제 값으로 정리.
- `check_ffmpeg.py`: GPL 구성 발견 시 WARN 대신 FAIL(exit 1). `ffmpeg -buildconf` 실행 출력을 license 근거 로그에 추가.
- 테스트: 148 passed / 6 skipped / 0 failed (hotfix 신규 regression 12개 포함).

### Final Hotfix 2
- PyInstaller spec helper import path 명시: `VoiceStudio.spec`은 `SPEC_DIR = Path(SPECPATH).resolve()`로 packaging/ 디렉터리를 계산하고, `import spec_helpers` 전에 `sys.path.insert(0, str(SPEC_DIR))`로 명시 추가한다(PyInstaller가 spec 디렉터리를 항상 import path에 넣는다고 가정하지 않음). ROOT는 `SPEC_DIR.parent`로 repository root를 가리키며 기존 SPECPATH 계약과 통합됨.
- import 경로 regression: `test_spec_adds_spec_dir_to_sys_path_before_helper_import`(insert가 import보다 먼저, SPEC_DIR=packaging/, ROOT=repository root)과 `test_spec_evaluation_imports_spec_helpers_smoke`(PyInstaller hooks/Analysis 가짜로 spec 전체 평가, spec_helpers import 성공 확인) 추가.
- 테스트: 150 passed / 6 skipped / 0 failed.
- 결과: **P12.3 Final Packaging Blockers — COMPLETE, READY FOR P13 WINDOWS GPU E2E**. 다음 단계는 실제 RTX 2070 SUPER Windows PC에서 FFmpeg 준비 → build_windows.bat → PyInstaller → frozen smoke → Inno Setup → 설치 → Qwen 모델 다운로드 → 목소리 등록 → FP16 실제 생성 → MP3 생성 → VRAM 반환 확인.

## P12.3 Final Packaging Blockers

### 코드 수준 완료
- P12.3-01/22 collect_all 반환 계약: `packaging/spec_helpers.apply_collect`로 datas/binaries/hiddenimports를 올바르게 매핑(test_p12_3_packaging).
- P12.3-02 spec repository root: SPECPATH 계약 기반 경로 존재 regression.
- P12.3-03 FFmpeg binary build prerequisite: `build_windows.bat`에서 ffmpeg.exe/ffprobe.exe 부재 시 빌드 중단.
- P12.3-04 FFmpeg license 검증 스크립트(`scripts/check_ffmpeg.py`: 실행 가능/libmp3lame/GPL 플래그/buildconf 출력).
- P12.3-05 `scripts/check_dist.py`: dist 파일 경로가 `RealFfmpegAdapter._resolve_binary` 탐색 경로와 일치 검증.
- P12.3-06/07 frozen diagnostics dispatch: `main.py --diagnostics`, `launcher.diagnostics_command`, dev/frozen 명령 형태 regression.
- P12.3-08/09/10/23 model_path 필수: register/narrate 모두 `ModelNotDownloadedError`, HF repo fallback은 명시적 opt-in만.
- P12.3-11/12 torch+torchaudio 동일 CUDA index 설치, 설치 전후 TORCH/TORCHAUDIO 버전 기록, pip check.
- P12.3-13 CUDA_TAG 환경변수로 조합 교체 가능(기본 cu126).
- P12.3-14 `check_cuda.py`에 torchaudio import 및 torch 버전 family 일치 검사 추가.
- P12.3-17 frozen smoke 확장: PySide6/worker·diagnostics dispatch/FFmpeg/FFprobe/heavy runtime(torch, torchaudio, CUDA, GPU, dtype policy, qwen_tts, Qwen3TTSModel). heavy 검사는 `heavy_smoke.py`로 분리해 일반 GUI heavy import 0건 계약 유지. 개발 환경에서는 heavy 미설치를 SKIP 처리, frozen에서는 FAIL.
- P12.3-18 성공 결과 캐시 lifecycle: 새 생성 시작 시 이전 캐시 삭제, MP3 저장 성공 후 캐시 폴더 삭제 및 `_last_output`을 사용자 파일로 교체, 앱 정상 종료 시 미저장 캐시 삭제, MP3 저장 OSError 사용자 안내.
- P12.3-19 stale jobs startup cleanup: `paths.cleanup_stale_jobs`(24h 이상 cache/jobs/<uuid> 삭제, 프로필/모델/설정 비접촉) + main 시작 호출.
- P12.3-24 VoiceStudioError positional 2-arg 오용 AST 검사 테스트.
- P12.3-25 동시 worker 1개 제한: `core/job_coordinator.JobCoordinator`를 AppContext에 두고 narrate/register 시작 시 try_acquire, 종료/닫힘 시 release. 동시 시작 거절 regression.
- P12.3-20/26/27 P13 실기 체크리스트 문서 작성(docs/08_P13_WINDOWS_GPU_CHECKLIST.md): VRAM 측정 시점 A~H, ffmpeg.exe orphan 확인, 오디오 sanity 기준.

### Windows 실기 대기(P13에서만 확인 가능)
- frozen PyInstaller 빌드/Inno 설치, bundled ffmpeg 실기 탐색, `--smoke-test`/`--diagnostics` 실기 실행.
- P12.3-21 FFMPEG_NOTICE 최종 확정: 실제 binary의 `-version` configuration/-buildconf 출력으로 license/GPL 여부/libmp3lame 확인 전까지 고지는 provisional. 현재 NOTICE에 근거 요건 명시.
- P12.3-16 torch DLL/CUDA runtime DLL의 실제 frozen 수집 범위(실기 import error 시에만 spec 보강).
- FP16 실제 생성 품질, VRAM 측정 A~H, ffmpeg orphan, 오디오 sanity.

### 테스트
- non-GPU 전체: 136 passed / 6 skipped / 0 failed (P12.2 기준 106+30 증가).

READY FOR P13 WINDOWS GPU E2E (코드 검토 레벨 blocker는 모두 처리, 남은 항목은 위 실기 대기 목록)

## ### P12.3 Final Hotfix 3: Stable ABI torch/torchaudio 검증 (2026-10-06, 실제 Windows build finding)

실제 Windows build 로그에서 torch 2.14.1+cu126 / torchaudio 2.11.0+cu126 조합이
exact-version equality 검사 때문에 false failure(CHECK_RUNTIME_PACKAGES_FAILED)로
빌드가 중단되는 것을 확인했다. TorchAudio 2.11은 PyTorch Stable ABI 기반이며
torch >= 2.11 이상(future release 포함)과 동작하므로 exact equality 요구를 제거했다.

- `scripts/version_compat.py` 신규: 공통 Stable ABI 규칙
  (torch >= 2.11, torchaudio >= 2.11, torchaudio <= torch, exact equality 미요구).
- `scripts/check_runtime_packages.py` / `scripts/check_cuda.py`: 위 helper 적용.
- SoX / flash-attn 경고는 non-fatal로 유지(번들/설치 추가 없음, P13에서 실기 확인).
- 모든 `*.bat` 파일을 ASCII-only로 정리(한글 REM/echo 제거, CMD 파싱 깨짐 방지),
  ASCII-only regression 테스트 추가.
- qwen-tts 설치 전후 조합은 실제 로그에서 동일하므로 build_windows.bat의
  설치 버전 pin은 변경하지 않음.

### P12.3 Final Hotfix 4: make_installer.bat ISCC portability (2026-10-06, 실제 Windows installer build finding)

실제 Windows에서 build_windows.bat가 성공(BUILD_OK dist\VoiceStudio)했지만
make_installer.bat가 `'iscc'은(는) 내부 또는 외부 명령...`으로 실패했다.
Inno Setup 6이 설치되어 있었으나 ISCC.exe가 PATH에 등록되지 않은 환경이었고,
기존 스크립트는 단순 `iscc` PATH 호출만 사용했다.

- `scripts/make_installer.bat`: ISCC 탐색 우선순위를
  PATH(`where iscc`) → `%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe`
  → `%ProgramFiles%\Inno Setup 6\ISCC.exe` → `%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe`
  순으로 개선. 미발견 시 명확한 `[FAIL]` + non-zero exit.
- 사용 경로 출력(`Using Inno Setup compiler:`), quoted path 실행
  (Program Files 공백 대응), `INSTALLER_OK installer\Output` 성공 마커 유지.
- `dist\VoiceStudio` prerequisite 정책 유지, ASCII-only 유지,
  사용자 이름 하드코딩 경로 금지.
- `tests/test_p12_3_installer_iscc.py` 신규 regression 9개
  (dist prerequisite, PATH 우선, fallback 3경로, 미발견 non-zero,
  quoted 실행, 실패 non-zero, 성공 마커, 하드코딩 경로 금지, ASCII-only).
- 아직 미완료(실기 대기): installer 생성 확인, 설치, 설치본 실행, 모델 다운로드,
  voice registration, 실제 TTS 생성, RTX 2070 SUPER P13, VRAM A~H.
  이 hotfix만으로 P13 COMPLETE로 표시하지 않음.

### 테스트
- `uv run pytest`: 170 passed / 6 skipped / 0 failed에서 신규 9개 증가.

Build/GPU Validation 역할 분리 (2026-10-06, 코드 레벨 완료)

### 목적
NVIDIA GPU가 없는 Windows 빌드 PC에서도 패키징이 성공하도록 build와 GPU validation을 분리.
실제 CUDA/GPU 검증은 P13 대상 PC(RTX 2070 SUPER 8GB)에서만 수행.

### 변경
- `scripts/check_runtime_packages.py` 신규: 빌드 PC용 패키지 무결성 검사
  (torch/torchaudio import, torch.version.cuda 존재, 버전 family 일치,
  qwen_tts/Qwen3TTSModel import). `torch.cuda.is_available()` / nvidia-smi는 검사하지 않음.
- `scripts/build_windows.bat`: `check_cuda.py` 필수 단계를 제거하고
  `check_runtime_packages.py`로 교체. GPU 없는 빌드 PC에서도 BUILD_OK 가능.
- `src/voice_studio/heavy_smoke.py`: `heavy_checks(checks, require_gpu=False)`로 변경.
  기본 smoke에서 CUDA available/GPU는 optional(SKIP), require_gpu=True에서만 fatal.
- `src/voice_studio/main.py`: `--smoke-test --require-gpu` strict 모드 추가.
  기본 smoke는 GPU optional(SKIP), require-gpu에서는 CUDA/GPU/dtype 실패가 FAIL이고
  dev SKIP 변환도 적용되지 않는다.
- `scripts/validate_gpu_windows.bat` 신규: P13 실기 GPU validation 스크립트
  (`check_cuda.py` + frozen `--smoke-test --require-gpu`).
- `scripts/check_cuda.py`: P13 실기 GPU validation 전용으로 역할 확정 (코드 변경 없음).

### 정책(변경 없음)
- 일반 앱 실행은 여전히 NVIDIA CUDA GPU 필수. CPU inference fallback 없음.
- dtype 정책: RTX 2070 SUPER(Turing, CC 7.5) → float16, FlashAttention-2 OFF 유지.

### 테스트
- `uv run pytest`: 158 passed / 6 skipped / 0 failed.
- 신규 regression: tests/test_p12_3_gpu_split.py 8개
  (GPU 없는 기본 smoke 성공, frozen에서 torch/qwen import 실패 시 기본 smoke 실패,
  --require-gpu + CUDA unavailable 실패/available 성공,
  build_windows.bat check_cuda 미호출·check_runtime_packages 호출,
  check_runtime_packages GPU 검사 부재, validate_gpu_windows.bat 계약, require_gpu 시그니처).

### P13 Windows Runtime Hotfix: HF progress 비활성화 + FFmpeg segment 계약 수정 (2026-10-06, 실제 설치본 실행 finding)

실제 Windows 설치본 실행에서 두 개의 runtime blocker가 확인됐다.

- 모델 다운로드 blocker: windowed PyInstaller(console=False)에서
  sys.stdout/sys.stderr가 None인데 Hugging Face Hub console progress(tqdm)가 이에 write해
  `모델 다운로드 실패: 'NoneType' object has no attribute 'write'`로 실패.
  `HF_HUB_DISABLE_PROGRESS_BARS=1` 환경변수 수동 설정 시 실제 다운로드 성공을 실기에서 확인.
- FFmpeg blocker: 선택 구간 들어보기·자동 받아쓰기·(잠재적)목소리 등록 FLAC이 모두
  `Error opening input file ...: Invalid argument`로 실패. 실제 오류:
  `[in#0] -to value smaller than -ss; aborting.` — 공통 FFmpeg segment extraction 계약 문제.

수정 내용:

- `src/voice_studio/services/model_manager.py`: `snapshot_download` 전에 공식 API
  `huggingface_hub.utils.disable_progress_bars()`를 호출해 console progress를 끈다.
  구버전 hub fallback은 프로세스 내 `HF_HUB_DISABLE_PROGRESS_BARS=1` setdefault.
  전역 stdout/stderr 교체·fake TextIO 주입·console=True 변경 없음.
  실패 시 기술 상세는 logging으로 남기고 사용자 메시지는
  `음성 모델을 받지 못했습니다. 인터넷 연결을 확인한 뒤 다시 시도해 주세요.`로 분리.
- `src/voice_studio/ui/settings_dialog.py`: `_ModelDownloadThread`가 VoiceStudioError의
  user_message만 노출하고 내부 예외는 검증된 안내 메시지로 대체, 상세는 로그.
- `src/voice_studio/infra/ffmpeg_adapter.py`: `decode_segment`/`decode_segment_to_flac`
  최종 계약을 `-ss <start> -i <input> -t <duration>`으로 변경(-to 제거).
  실행 전 `start_s >= 0`, `end_s > start_s`, `duration_s > 0` 검증, 위반 시
  FFmpeg를 실행하지 않고 `UnsupportedAudioError`. 경로는 subprocess list argument 유지
  (shell=True 없음, 한글/공백 경로 지원).
- `src/voice_studio/ui/voice_editor_dialog.py`: 들어보기 실패 안내를
  `선택한 음성 구간을 재생할 수 없습니다.\n오디오 파일과 선택 구간을 확인해 주세요.`로 변경하고
  FFmpeg detail은 로그로. 받아쓰기 실패는 원인별 분리(UnsupportedAudioError →
  `선택한 음성 구간을 읽을 수 없습니다.`, 모델 준비/네트워크 →
  `자동 받아쓰기 모델을 준비할 수 없습니다.`, 그 외 → `자동 받아쓰기를 실행할 수 없습니다.`).
- SoX/flash-attn 경고는 blocker 아님: SoX 번들/설치 금지, flash-attn 미설치 유지
  (RTX 2070 SUPER Turing CC 7.5 FP16, FlashAttention-2 OFF 정책 유지).
- `tests/test_p13_runtime_hotfix.py` 신규 regression 16개 + ffmpeg 미설치 skip 1개:
  -ss/-t 계약(0~15, 75~87), -to 부재, invalid range(동일 구간/역순/음수) subprocess 미호출,
  FLAC 동일 계약, 한글+공백 경로 list argument 유지,
  disable_progress_bars가 snapshot_download 선행, console-less(sys.stdout=None) 다운로드 성공,
  snapshot 실패 시 friendly OfflineError + partial 미완료 유지, 기존 정상 모델 보존,
  받아쓰기 안내 3분류. 실제 ffmpeg 환경 integration은 75~87초 → 약 12초 출력 검증
  (ffmpeg 미설치 환경에서는 skip).

### Hotfix 후속 정정(동일 커밋 범위, 2026-10-06)

- 전체 스위트 재검증 중 발견한 계약 불일치 2건 정정:
  1. `src/voice_studio/heavy_smoke.py`: CUDA/GPU optional 실패를
     `checks.append((name, True, "SKIP (...)"))`(ok=True)로 기록해 `main.run_smoke_test`의
     개발환경 SKIP 경로(okflag=False)가 카운트되지 않고 `skipped=2` 계약이 깨져 있었다.
     ok=False로 정정해 dev에서는 `SKIP (dev)`로 카운트, frozen/require-gpu에서는 기존과
     같이 FAIL로 처리된다(이름·detail 그대로 유지).
  2. `tests/test_p12_2_user_paths.py::test_d_ffmpeg_missing_gives_user_error`는 개발 머신
     PATH에 ffmpeg가 있으면 통과할 수 없는 환경 의존 테스트였다. `shutil.which` 차단 +
     `bundled_bin_dirs` 비움(monkeypatch)으로 환경 독립화했다.
  3. `tests/test_p12_3_gpu_split.py::test_default_smoke_succeeds_without_gpu`의
     `skipped=2` 개수 고정 계약은 ffmpeg 유무에 따라 2/4로 변하므로, SKIP 카운트 대신
     `CUDA available: SKIP` / `GPU: SKIP` / FAIL 부재를 stdout으로 검증하도록 정정했다.
- 정정 후 전체 스위트 두 조건 모두 통과:
  - ffmpeg PATH 포함: 229 passed / 2 skipped / 0 failed(실제 ffmpeg integration 실행).
  - ffmpeg PATH 미포함: 221 passed / 10 skipped / 0 failed.

아직 미완료(실기 대기, P13 진행 중):

- 수정 installer 재빌드(`scripts\build_windows.bat` → `scripts\make_installer.bat`)·재설치
- 환경변수 없이 모델 다운로드 실기 성공
- 실제 preview/STT/voice registration/Qwen voice clone generation 실기 성공
- RTX 2070 SUPER 최종 P13 E2E, VRAM A~H
- 재검증 순서는 docs/08_P13_WINDOWS_GPU_CHECKLIST.md 및 아래 P13 checklist 따름

### 테스트
- `uv run pytest`(offscreen Qt + 로컬 libGL): 195 passed / 7 skipped / 0 failed
  (기존 179 passed에서 신규 16개 증가, ffmpeg 미설치 integration skip 1개 추가).

### P13 Reference Audio / Waveform Runtime Hotfix (2026-10-06, 실제 설치본에서 M4A 파형 빈 화면 finding)

실제 Windows 설치본에서 M4A(카카오톡 수신 파일, 한글/공백 경로) 선택 시 파형이 빈 검은 화면으로
남고 선택 구간이 0.0~0.0초로 유지되는 증상이 확인됐다.

- **0.0초 실제 원인**: `_WaveformLoader.run()`이 모든 `VoiceStudioError`를
  `except VoiceStudioError: self.done.emit([], 0.0)`으로 삼켜서 실패를 빈 파형 + 0.0초 성공으로
  위장했다. 또한 `RealFfmpegAdapter.probe()`가 duration 부재 시
  `float(... or 0.0)`으로 0.0을 정상값처럼 반환했다. probe/waveform 어느 단계에서 실패했는지
  사용자도 로그도 알 수 없었다.
- 수정 내용:
  - `src/voice_studio/ui/voice_editor_dialog.py`:
    - `_WaveformLoader` 계약 변경: `done = Signal(list, float)` 성공 1회 /
      `failed = Signal(object)` 실패 1회. probe → duration 검증(NaN/inf/<=0 포함) → waveform
      순서로 분리하고 단계별 로그 기록. `VoiceStudioError`와 unexpected 예외 모두 failed로
      전달(UI crash 없음).
    - `pick_file()`은 새 파일 분석 전 `_reset_waveform_state()`로 이전 파일의 파형/선택/기간을
      완전히 초기화하고, "오디오 파일을 분석하는 중…" 상태 표시. 분석 중에는 들어보기/자동
      받아쓰기/등록이 차단된다(`_audio_ready()` guard).
    - `_on_load_failed()`는 사용자 안내
      (`오디오 파일의 재생 시간을 확인할 수 없습니다. 다른 파일을 선택하거나 오디오 파일을 확인해 주세요.`)
      + 완전 reset(duration=0, peaks clear, "선택 구간: 없음"). 기술 상세는 로그만.
    - 성공 시 기존 계약 유지: `set_peaks(peaks, duration)` 후
      `set_selection(0.0, min(duration, REFERENCE_TARGET_SECONDS))`.
  - `src/voice_studio/ui/waveform_widget.py`: `clear()` reset API 추가(파형/기간/선택 초기화,
    selection_changed 미발사).
  - `src/voice_studio/infra/ffmpeg_adapter.py`:
    - `parse_duration_seconds()`: format.duration → stream.duration 순서, 부재/NaN/inf/<=0은
      `UnsupportedAudioError`(0.0 변환 제거). probe/waveform/decode_segment 실패 시
      operation/path/exception/stderr/returncode를 logging으로 기록(기존 logging_setup 체계,
      UI에 traceback 노출 없음). waveform 디코딩 결과가 비면 0.0 버킷 대신 실패.
    - FFmpeg 호출은 계속 subprocess list argument(shell 금지), 한글/공백 경로 유지.
- `tests/test_p13_waveform_hotfix.py` 신규 regression 29개: loader 성공/실패 계약,
  invalid duration 5종은 실패 처리, probe 실패 시 waveform 미실행, UI 시나리오
  A(600초→0~15)/B(8초→0~8)/C(probe 실패→reset+안내+등록 차단)/D(waveform 실패→reset)/
  E(첫 파일 60초 성공 후 probe 실패→이전 state 재사용 금지), 분석 중 액션 차단,
  `WaveformWidget.clear()`, probe duration 파싱(format 우선/stream fallback/부재·N/A·Infinity
  실패/오디오 스트림 없음/returncode 실패), 한글+공백 M4A 경로, shell 금지 list argument.
  실제 FFmpeg integration(lavfi sine 20초 WAV probe≈20s, waveform bucket/peak,
  AAC M4A 변환 + 한글 폴더 경로 probe/waveform) 포함, ffmpeg 미설치 환경 skip.
- 기존 `tests/test_ffmpeg_real.py`는 ffmpeg 미설치로 항상 skip되어 숨어 있던 결함을 수정:
  tone 생성 시 어댑터 binary 경로 누락, `decode_reference_segment` 최소 3초 계약 위반(2.0/1.5/1.0초
  구간) → 5초 tone + 3초 구간으로 정합화. 실제 ffmpeg PATH 환경에서 4개 전부 통과 확인.
- installer 식별(선택 항목 B): `installer/voice-studio.iss` `MyAppVersion`을 0.1.0 → 0.1.1로
  bump해 기존 설치본(`VoiceStudio-Setup-0.1.0.exe`)과 새 설치본을 구분 가능. 모델/빌드 스크립트/
  PyInstaller spec/Inno Setup 로직 자체는 변경 없음.

### 테스트
- `uv run pytest`(offscreen Qt + 로컬 libGL, ffmpeg PATH 미포함):
  221 passed / 10 skipped / 0 failed (기존 195 passed에서 신규 29개 증가, ffmpeg integration skip).
- 실제 FFmpeg 7.0.2 연동 확인: `tests/test_ffmpeg_real.py` + `tests/test_p13_waveform_hotfix.py`
  = 33 passed / 0 failed (probe duration≈20s, waveform bucket 64/128, AAC M4A 한글 경로 포함).

### Hotfix 후속 정정(동일 커밋 범위, 2026-10-06)

- 전체 스위트 재검증 중 발견한 계약 불일치 2건 정정:
  1. `src/voice_studio/heavy_smoke.py`: CUDA/GPU optional 실패를
     `checks.append((name, True, "SKIP (...)"))`(ok=True)로 기록해 `main.run_smoke_test`의
     개발환경 SKIP 경로(okflag=False)가 카운트되지 않고 `skipped=2` 계약이 깨져 있었다.
     ok=False로 정정해 dev에서는 `SKIP (dev)`로 카운트, frozen/require-gpu에서는 기존과
     같이 FAIL로 처리된다(이름·detail 그대로 유지).
  2. `tests/test_p12_2_user_paths.py::test_d_ffmpeg_missing_gives_user_error`는 개발 머신
     PATH에 ffmpeg가 있으면 통과할 수 없는 환경 의존 테스트였다. `shutil.which` 차단 +
     `bundled_bin_dirs` 비움(monkeypatch)으로 환경 독립화했다.
  3. `tests/test_p12_3_gpu_split.py::test_default_smoke_succeeds_without_gpu`의
     `skipped=2` 개수 고정 계약은 ffmpeg 유무에 따라 2/4로 변하므로, SKIP 카운트 대신
     `CUDA available: SKIP` / `GPU: SKIP` / FAIL 부재를 stdout으로 검증하도록 정정했다.
- 정정 후 전체 스위트 두 조건 모두 통과:
  - ffmpeg PATH 포함: 229 passed / 2 skipped / 0 failed(실제 ffmpeg integration 실행).
  - ffmpeg PATH 미포함: 221 passed / 10 skipped / 0 failed.

아직 미완료(실기 대기, P13 진행 중):

- Windows 실기 재검증 전까지 preview/STT/registration 성공으로 표시하지 않음(기존 P13 미완료 항목 유지)
