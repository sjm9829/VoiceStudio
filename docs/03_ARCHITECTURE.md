# 03. 아키텍처

## 프로세스 구조
```
[PySide6 메인 프로세스]  ←→  [Qwen worker 프로세스]
      │ QProcess + stdout JSONL        │ job JSON stdin 입력
      │                                │ CUDA + Qwen 모델 로드
      └ ffmpeg/ffprobe(AudioService)    └ 완료/취소/오류 후 반드시 종료
```

- 메인 프로세스는 Qwen/CUDA 모듈을 import·load하지 않습니다. 평상시 그래픽 메모리 점유 0.
- worker는 job JSON을 입력받고 stdout으로 JSONL 이벤트(status/progress/error/result)를 출력합니다.
- worker 완료/취소/오류 시 프로세스 종료로 CUDA context를 소멸시킵니다. `empty_cache()`에 의존하지 않습니다.
- worker 크래시가 UI 크래시로 이어지지 않게 QProcess 시그널로 분리하고 로그/사용자 메시지를 분리합니다.

## 레이어
| 레이어 | 위치 | 내용 |
|---|---|---|
| core | `src/voice_studio/core/` | paths, config, logging, errors |
| domain | `domain/` | VoiceProfile, GenerationJob, enums |
| services | `services/` | AudioService, ProfileService, ModelManager, NarrationService, TranscriptionService, TextSegmenter |
| infra | `infra/` | profile repository, ffmpeg adapter, qwen adapter, settings repository |
| ui | `ui/` | MainWindow, VoiceManagerDialog, VoiceEditorDialog, SettingsDialog, WaveformWidget |
| workers | `workers/` | worker_main.py, job/event 스키마 |

- UI는 qwen/ffmpeg/faster-whisper를 직접 import하지 않습니다.

## Qwen 어댑터 캡슐화 (공식 API 함정 대응)
- 공식 `_prompt_items_to_voice_clone_prompt()` dict에는 ref_text가 포함되지 않으므로,
  프로필에 `ref_code`, `ref_spk_embedding`, `x_vector_only_mode`, `icl_mode`, `ref_text`를 모두 영속화하고
  생성 시 공식 `VoiceClonePromptItem`을 재구성해 **list로 전달**하는 경로를 사용합니다.
- 이 복원/재구성은 qwen 어댑터 안에 캡슐화하고 private API 종속을 어댑터로 국한합니다.
- 모델 사용 전 공식 코드를 재확인합니다(P00). 불확실하면 문서에 기록합니다.

## 모델 관리
- huggingface_hub `snapshot_download`, 진행률 표시, 중단/재시도 구조화.
- 앱 시작 시 강제 다운로드 금지. `음성 생성`/`목소리 등록` 최초 사용 또는 설정 `모델 받기`에서 다운로드.
- 인터넷 없음 + 모델 없음 → 오프라인 안내.
