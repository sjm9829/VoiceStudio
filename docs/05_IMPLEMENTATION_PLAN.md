# 05. 구현 계획 (P00~P11)

| Phase | 내용 | 검증 |
|---|---|---|
| P00 | Qwen 공식 API/Windows CUDA/Python 3.12/FFmpeg/faster-whisper/PySide6 사실확인. 최소 Qwen smoke script | 문서 기록, smoke 스크립트 |
| P01 | scaffold/domain/config/paths/logging | 단위 테스트 |
| P02 | AudioService/ffprobe/segment decode/waveform/MP3 encode | fake adapter 테스트 |
| P03 | ProfileRepository/safetensors/schema/migration | 단위 테스트 |
| P04 | Qwen adapter + worker protocol + prompt 생성/저장/복원 | fake adapter unit test, opt-in GPU smoke |
| P05 | PySide6 shell + 메인 화면 + QProcess 진행률/취소 | pytest-qt |
| P06 | 목소리 관리/등록 dialog + waveform selector + playback + 동의 + CRUD | pytest-qt |
| P07 | faster-whisper CPU 자동 받아쓰기, lazy model manager | Transcriber 인터페이스 테스트 |
| P08 | 긴 대본 segmenter + 순차 생성 + chunk join + 최종 MP3 | segmenter 단위 테스트 |
| P09 | 설정/model download/오프라인·오류 UX | 단위 테스트 |
| P10 | 통합 테스트/장애 복구/취소/임시 정리/worker 종료 검증 | 통합 테스트 |
| P11 | Windows packaging(PyInstaller onedir + Inno Setup), licenses/notices, release checklist | 스크립트 + `Windows validation pending` 명시 |

각 Phase 완료 시 `IMPLEMENTATION_STATUS.md`를 갱신하고 의미 있는 단위로 commit합니다.
사용자 확인을 기다리지 않고 다음 Phase로 자율 진행합니다(파괴적/외부 변경 제외).
