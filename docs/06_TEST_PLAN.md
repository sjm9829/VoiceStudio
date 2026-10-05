# 06. 테스트 계획

## 완료 기준 매핑
| 요구 | 검증 방법 |
|---|---|
| idle 상태 worker/그래픽 메모리 없음 | 메인 프로세스가 qwen 모듈을 import하지 않는 구조 테스트 + 진단 스크립트 |
| 참조 파일 로드/파형/대사/등록 | pytest-qt + fake AudioService |
| 재실행 후 profile 로드·ICL 생성 | ProfileRepository 영속 테스트 + fake qwen 어댑터 |
| 10분 대본 자동 분할→MP3 | segmenter 단위 테스트 + fake worker 통합 테스트 |
| 취소/크래시/모델 부족/FFmpeg 없음/디스크 부족 | 오류 경로 테스트, UI freeze 없음(pytest-qt) |
| 생성 후 worker 종료 | QProcess 종료 시그널 테스트 |
| 한글/공백/긴 파일명 | 경로 fixture 테스트 |
| 특정 개인 표현 0건 | `아빠/아버지/father` 문자열 스캔 테스트 |
| 산출물 MP3 하나 | NarrationService 출력 계약 테스트 |

## 규칙
- 유닛 테스트는 실제 2.5GB 모델 없이 동작(fake adapter).
- 실제 모델 테스트는 opt-in `gpu` marker(`pytest -m gpu`).
- GPU/Windows 검증이 불가한 환경에서는 mock으로 성공 처리하지 않고 `미검증`으로 기록.
