# 04. 오디오와 목소리 프로필

## 오디오 처리 (AudioService 뒤에 FFmpeg 은닉)
- ffprobe로 duration/format 검증.
- 파형: 전체 파일을 저해상도 mono로 디코딩/다운샘플해 peak envelope 생성. 배경 작업으로 UI 비활성 방지.
- 참조 구간: 선택 범위만 디코딩해 24kHz mono float32 numpy로 worker에 제공. WAV 변환 요구 금지.
- 프로필 보관: 선택 구간을 24kHz mono lossless FLAC으로 저장(마이그레이션 시 재생성). 원본 전체 파일은 복사하지 않음.
- 최종 출력: Qwen 24kHz 출력을 CPU에서 이어붙인 뒤 48kHz mono MP3로 한 번만 인코딩(libmp3lame).
  영상 편집 호환성을 위해 48kHz로 내보냅니다.
- 임시 파일: `%LOCALAPPDATA%\VoiceStudio\cache\jobs\`, 성공/취소/실패 시 정리.

## 목소리 프로필 저장
- 경로: `%LOCALAPPDATA%\VoiceStudio\profiles\<uuid>\`
- 파일: `metadata.json`, `prompt.safetensors`, `reference.flac`
- metadata: `schema_version`(필수), `name`, `uuid`, `created_at`, `updated_at`, `ref_text`,
  `reference_duration_ms`, `model_id`, `qwen_tts_version`, `x_vector_only_mode=false`, `icl_mode=true`
- safetensors에 `ref_code`, `ref_spk_embedding` 저장. 로드 시 dtype/shape 검증. pickle 임의 객체 저장 금지.
- ProfileRepository가 schema_version 기반 migration을 담당합니다.

## 긴 대본 처리
- Korean-first TextSegmenter: 문단/문장 부호 우선, 짧은 문장은 이웃과 합침.
- 기본 target 120~180자, hard max 240자(상수/설정으로 분리, 실제 샘플 테스트 후 조정).
- 한 worker가 모델을 한 번 로드한 뒤 chunk를 순차 생성(배치 기본 미사용)해 그래픽 메모리 피크를 낮춤.
- chunk 간 무음: 기본 180ms, 문단 경계 320ms(상수화).
- 모든 chunk는 동일 VoiceClonePromptItem 사용.
- 실패 chunk index를 기록하고 사용자에게 이해 가능한 오류 표시.
