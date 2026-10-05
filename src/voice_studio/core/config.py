"""생성/오디오/세그먼트 상수. 실제 샘플 테스트 후 조정 대상."""

TARGET_SEGMENT_CHARS = 150      # 목표 구간 길이(자) 120~180 중앙값
MIN_SEGMENT_CHARS = 80          # 이보다 짧은 문장은 이웃과 합침
HARD_MAX_SEGMENT_CHARS = 240    # 이 값을 넘는 구간은 강제 분할
CHUNK_GAP_MS = 180              # chunk 사이 기본 무음
PARAGRAPH_GAP_MS = 320          # 문단 경계 무음
REFERENCE_TARGET_SECONDS = 15.0 # 권장 참조 길이 (안내용, 강제 아님)
REFERENCE_WARN_SECONDS = 30.0   # 이보다 긴 선택은 경고만 표시
REFERENCE_APP_MIN_SECONDS = 3.0  # 앱 품질 보호를 위한 최소 선택 길이. Qwen 공식 API의 하드 제한을 의미하지 않음.
REFERENCE_SAMPLE_RATE = 24000   # 참조 FLAC/numpy 샘플레이트
OUTPUT_SAMPLE_RATE = 48000      # 최종 MP3 샘플레이트 (영상 편집 호환)
MP3_BITRATE_KBPS_DEFAULT = 192
MP3_BITRATE_CHOICES = (128, 192, 256)
DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
STT_MODEL_ID = "small"
STT_COMPUTE_TYPE = "int8"
ICL_MODE = True
X_VECTOR_ONLY_MODE = False
