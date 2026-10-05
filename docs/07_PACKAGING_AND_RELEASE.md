# 07. 패키징과 릴리스

## 방식
- PyInstaller `onedir` (거대 onefile 미사용) + Inno Setup 설치 프로그램.
- 모델 가중치는 설치 프로그램에 포함하지 않고 첫 필요 시 공식 Hugging Face에서 내려받음.
- FFmpeg 바이너리는 설치 프로그램에 포함하거나, 없으면 첫 실행 안내로 제공.

## 스크립트
- `scripts/build_windows.bat`: venv 생성 → 의존성 설치 → PyInstaller onedir 빌드
- `scripts/make_installer.bat`: Inno Setup(iscc) 실행
- `scripts/check_cuda.py`: CUDA/그래픽 카드 진단(개발 모드)

## 라이선스/고지
- `third_party/LICENSES.md`: FFmpeg(libmp3lame 포함), PySide6, faster-whisper, PyInstaller 등
- `third_party/MODEL_NOTICE.md`: Qwen 모델 Apache-2.0 + Qwen 고지

## 릴리스 체크리스트
1. `pytest -m "not gpu"` 전부 통과
2. Windows 실기 빌드 + 설치 + 실행 + 생성 E2E (Windows validation pending 관리)
3. 한글 경로/공백 경로/긴 파일명 실기 확인
4. 설치 후 재실행 시 profile 유지 확인
5. 제거 시 사용자 데이터 정책 문서화(기본 유지)
