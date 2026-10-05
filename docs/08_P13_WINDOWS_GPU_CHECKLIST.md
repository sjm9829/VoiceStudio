# 08 P13 Windows + RTX 2070 SUPER 실기 검증 체크리스트(P12.3-20/26/27)

이 문서는 코드가 아니라 실기(P13)에서 순서대로 수행하고 기록할 체크리스트다.
Linux/mock 단계에서는 여기 어떤 항목도 "완료"로 기록하지 않는다.

## 환경 기록(P12.3-13)

빌드에 사용한 조합을 정확히 기록한다. 한 조합을 먼저 만들고 고정한다.

- Python:
- PyTorch:
- Torchaudio:
- CUDA wheel(CUDA_TAG):
- NVIDIA Driver:
- qwen-tts:
- transformers:

## 빌드/설치 순서

1. Windows NVIDIA driver 확인
2. repository clone
3. FFmpeg build prerequisite 준비(`third_party/bin/ffmpeg.exe`, `ffprobe.exe`)
4. `scripts/check_ffmpeg.py` 실행 → -version/-buildconf 출력 보존
5. `build_windows.bat`
6. PyInstaller 성공
7. `scripts/check_dist.py` 성공(dist 경로와 `_resolve_binary` 탐색 경로 일치)
8. `VoiceStudio.exe --smoke-test` 성공(logs/smoke-test.log에 SMOKE_OK)
9. Inno Setup installer 생성
10. 설치 후 일반 앱 실행

## VRAM 측정 체크리스트(P12.3-26)

각 시점에서 `nvidia-smi`의 GPU memory used(MiB), GPU utilization, process list를 기록한다.

| 시점 | 설명 | used MiB | util % | processes |
| --- | --- | --- | --- | --- |
| A | VoiceStudio GUI만 실행 | | | |
| B | register worker Qwen load 직후 | | | |
| C | voice prompt 생성 peak | | | |
| D | register worker 종료 후 | | | |
| E | narrate worker Qwen load | | | |
| F | 첫 chunk 생성 | | | |
| G | 긴 대본 생성 peak | | | |
| H | narrate worker 종료 후 | | | |

완료 기준: OOM 없음, worker 종료 후 Qwen process 없음, idle로 VRAM 복귀.
정확히 0 MiB일 필요는 없다(Windows desktop/driver 사용분 제외).

## FFmpeg orphan 확인(P12.3-20)

취소 테스트와 각 생성 종료 후 작업 관리자/nvidia-smi로 확인한다.

- [ ] 생성 정상 완료 후 ffmpeg.exe 잔존 없음
- [ ] 생성 취소(terminate) 후 ffmpeg.exe 잔존 없음
- [ ] worker 강제 종료(kill) 후 ffmpeg.exe 잔존 여부 기록

worker는 QProcess terminate→kill fallback을 사용한다. Windows Job Object 같은
추가 구현은 실기에서 orphan이 확인된 경우에만 검토한다.

## 생성 오디오 sanity 기준(P12.3-27)

실기 생성 결과마다 기록한다.

- [ ] PCM non-empty, duration 정상
- [ ] `np.isfinite(wav).all()` (NaN/Inf 없음)
- [ ] `np.max(np.abs(wav)) > epsilon` (완전 무음 아님)
- [ ] MP3 decode 가능(ffprobe로 duration/stereo/bitrate 확인)
- [ ] 한국어 음성이 깨지지 않음(청취)
- [ ] voice similarity 주관 테스트

## 일반 시나리오

파형 표시 → 5~15초 구간 선택 → 들어보기 → 자동 받아쓰기 → 대사 확인/수정 →
목소리 등록 → 짧은 한국어 대본 생성 → 재생 → MP3 저장 → 긴 대본 생성 → 취소 →
인터넷 차단 후 기존 모델/프로필로 오프라인 생성.
