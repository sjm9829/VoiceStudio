"""PyInstaller 빌드 결과물의 FFmpeg/GGUF 엔진 실제 포함 검증(P12.3-05, P17-H1).

dist/VoiceStudio의 exe, _internal/bin/ffmpeg.exe, ffprobe.exe, 그리고
_internal/engine/llama-cuda의 llama-tts.exe와 주요 CUDA DLL 존재를 확인한다.
빌드 후 build_windows.bat이 자동으로 실행한다.

사용: python scripts/check_dist.py [dist 디렉터리]
기본값: dist/VoiceStudio
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    dist = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist" / "VoiceStudio"
    exe = dist / "VoiceStudio.exe"
    internal = dist / "_internal"
    internal_bin = dist / "_internal" / "bin"
    engine = internal / "engine" / "llama-cuda"
    failures = []
    if not exe.is_file():
        failures.append(f"missing: {exe}")
    for name in ("ffmpeg.exe", "ffprobe.exe"):
        p = internal_bin / name
        if not p.is_file():
            failures.append(f"missing: {p}")
        else:
            print(f"[OK] {p}")
    # P17-H1: GGUF 음성 엔진 실제 포함 검증. llama-tts.exe와 CUDA 런타임 DLL.
    tts = engine / "llama-tts.exe"
    if not tts.is_file():
        failures.append(f"missing: {tts}")
    else:
        print(f"[OK] {tts}")
    if not (engine / ".complete").is_file():
        failures.append(f"missing: {engine / '.complete'} (엔진 준비 마커)")
    for dll in ("cudart64_12.dll",):
        p = engine / dll
        if not p.is_file():
            # cudart는 zip 해제 결과 이름이 다를 수 있으므로 glob 후보도 인정
            candidates = list(engine.glob("cudart64*.dll"))
            if candidates:
                print(f"[OK] cudart runtime: {candidates[0].name}")
                continue
            failures.append(f"missing: {p} (cudart 계열 DLL 없음)")
        else:
            print(f"[OK] {p}")
    # cuBLAS 등 llama.cpp 필수 DLL 대표 검증(이름은 release zip 기준 유동적)
    if not any(engine.glob("cublas*.dll")):
        failures.append(f"missing: cublas*.dll in {engine}")
    if failures:
        for f in failures:
            print(f"[FAIL] {f}")
        print("[FAIL] CHECK_DIST_FAILED - frozen 탐색 경로(_internal)와 실제 배치가 일치해야 한다")
        return 1
    print("[OK] CHECK_DIST_OK - frozen 탐색 경로(_internal/bin, _internal/engine)와 일치")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
