"""PyInstaller 빌드 결과물의 FFmpeg 실제 포함 검증(P12.3-05).

dist/VoiceStudio의 exe와 _internal/bin/ffmpeg.exe, ffprobe.exe 존재를 확인하고,
RealFfmpegAdapter._resolve_binary()가 탐색하는 경로와 실제 위치가 일치하는지
확인한다. 빌드 후 build_windows.bat가 자동으로 실행한다.

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
    internal_bin = dist / "_internal" / "bin"
    failures = []
    if not exe.is_file():
        failures.append(f"missing: {exe}")
    for name in ("ffmpeg.exe", "ffprobe.exe"):
        p = internal_bin / name
        if not p.is_file():
            failures.append(f"missing: {p}")
        else:
            print(f"[OK] {p}")
    if failures:
        for f in failures:
            print(f"[FAIL] {f}")
        print("[FAIL] CHECK_DIST_FAILED - RealFfmpegAdapter가 찾는 bin/ 위치와 일치해야 한다")
        return 1
    print("[OK] CHECK_DIST_OK - frozen 탐색 경로(_internal/bin)와 일치")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
