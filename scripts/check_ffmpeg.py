"""FFmpeg 빌드 prerequisite license/buildconf 검증(P12.3-04).

빌드에 공급할 third_party/bin/ffmpeg.exe, ffprobe.exe가 실제로 실행 가능한지,
MP3 인코더(libmp3lame)가 있는지, GPL 구성인지, build configuration은 무엇인지
확인한다. license 문서는 이 스크립트의 실제 출력을 근거로만 작성한다(P12.3-21).

사용: python scripts/check_ffmpeg.py [ffmpeg.exe 경로]
기본값: third_party/bin/ffmpeg.exe (repository root 기준)
"""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(exe: Path, *args: str) -> str:
    try:
        r = subprocess.run([str(exe), *args], capture_output=True, text=True,
                           timeout=60, errors="replace")
    except OSError as e:
        print(f"[FAIL] {exe.name} 실행 불가: {e}")
        raise SystemExit(1)
    if r.returncode != 0:
        print(f"[FAIL] {exe.name} {' '.join(args)} exit={r.returncode}")
        raise SystemExit(1)
    return r.stdout + r.stderr


def main() -> int:
    exe = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "third_party" / "bin" / "ffmpeg.exe"
    probe = exe.with_name("ffprobe.exe")
    if not exe.is_file():
        print(f"[FAIL] ffmpeg binary 없음: {exe}")
        return 1
    if not probe.is_file():
        print(f"[FAIL] ffprobe binary 없음: {probe}")
        return 1

    version = _run(exe, "-version")
    print("=== ffmpeg -version ===")
    print(version)
    config_line = next((l for l in version.splitlines() if l.strip().startswith("configuration:")), "")
    # P12.3 Final Hotfix: 배포 정책은 GPL build 미배포이므로 GPL 구성은 빌드를 차단한다.
    if "--enable-gpl" in config_line:
        print("[FAIL] GPL-enabled FFmpeg build는 현재 배포 정책에서 허용하지 않습니다.")
        return 1
    print("[INFO] GPL 플래그 없음(LGPL 계열로 보임). buildconf를 근거로 고지를 확정할 것.")

    # P12.3 Final Hotfix: -version configuration뿐 아니라 실제 -buildconf도 실행해 기록한다.
    buildconf = _run(exe, "-buildconf")
    print("=== ffmpeg -buildconf ===")
    print(buildconf)

    encoders = _run(exe, "-hide_banner", "-encoders")
    if "libmp3lame" in encoders:
        print("[OK] libmp3lame encoder 존재")
    else:
        print("[FAIL] libmp3lame encoder 없음 - MP3 저장 불가")
        return 1

    _run(probe, "-version")
    print("[OK] ffprobe 실행 가능")
    print("[OK] CHECK_FFMPEG_OK")
    print("위 -version/configuration 출력을 third_party/FFMPEG_NOTICE.txt 와")
    print("docs/07_PACKAGING_AND_RELEASE.md 의 license 근거로 그대로 기록하십시오.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
