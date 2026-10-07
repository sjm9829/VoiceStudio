#!/usr/bin/env python3
"""FFmpeg 빌드 prerequisite 자동 준비(P13 validation correction).

third_party/bin/ffmpeg.exe, ffprobe.exe가 없으면 고정된 버전의 LGPL Windows
build를 내려받아 배치한다. 정책:

- 정상 pair가 이미 있으면 네트워크를 사용하지 않는다.
- "latest" URL 금지: 정확한 release tag / artifact / SHA-256을 코드에 고정한다.
- SHA-256 mismatch면 설치 금지, temp artifact 삭제, third_party/bin 오염 금지.
- 압축 해제는 temp directory에서 수행하고 검증 성공 후에만 배치한다.
- 기존 정상 FFmpeg가 있어도 다운로드/추출 실패 시 훼손하지 않는다.
- partial 파일이 third_party/bin에 남지 않는다(staging + os.replace).

사용: python scripts/prepare_ffmpeg.py [--force-rebuild-check 없음. idempotent]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 고정 artifact(P13 §1): BtbN FFmpeg-Builds의 특정 autobuild tag에 속한
# 정적(static, DLL 불필요) LGPL win64 build. shared build가 아니므로
# ffmpeg.exe/ffprobe.exe 두 파일만으로 실행 가능하다. libmp3lame 포함,
# --enable-gpl 없음(LGPL 배포 정책과 일치).
FFMPEG_URL = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/"
    "autobuild-2026-10-06-13-06/"
    "ffmpeg-n8.1.3-14-g330caae0c1-win64-lgpl-8.1.zip"
)
FFMPEG_SHA256 = "0b61370a3ed65970dae2624784a22a7667365431953ec1784f57eed10a06efd3"
FFMPEG_SIZE_BYTES = 170541087
FFMPEG_VERSION_NOTE = "ffmpeg n8.1.3-14-g330caae0c1 win64 LGPL static (BtbN autobuild 2026-10-06)"

BIN_DIR = ROOT / "third_party" / "bin"
FFMPEG_EXE = "ffmpeg.exe"
FFPROBE_EXE = "ffprobe.exe"


class PrepareFfmpegError(RuntimeError):
    pass


def _pair_complete(bin_dir: Path) -> bool:
    """두 binary가 모두 존재하고 0바이트가 아니면 완전한 pair로 본다."""
    for name in (FFMPEG_EXE, FFPROBE_EXE):
        p = bin_dir / name
        if not p.is_file() or p.stat().st_size == 0:
            return False
    return True


def _download_to(url: str, dest: Path, size_hint: int) -> None:
    """지정 파일로 스트리밍 다운로드. 실패 시 partial 파일을 남기지 않는다."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
        if size_hint and tmp.stat().st_size != size_hint:
            raise PrepareFfmpegError(
                f"downloaded size mismatch: {tmp.stat().st_size} != {size_hint}")
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def default_downloader(url: str, dest: Path) -> None:
    _download_to(url, dest, FFMPEG_SIZE_BYTES)


def default_extractor(archive: Path, out_dir: Path) -> None:
    """zip 안의 bin/ffmpeg.exe, bin/ffprobe.exe만 out_dir로 추출한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        wanted = {Path(n).name for n in z.namelist()
                  if n.endswith(".exe") and Path(n).name in (FFMPEG_EXE, FFPROBE_EXE)}
        missing = {FFMPEG_EXE, FFPROBE_EXE} - wanted
        if missing:
            raise PrepareFfmpegError(f"archive missing binaries: {sorted(missing)}")
        for info in z.infolist():
            name = Path(info.filename).name
            if name in wanted:
                with z.open(info) as src, open(out_dir / name, "wb") as fh:
                    shutil.copyfileobj(src, fh)


def verify_sha256(path: Path, expected: str) -> None:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != expected:
        raise PrepareFfmpegError(
            f"SHA-256 mismatch: {h.hexdigest()} != {expected}")


def prepare(bin_dir: Path = BIN_DIR,
            downloader=default_downloader,
            extractor=default_extractor,
            verifier=verify_sha256,
            url: str = FFMPEG_URL,
            expected_sha256: str = FFMPEG_SHA256) -> None:
    """정상 pair가 있으면 아무 것도 하지 않고, 없으면 안전하게 준비한다.

    - 다운로드 → SHA-256 검증 → temp 추출 → staging 복사 → os.replace 순서.
    - 어떤 단계에서 실패해도 기존 third_party/bin 내용은 훼손되지 않는다.
    """
    if _pair_complete(bin_dir):
        print(f"[OK] FFmpeg pair already prepared: {bin_dir}")
        print(f"     {FFMPEG_VERSION_NOTE}")
        return
    print(f"[INFO] FFmpeg pair incomplete in {bin_dir}. Preparing fixed artifact:")
    print(f"       {url}")
    bin_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="prepare_ffmpeg_"))
    try:
        archive = tmp_root / "ffmpeg.zip"
        print("[INFO] downloading...")
        downloader(url, archive)
        print("[INFO] verifying SHA-256...")
        verifier(archive, expected_sha256)
        extract_dir = tmp_root / "extract"
        print("[INFO] extracting to temp...")
        extractor(archive, extract_dir)
        staged = []
        try:
            for name in (FFMPEG_EXE, FFPROBE_EXE):
                src = extract_dir / name
                if not src.is_file() or src.stat().st_size == 0:
                    raise PrepareFfmpegError(f"extracted binary missing/empty: {name}")
                staging = bin_dir / f".{name}.staging-{os.getpid()}"
                shutil.copyfile(src, staging)
                staged.append((staging, bin_dir / name))
            # 검증이 모두 끝난 뒤에만 교체(atomic replace, partial 없음)
            for staging_path, final_path in staged:
                os.replace(staging_path, final_path)
        finally:
            for staging_path, _ in staged:
                if staging_path.exists():
                    staging_path.unlink(missing_ok=True)
        print(f"[OK] FFmpeg pair prepared: {bin_dir}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)
    try:
        prepare()
    except PrepareFfmpegError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
