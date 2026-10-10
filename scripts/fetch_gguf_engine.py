#!/usr/bin/env python3
"""P17-D/H1: Windows 빌드 전 GGUF 엔진(llama.cpp b11540 CUDA)을 준비한다.

packaging/engine-src/llama-cuda에 llama-tts.exe와 필요한 DLL을 풀어 둔다.
PyInstaller spec은 이 디렉터리가 있으면 번들에 포함한다. 고정 release만 사용하며
SHA-256 검증을 통과한 zip만 압축을 푼다. 실패 시 기존 정상 엔진을 보존한다.
"""
from __future__ import annotations
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from voice_studio.core import gguf as cfg  # noqa: E402

DEST = ROOT / "packaging" / "engine-src" / "llama-cuda"
ASSET_SHA256 = {
    cfg.LLAMA_WIN_CUDA_ASSET: cfg.LLAMA_WIN_CUDA_SHA256,
    cfg.LLAMA_CUDART_ASSET: cfg.LLAMA_CUDART_SHA256,
}


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _engine_ok(directory: Path) -> bool:
    """성공 판정: llama-tts 실행 파일 존재 + .complete 마커."""
    return (directory / ".complete").is_file() and any(directory.glob("llama-tts*"))


def main() -> int:
    if _engine_ok(DEST):
        print(f"engine already prepared: {DEST}")
        return 0
    base = (f"https://github.com/ggml-org/llama.cpp/releases/download/"
            f"{cfg.LLAMA_RELEASE_TAG}")
    tmp = DEST.with_name(DEST.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        for name, expected in ASSET_SHA256.items():
            print("downloading", name)
            dest = tmp / name
            with urllib.request.urlopen(f"{base}/{name}", timeout=120) as resp, open(dest, "wb") as fh:
                shutil.copyfileobj(resp, fh, length=1 << 22)
            actual = _sha256(dest)
            if actual != expected:
                raise RuntimeError(f"SHA-256 mismatch for {name}: {actual} != {expected}")
            print(f"sha256 ok: {name}")
            with zipfile.ZipFile(dest) as zf:
                zf.extractall(tmp / "unpacked")
            dest.unlink()
        for item in (tmp / "unpacked").iterdir():
            shutil.move(str(item), str(tmp / item.name))
        (tmp / "unpacked").rmdir()
        (tmp / ".complete").write_text(cfg.LLAMA_RELEASE_TAG)
        if not any((tmp).glob("llama-tts*")):
            raise RuntimeError("llama-tts binary not found in unpacked release")
        # 성공 시에만 정식 경로 교체(atomic swap) - 기존 정상 엔진 보존
        old = DEST.with_name(DEST.name + ".old")
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)
        if DEST.exists():
            shutil.move(str(DEST), str(old))
        tmp.rename(DEST)
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)
    except Exception as exc:
        print(f"engine prepare failed: {exc}", file=sys.stderr)
        shutil.rmtree(tmp, ignore_errors=True)
        if not _engine_ok(DEST):
            return 1
        print("keeping previous working engine", file=sys.stderr)
        return 0
    print("prepared", DEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
