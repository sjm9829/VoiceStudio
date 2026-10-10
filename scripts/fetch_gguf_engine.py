#!/usr/bin/env python3
"""P17-D: Windows 빌드 전 GGUF 엔진(llama.cpp b11540 CUDA)을 준비한다.

packaging/engine-src/llama-cuda에 llama-tts.exe와 필요한 DLL을 풀어 둔다.
PyInstaller spec은 이 디렉터리가 있으면 번들에 포함한다. 고정 release만 사용한다.
"""
from __future__ import annotations
import shutil
import sys
import zipfile
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from voice_studio.core import gguf as cfg  # noqa: E402

DEST = ROOT / "packaging" / "engine-src" / "llama-cuda"


def main() -> int:
    base = (f"https://github.com/ggml-org/llama.cpp/releases/download/"
            f"{cfg.LLAMA_RELEASE_TAG}")
    if (DEST / ".complete").is_file() and any(DEST.glob("llama-tts*")):
        print(f"engine already prepared: {DEST}")
        return 0
    tmp = DEST.with_suffix(".tmp")
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for name in (cfg.LLAMA_WIN_CUDA_ASSET, cfg.LLAMA_CUDART_ASSET):
        print("downloading", name)
        dest = tmp / name
        with urllib.request.urlopen(f"{base}/{name}", timeout=120) as resp, open(dest, "wb") as fh:
            shutil.copyfileobj(resp, fh, length=1 << 22)
        with zipfile.ZipFile(dest) as zf:
            zf.extractall(tmp / "unpacked")
        dest.unlink()
    for item in (tmp / "unpacked").iterdir():
        shutil.move(str(item), str(tmp / item.name))
    (tmp / ".complete").write_text(cfg.LLAMA_RELEASE_TAG)
    if DEST.exists():
        shutil.rmtree(DEST, ignore_errors=True)
    tmp.rename(DEST)
    print("prepared", DEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
