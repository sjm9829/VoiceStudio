# -*- mode: python ; coding: utf-8 -*-
"""VoiceStudio PyInstaller 빌드 설정(P12.1-12).

Qwen/PyTorch/faster-whisper는 지연 import가 많아 자동 탐지가 누락될 수 있으므로
hiddenimports/collect를 명시한다. worker도 같은 exe의 `--worker` 모드로 실행된다.
실제 Windows 빌드 검증 전까지는 이 spec 자체가 미검증 상태로 기록된다.
"""

import sys
from pathlib import Path

block_cipher = None
ROOT = Path(SPECPATH).resolve().parent

a = Analysis(
    [str(ROOT / "src" / "voice_studio" / "main.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=[(str(ROOT / "third_party"), "third_party")],
    hiddenimports=[
        # TTS / 모델 런타임(worker 지연 import)
        "qwen_tts",
        "torch",
        "transformers",
        "tokenizers",
        "safetensors",
        "safetensors.numpy",
        "huggingface_hub",
        # STT
        "faster_whisper",
        "ctranslate2",
        # 오디오/UI
        "PySide6.QtMultimedia",
        "numpy",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
)

# 지연 import되는 대형 패키지의 서브모듈/데이터를 통째로 수집한다.
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_all
hidden = collect_submodules("qwen_tts") + collect_submodules("faster_whisper") + collect_submodules("transformers")
a.hiddenimports += [h for h in hidden if h not in a.hiddenimports]
# qwen_tts: 모델 설정/토크나이저 데이터 파일 누락 시 frozen에서 from_pretrained가 실패한다(P12.2-07).
a.datas += collect_data_files("qwen_tts", include_py_files=True)
a.datas += collect_data_files("transformers")
a.datas += collect_data_files("huggingface_hub")
# 의존성의 data/DLL 보강(P12.2-08): ctranslate2/tokenizers의 동적 라이브러리와
# tokenizers rust 확장 데이터, safetensors의 데이터 파일을 명시적으로 수집한다.
libs, bins, datas = collect_all("ctranslate2")
a.binaries += bins
a.datas += datas
libs, bins, datas = collect_all("tokenizers")
a.binaries += bins
a.datas += datas
libs, bins, datas = collect_all("safetensors")
a.binaries += bins
a.datas += datas

# FFmpeg 번들(P12.2-05/06): third_party/bin의 ffmpeg.exe/ffprobe.exe를 설치 폴더의
# bin/에 두고 RealFfmpegAdapter가 frozen에서 그 위치를 먼저 탐색한다.
ffmpeg_src = ROOT / "third_party" / "bin"
if ffmpeg_src.is_dir():
    for exe_name in ("ffmpeg.exe", "ffprobe.exe"):
        f = ffmpeg_src / exe_name
        if f.is_file():
            a.datas.append((str(f), "bin"))
# 라이선스 고지(FFMPEG_NOTICE.txt)도 함께 배포한다.
notice = ROOT / "third_party" / "FFMPEG_NOTICE.txt"
if notice.is_file():
    a.datas.append((str(notice), "bin"))

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="VoiceStudio",
    console=False,
)

coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    name="VoiceStudio",
    upx=False,
)
