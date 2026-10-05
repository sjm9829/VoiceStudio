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
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
hidden = collect_submodules("qwen_tts") + collect_submodules("faster_whisper") + collect_submodules("transformers")
a.hiddenimports += [h for h in hidden if h not in a.hiddenimports]
a.datas += collect_data_files("transformers")
a.datas += collect_data_files("huggingface_hub")

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
