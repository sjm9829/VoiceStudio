# -*- mode: python ; coding: utf-8 -*-
"""VoiceStudio PyInstaller 빌드 설정(P12.1-12).

Qwen/PyTorch/faster-whisper는 지연 import가 많아 자동 탐지가 누락될 수 있으므로
hiddenimports/collect를 명시한다. worker도 같은 exe의 `--worker` 모드로 실행된다.
실제 Windows 빌드 검증 전까지는 이 spec 자체가 미검증 상태로 기록된다.
"""

import sys
from pathlib import Path

block_cipher = None
# P12.3-02: 실제 PyInstaller 규칙상 SPECPATH는 spec 파일이 있는 packaging/ 디렉터리이므로
# repository root는 그 부모다. 계약은 packaging/spec_helpers.compute_repo_root와
# tests/test_p12_3_packaging.py 로 검증한다.
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
# P12.3-01/22: collect_all 반환 계약(datas, binaries, hiddenimports)은
# packaging/spec_helpers.py의 테스트 가능한 헬퍼로만 사용한다.
import spec_helpers
hidden = collect_submodules("qwen_tts") + collect_submodules("faster_whisper") + collect_submodules("transformers")
a.hiddenimports += [h for h in hidden if h not in a.hiddenimports]
# qwen_tts: 모델 설정/토크나이저 데이터 파일 누락 시 frozen에서 from_pretrained가 실패한다(P12.2-07).
a.datas += collect_data_files("qwen_tts", include_py_files=True)
a.datas += collect_data_files("transformers")
a.datas += collect_data_files("huggingface_hub")
# 의존성의 data/DLL 보강(P12.2-08/P12.3-01): ctranslate2/tokenizers/safetensors의
# 동적 라이브러리와 데이터 파일을 올바른 매핑으로 수집한다.
for _pkg in ("ctranslate2", "tokenizers", "safetensors"):
    spec_helpers.apply_collect(a, _pkg)

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
