"""P12.1-17: 앱 시작 스모크 — 메인 창 생성 시 torch/CUDA/Qwen/Whisper 모델 로드 없음."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

pytest.importorskip("PySide6", reason="PySide6 미설치 환경")


def test_startup_loads_no_heavy_models(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICE_STUDIO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    for m in ("torch", "qwen_tts", "faster_whisper", "ctranslate2"):
        monkeypatch.delitem(sys.modules, m, raising=False)

    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from voice_studio.app_context import create_context
    from voice_studio.ui.main_window import MainWindow
    ctx = create_context()
    win = MainWindow(ctx)
    win.show()
    win.close()

    loaded = [m for m in ("torch", "qwen_tts", "faster_whisper", "ctranslate2") if m in sys.modules]
    assert loaded == [], f"앱 시작에 heavy 모듈이 로드됨: {loaded}"
    # production fake 0건
    from voice_studio.infra.ffmpeg_adapter import FakeFfmpegAdapter
    assert not isinstance(ctx.audio.adapter, FakeFfmpegAdapter)
