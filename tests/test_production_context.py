"""production 컨텍스트 계약: fake 어댑터는 production 경로에 주입되지 않는다."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_studio.app_context import AppContext, dev_context
from voice_studio.infra.ffmpeg_adapter import FakeFfmpegAdapter
from voice_studio.infra.qwen_adapter import FakeQwenAdapter
from voice_studio.services.transcription_service import FakeTranscriber, FasterWhisperTranscriber
from voice_studio.services.profile_service import ProfileService


def _make_context(tmp_path):
    import os
    os.environ["VOICE_STUDIO_DATA_DIR"] = str(tmp_path / "data")
    return AppContext()


def test_production_context_uses_no_fakes(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICE_STUDIO_DATA_DIR", str(tmp_path / "data"))
    ctx = AppContext()
    assert not isinstance(ctx.audio.adapter, FakeFfmpegAdapter)  # 실제 ffmpeg 탐색 어댑터
    assert not isinstance(ctx.transcriber, FakeTranscriber)
    assert isinstance(ctx.transcriber, FasterWhisperTranscriber)
    assert isinstance(ctx.profile_service, ProfileService)
    # ProfileService는 CRUD 전용(qwen 어댑터 없음) → 등록은 worker가 담당
    assert ctx.profile_service.qwen is None


def test_fakes_only_in_dev_context(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICE_STUDIO_DATA_DIR", str(tmp_path / "data2"))
    ctx = dev_context()
    assert isinstance(ctx.profile_service.qwen, FakeQwenAdapter)
    assert isinstance(ctx.transcriber, FakeTranscriber)


def test_main_process_does_not_import_heavy_modules(tmp_path, monkeypatch):
    """UI/컨텍스트 계약: qwen_tts/torch/faster_whisper 모듈 레벨 import 없음(정적)."""
    import re
    root = Path(__file__).resolve().parents[1] / "src" / "voice_studio"
    pattern = re.compile(r"^\s*(from|import)\s+(qwen_tts|torch|faster_whisper)", re.M)
    offenders = []
    for rel in ["app_context.py", "main.py", "ui/main_window.py", "ui/voice_editor_dialog.py"]:
        text = (root / rel).read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(rel)
    assert offenders == []
