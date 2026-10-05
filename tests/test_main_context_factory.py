"""P12.1-01: create_context() production 팩토리 계약."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_create_context_import_and_type(tmp_path, monkeypatch):
    from voice_studio.app_context import AppContext, create_context
    monkeypatch.setenv("VOICE_STUDIO_DATA_DIR", str(tmp_path / "data"))
    ctx = create_context()
    assert isinstance(ctx, AppContext)


def test_create_context_has_no_fakes(tmp_path, monkeypatch):
    """production context에 fake 어댑터(FakeQwen/FakeTranscriber/FakeFfmpeg)가 없어야 한다."""
    from voice_studio.app_context import create_context
    from voice_studio.infra.ffmpeg_adapter import FakeFfmpegAdapter
    from voice_studio.infra.qwen_adapter import FakeQwenAdapter
    from voice_studio.services.transcription_service import FakeTranscriber
    monkeypatch.setenv("VOICE_STUDIO_DATA_DIR", str(tmp_path / "data2"))
    ctx = create_context()
    fakes = (FakeFfmpegAdapter, FakeQwenAdapter, FakeTranscriber)
    for attr in ("audio", "transcriber", "profile_service", "model_manager"):
        obj = getattr(ctx, attr)
        assert not isinstance(obj, fakes), f"{attr}에 fake가 주입되었다"
