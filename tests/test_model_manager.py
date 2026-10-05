import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voice_studio.services.model_manager import ModelManager
from voice_studio.core.errors import OfflineError, ModelNotDownloadedError

def test_not_downloaded_initially(tmp_path):
    m = ModelManager(cache_dir=tmp_path)
    assert m.is_downloaded() is False
    assert "아직" in m.status_text()

def test_marker_file_makes_downloaded(tmp_path):
    base = tmp_path / "Qwen__Qwen3-TTS-12Hz-0.6B-Base"
    base.mkdir()
    (base / ".complete").write_text("ok")
    m = ModelManager(cache_dir=tmp_path)
    assert m.is_downloaded() and m.local_snapshot() == base
    assert "받아짐" in m.status_text()

def test_download_skips_when_cached(tmp_path):
    base = tmp_path / "Qwen__Qwen3-TTS-12Hz-0.6B-Base"
    base.mkdir(); (base / ".complete").write_text("ok")
    m = ModelManager(cache_dir=tmp_path)
    assert m.download() == base

def test_download_without_hub_raises(tmp_path, monkeypatch):
    m = ModelManager(cache_dir=tmp_path)
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *a, **k):
        if name.startswith("huggingface_hub"):
            raise ImportError("no hub")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        m.download()
        assert False
    except (OfflineError, ModelNotDownloadedError):
        pass
