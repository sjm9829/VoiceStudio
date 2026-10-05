import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voice_studio.services.model_manager import ModelManager
from voice_studio.core.errors import OfflineError, ModelNotDownloadedError

def test_not_downloaded_initially(tmp_path):
    m = ModelManager(cache_dir=tmp_path)
    assert m.is_downloaded() is False
    assert "아직" in m.status_text()

def _make_snapshot(base):
    """마커 + 실제 실행에 필요한 핵심 파일을 모두 갖춘 로컬 스냅샷을 만든다."""
    base.mkdir()
    (base / ".complete").write_text("ok")
    from voice_studio.services.model_manager import REQUIRED_MODEL_FILES
    for rel in REQUIRED_MODEL_FILES:
        f = base / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x", encoding="utf-8")
    return base

def test_marker_file_makes_downloaded(tmp_path):
    base = _make_snapshot(tmp_path / "Qwen__Qwen3-TTS-12Hz-0.6B-Base")
    m = ModelManager(cache_dir=tmp_path)
    assert m.is_downloaded() and m.local_snapshot() == base
    assert "받아짐" in m.status_text()

def test_marker_without_required_files_is_not_downloaded(tmp_path):
    base = tmp_path / "Qwen__Qwen3-TTS-12Hz-0.6B-Base"
    base.mkdir(); (base / ".complete").write_text("ok")
    m = ModelManager(cache_dir=tmp_path)
    assert m.is_downloaded() is False and m.local_snapshot() is None

def test_download_skips_when_cached(tmp_path):
    base = _make_snapshot(tmp_path / "Qwen__Qwen3-TTS-12Hz-0.6B-Base")
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
