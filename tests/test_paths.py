import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voice_studio.core.paths import (app_data_dir, profiles_dir, jobs_cache_dir,
                                     logs_dir, default_mp3_dir, safe_job_cache_dir)

def test_app_data_dir_under_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICE_STUDIO_DATA_DIR", str(tmp_path))
    assert app_data_dir() == tmp_path / "VoiceStudio"
    assert profiles_dir() == tmp_path / "VoiceStudio" / "profiles"
    assert jobs_cache_dir() == tmp_path / "VoiceStudio" / "cache" / "jobs"

def test_default_mp3_dir_korean_name():
    assert default_mp3_dir().name == "보이스 스튜디오"

def test_job_cache_dir_creates_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICE_STUDIO_DATA_DIR", str(tmp_path))
    d = safe_job_cache_dir("abc-123")
    assert d.is_dir() and "abc-123" in str(d)

def test_korean_space_paths(tmp_path):
    weird = tmp_path / "한글 폴더" / "긴 파일명 테스트 아주 긴 이름입니다.wav"
    weird.parent.mkdir(parents=True)
    weird.write_bytes(b"x")
    assert weird.exists() and len(weird.name) > 10


def test_override_wins_over_localappdata(tmp_path, monkeypatch):
    """A: VOICE_STUDIO_DATA_DIR이 Windows LOCALAPPDATA보다 우선한다."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "real-localappdata"))
    monkeypatch.setenv("VOICE_STUDIO_DATA_DIR", str(tmp_path / "override"))
    assert app_data_dir() == tmp_path / "override" / "VoiceStudio"

def test_windows_localappdata_without_override(monkeypatch, tmp_path):
    """B: override가 없고 Windows면 LOCALAPPDATA/VoiceStudio."""
    import voice_studio.core.paths as vp
    monkeypatch.delenv("VOICE_STUDIO_DATA_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "la"))
    monkeypatch.setattr(vp.sys, "platform", "win32")
    assert vp.app_data_dir() == tmp_path / "la" / "VoiceStudio"

def test_nonwindows_default_without_override(monkeypatch, tmp_path):
    """B(비-Windows): override가 없으면 ~/.local/state/VoiceStudio."""
    import voice_studio.core.paths as vp
    monkeypatch.delenv("VOICE_STUDIO_DATA_DIR", raising=False)
    monkeypatch.setattr(vp.sys, "platform", "linux")
    assert vp.app_data_dir() == Path.home() / ".local" / "state" / "VoiceStudio"

def test_derived_dirs_under_override(tmp_path, monkeypatch):
    """C: profiles/jobs/logs/preview 모두 override 아래에 위치한다."""
    monkeypatch.setenv("VOICE_STUDIO_DATA_DIR", str(tmp_path / "override"))
    base = tmp_path / "override" / "VoiceStudio"
    assert profiles_dir() == base / "profiles"
    assert jobs_cache_dir() == base / "cache" / "jobs"
    assert logs_dir() == base / "logs"
    from voice_studio.core.paths import preview_cache_dir
    assert preview_cache_dir() == base / "cache" / "preview"

def test_safe_job_cache_dir_isolated_from_real_user_data(tmp_path, monkeypatch):
    """D: safe_job_cache_dir는 실제 사용자 LOCALAPPDATA를 절대 건드리지 않는다."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "real-la"))
    monkeypatch.setenv("VOICE_STUDIO_DATA_DIR", str(tmp_path / "iso"))
    d = safe_job_cache_dir("abc-123")
    assert d.is_dir() and str(d).startswith(str(tmp_path / "iso"))
    real = tmp_path / "real-la" / "VoiceStudio"
    assert not real.exists()
