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


# ---- P17-H1: frozen(PyInstaller) 엔진 경로 규약 ----

def test_engines_dir_frozen_uses_meipass(monkeypatch, tmp_path):
    """frozen에서 sys._MEIPASS(engine 리소스 루트)를 우선한다(PyInstaller 공식 규약)."""
    import voice_studio.core.paths as vp
    meipass = tmp_path / "app" / "_internal"
    (meipass / "engine").mkdir(parents=True)
    monkeypatch.setattr(vp.sys, "frozen", True, raising=False)
    monkeypatch.setattr(vp.sys, "_MEIPASS", str(meipass), raising=False)
    monkeypatch.setattr(vp.sys, "executable", str(tmp_path / "app" / "VoiceStudio.exe"))
    assert vp.engines_dir() == meipass / "engine"
    assert vp.gguf_engine_dir() == meipass / "engine" / "llama-cuda"


def test_engines_dir_frozen_fallback_internal_dir(monkeypatch, tmp_path):
    """_MEIPASS가 없는 구형/변형 빌드에선 exe 인접 _internal/engine을 탐색한다."""
    import voice_studio.core.paths as vp
    exe_dir = tmp_path / "app"
    internal = exe_dir / "_internal"
    (internal / "engine").mkdir(parents=True)
    monkeypatch.setattr(vp.sys, "frozen", True, raising=False)
    monkeypatch.delattr(vp.sys, "_MEIPASS") if hasattr(vp.sys, "_MEIPASS") else None
    monkeypatch.setattr(vp.sys, "executable", str(exe_dir / "VoiceStudio.exe"))
    assert vp.engines_dir() == internal / "engine"


def test_engines_dir_dev_uses_repo_src():
    """개발 환경 규약 유지: src/voice_studio/core/paths.py 기준 repo/src/engine(기존 동작 보존)."""
    from voice_studio.core import paths as vp_mod
    repo_src = Path(__file__).resolve().parents[1] / "src"
    d = vp_mod.engines_dir()
    assert d == repo_src / "engine"
