import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voice_studio.core.paths import (app_data_dir, profiles_dir, jobs_cache_dir,
                                     default_mp3_dir, safe_job_cache_dir)

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
