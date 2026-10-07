import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pytest
from voice_studio.infra.ffmpeg_adapter import FakeFfmpegAdapter
from voice_studio.infra.profile_repository import ProfileRepository
from voice_studio.infra.qwen_adapter import FakeQwenAdapter
from voice_studio.services.audio_service import AudioService
from voice_studio.services.profile_service import ProfileService

@pytest.fixture
def fake_audio(tmp_path):
    return FakeFfmpegAdapter(duration=30.0)

@pytest.fixture
def services(tmp_path, fake_audio):
    qwen = FakeQwenAdapter()
    audio = AudioService(fake_audio)
    repo = ProfileRepository(tmp_path / "profiles")
    return ProfileService(repo, audio, qwen), repo, audio, qwen

@pytest.fixture
def registered(services, tmp_path):
    service, repo, audio, qwen = services
    src = str(tmp_path / "ref.wav")
    p = service.register(
        name="테스트 목소리", source_path=src, start_s=1.0, end_s=16.0,
        ref_text="안녕하세요. 테스트 대사입니다.", consent=True)
    return service, repo, audio, qwen, p

@pytest.fixture(autouse=True)
def _isolated_app_data(tmp_path, monkeypatch):
    """모든 테스트의 app data를 temp로 격리해 실제 사용자 데이터를 보호한다.

    env 자체를 검증하는 tests/test_paths.py는 자체 monkeypatch로
    override/unset 상황을 재정의하므로 이 fixture와 충돌하지 않는다.
    """
    monkeypatch.setenv("VOICE_STUDIO_DATA_DIR", str(tmp_path / "appdata"))
