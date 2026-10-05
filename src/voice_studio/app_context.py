"""애플리케이션 공용 컨텍스트. UI가 서비스 계층에 접근하는 유일한 창구."""

from __future__ import annotations
import sys
from pathlib import Path
from .core.paths import ensure_app_dirs, profiles_dir
from .core.logging_setup import setup_logging
from .infra.settings_repository import SettingsRepository
from .infra.profile_repository import ProfileRepository
from .infra.ffmpeg_adapter import FakeFfmpegAdapter
from .infra.qwen_adapter import FakeQwenAdapter
from .services.audio_service import AudioService
from .services.profile_service import ProfileService
from .services.model_manager import ModelManager
from .services.transcription_service import FakeTranscriber

class AppContext:
    """의존성 조립. 실제 어댑터(qwen/ffmpeg/whisper)는 worker/서비스 내부에서만 지연 import된다."""

    def __init__(self):
        self.paths = ensure_app_dirs()
        setup_logging()
        self.settings_repo = SettingsRepository()
        self.settings = self.settings_repo.load()
        self.profile_repository = ProfileRepository()
        self.audio = AudioService()  # 어댑터 미지정 시 첫 사용에 실제 ffmpeg 탐색
        self.model_manager = ModelManager()
        self.transcriber = FakeTranscriber()  # 실제 구현은 지연 로드(설정에서 선택)
        self.profile_service = ProfileService(self.profile_repository, self.audio, FakeQwenAdapter())

    def save_settings(self, data: dict) -> None:
        self.settings = dict(data)
        self.settings_repo.save(self.settings)

def dev_context(fake_audio: FakeFfmpegAdapter | None = None) -> AppContext:
    """테스트/개발용 가짜 컨텍스트(ffmpeg/qwen 미설치 환경)."""
    ctx = AppContext.__new__(AppContext)
    ctx.paths = ensure_app_dirs()
    ctx.settings_repo = SettingsRepository()
    ctx.settings = ctx.settings_repo.load()
    adapter = fake_audio or FakeFfmpegAdapter()
    ctx.audio = AudioService(adapter)
    qwen = FakeQwenAdapter()
    ctx.profile_repository = ProfileRepository()
    ctx.profile_service = ProfileService(ctx.profile_repository, ctx.audio, qwen)
    ctx.model_manager = ModelManager()
    ctx.transcriber = FakeTranscriber()
    return ctx
