"""애플리케이션 공용 컨텍스트. UI가 서비스 계층에 접근하는 유일한 창구.

production 원칙:
- 메인 UI 프로세스는 Qwen 모델을 import/load하지 않는다. qwen/STT 실제 실행은 worker 프로세스가 담당.
- ProfileService는 qwen 어댑터 없이 프로필 CRUD만 담당한다(등록은 register worker가 실행).
- 받아쓰기는 실제 FasterWhisperTranscriber. UI 스레드를 막지 않기 위해 별도 스레드에서 호출한다.
"""

from __future__ import annotations
from pathlib import Path
from .core.paths import ensure_app_dirs, profiles_dir
from .core.logging_setup import setup_logging
from .infra.settings_repository import SettingsRepository
from .infra.profile_repository import ProfileRepository
from .services.audio_service import AudioService
from .services.profile_service import ProfileService
from .services.model_manager import ModelManager
from .services.transcription_service import FasterWhisperTranscriber

class AppContext:
    """의존성 조립. 무거운 모델 의존성은 worker/서비스 내부에서 지연 import된다."""

    def __init__(self):
        self.paths = ensure_app_dirs()
        setup_logging()
        self.settings_repo = SettingsRepository()
        self.settings = self.settings_repo.load()
        self.profile_repository = ProfileRepository()
        self.audio = AudioService()  # 어댑터 미지정 시 첫 사용에 실제 ffmpeg 탐색
        self.model_manager = ModelManager()
        self.transcriber = FasterWhisperTranscriber()  # 실제 구현, cpu/int8 지연 로드
        self.profile_service = ProfileService(self.profile_repository, self.audio)

def dev_context(fake_audio=None):
    """테스트/개발용 가짜 컨텍스트(ffmpeg/qwen 미설치 환경). fake 어댑터는 여기서만 사용한다."""
    from .infra.ffmpeg_adapter import FakeFfmpegAdapter
    from .infra.qwen_adapter import FakeQwenAdapter
    from .services.transcription_service import FakeTranscriber
    ctx = AppContext.__new__(AppContext)
    ctx.paths = ensure_app_dirs()
    ctx.settings_repo = SettingsRepository()
    ctx.settings = ctx.settings_repo.load()
    adapter = fake_audio or FakeFfmpegAdapter()
    ctx.audio = AudioService(adapter)
    ctx.profile_repository = ProfileRepository()
    ctx.profile_service = ProfileService(ctx.profile_repository, ctx.audio, FakeQwenAdapter())
    ctx.model_manager = ModelManager()
    ctx.transcriber = FakeTranscriber()
    return ctx

def create_context() -> AppContext:
    """production context. fake 어댑터(FakeQwen/FakeTranscriber/FakeFfmpeg)는 절대 사용하지 않는다."""
    return AppContext()

def cleanup_stale_previews() -> int:
    """이전 실행에서 정리 실패로 남은 미리 듣기 임시 WAV를 시작 시 삭제한다."""
    from .core.paths import cleanup_preview_cache
    return cleanup_preview_cache()
