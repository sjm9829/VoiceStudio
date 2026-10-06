"""모델 다운로드/상태 관리. 앱 시작 시 강제 다운로드하지 않는다."""

from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Callable
from ..core import config
from ..core.errors import OfflineError, ModelNotDownloadedError

_logger = logging.getLogger(__name__)

# 공식 모델 repo 파일 목록(HF api 기준) 중 실행에 반드시 필요한 파일.
# allow_patterns로 일부만 받으면 tokenizer/config 로드가 실패하므로 전체 파일을 받는다.
REQUIRED_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors",
    "speech_tokenizer/config.json",
    "speech_tokenizer/configuration.json",
    "speech_tokenizer/model.safetensors",
    "speech_tokenizer/preprocessor_config.json",
)

def default_model_cache_dir() -> Path:
    from ..core.paths import app_data_dir
    return app_data_dir() / "models"

class ModelManager:
    """Hugging Face snapshot_download 기반 모델 관리자.

    완료 판정은 .complete 마커만이 아니라 실행에 필요한 핵심 파일 존재 여부까지 검증한다.
    """

    def __init__(self, model_id: str = config.DEFAULT_MODEL_ID, cache_dir: Path | None = None):
        self.model_id = model_id
        self.cache_dir = cache_dir or default_model_cache_dir()

    def local_snapshot(self) -> Path | None:
        """이미 받아진 유효한 로컬 스냅샷 경로를 반환한다. 핵심 파일이 빠졌으면 None."""
        base = self.cache_dir / self.model_id.replace("/", "__")
        if not (base / ".complete").is_file():
            return None
        missing = self.missing_required_files(base)
        return None if missing else base

    def missing_required_files(self, base: Path) -> list[str]:
        return [name for name in REQUIRED_MODEL_FILES if not (base / name).is_file()]

    def is_downloaded(self) -> bool:
        return self.local_snapshot() is not None

    def download(self, progress_cb: Callable[[dict], None] | None = None,
                 force: bool = False) -> Path:
        """모델을 내려받는다(전체 파일). 완료 후 핵심 파일 존재를 검증한다."""
        snapshot = self.local_snapshot()
        if snapshot and not force:
            return snapshot
        try:
            from huggingface_hub import snapshot_download  # 지연 import: UI는 의존하지 않음
        except ImportError as exc:
            raise ModelNotDownloadedError("모델 다운로드 라이브러리를 찾을 수 없습니다.") from exc
        self._disable_hub_console_progress()
        target = self.cache_dir / self.model_id.replace("/", "__")
        target.mkdir(parents=True, exist_ok=True)
        try:
            path = snapshot_download(repo_id=self.model_id, local_dir=target)
        except Exception as exc:  # 네트워크 오류 → 오프라인 안내
            # 기술 상세는 로그로 남기고 사용자에게는 짧은 안내만 노출한다(P13 hotfix).
            _logger.warning("모델 다운로드 실패(%s): %s", self.model_id, exc, exc_info=True)
            if not self.is_downloaded():
                raise OfflineError(
                    f"모델 다운로드 실패: {exc}",
                    user_message="음성 모델을 받지 못했습니다. 인터넷 연결을 확인한 뒤 다시 시도해 주세요.",
                ) from exc
            return target
        missing = self.missing_required_files(target)
        if missing:
            raise OfflineError(f"모델 파일이 불완전합니다: {', '.join(missing)}")
        (target / ".complete").write_text(str(path))
        return target

    @staticmethod
    def _disable_hub_console_progress() -> None:
        """windowed frozen app(sys.stdout=None)에서 HF console progress가 crash하지 않게 한다(P13).

        PyInstaller console=False 환경에서 sys.stdout/sys.stderr가 None인데
        Hugging Face Hub의 tqdm progress bar가 이에 write하면
        'NoneType' object has no attribute 'write'로 실패한다.
        전역 stdout/stderr 교체·환경변수 요구 없이 공식 API로 progress를 끈다.
        UI에는 이미 indeterminate progress bar가 있으므로 콘솔 progress는 필요 없다.
        """
        try:
            from huggingface_hub.utils import disable_progress_bars
        except ImportError:
            # 구버전 hub 대체 수단: 공식 지원 환경변수를 프로세스 내에서만 설정한다.
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            return
        disable_progress_bars()

    def status_text(self) -> str:
        if self.is_downloaded():
            return f"받아짐 ({self.model_id})"
        return f"아직 받지 않음 ({self.model_id})"

    def model_path(self) -> str:
        """runtime job에 전달할 로컬 모델 경로. 없으면 사용자용 오류를 던진다."""
        snapshot = self.local_snapshot()
        if snapshot is None:
            raise ModelNotDownloadedError(
                "음성 모델이 아직 받아지지 않았습니다. 설정에서 모델을 받은 뒤 다시 시도해 주세요.")
        return str(snapshot)
