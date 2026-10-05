"""모델 다운로드/상태 관리. 앱 시작 시 강제 다운로드하지 않는다."""

from __future__ import annotations
from pathlib import Path
from typing import Callable
from ..core import config
from ..core.errors import OfflineError, ModelNotDownloadedError

def default_model_cache_dir() -> Path:
    from ..core.paths import app_data_dir
    return app_data_dir() / "models"

class ModelManager:
    """Hugging Face snapshot_download 기반 모델 관리자."""

    def __init__(self, model_id: str = config.DEFAULT_MODEL_ID, cache_dir: Path | None = None):
        self.model_id = model_id
        self.cache_dir = cache_dir or default_model_cache_dir()

    def local_snapshot(self) -> Path | None:
        """이미 받아진 스냅샷 경로(완전성 표식 파일 존재 시) 반환."""
        base = self.cache_dir / self.model_id.replace("/", "__")
        marker = base / ".complete"
        if marker.is_file():
            return base
        return None

    def is_downloaded(self) -> bool:
        return self.local_snapshot() is not None

    def download(self, progress_cb: Callable[[dict], None] | None = None,
                 force: bool = False) -> Path:
        """모델을 내려받는다. 진행률은 progress_cb(dict)로 보고한다. 중단/재시도 가능."""
        snapshot = self.local_snapshot()
        if snapshot and not force:
            return snapshot
        try:
            from huggingface_hub import snapshot_download  # 지연 import: UI는 의존하지 않음
        except ImportError as exc:
            raise ModelNotDownloadedError("모델 다운로드 라이브러리를 찾을 수 없습니다.") from exc
        target = self.cache_dir / self.model_id.replace("/", "__")
        target.mkdir(parents=True, exist_ok=True)
        try:
            path = snapshot_download(
                repo_id=self.model_id, local_dir=target,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.md"])
        except Exception as exc:  # 네트워크 오류 → 오프라인 안내
            if not self.is_downloaded():
                raise OfflineError(f"모델 다운로드 실패: {exc}") from exc
            return target
        (target / ".complete").write_text(path)
        return target

    def status_text(self) -> str:
        if self.is_downloaded():
            return f"받아짐 ({self.model_id})"
        return f"아직 받지 않음 ({self.model_id})"
