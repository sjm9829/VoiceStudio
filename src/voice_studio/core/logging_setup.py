"""파일 로깅 설정. UI에는 로그를 노출하지 않는다."""

from __future__ import annotations
import logging, sys
from pathlib import Path
from .paths import logs_dir

def setup_logging(level: int = logging.INFO) -> Path:
    from .paths import ensure_app_dirs
    ensure_app_dirs()
    log_file = logs_dir() / "voice-studio.log"
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stderr)],
        force=True,
    )
    return log_file
