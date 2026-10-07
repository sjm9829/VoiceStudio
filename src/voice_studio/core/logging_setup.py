"""파일 로깅 설정. UI에는 로그를 노출하지 않는다."""

from __future__ import annotations
import logging, sys
from pathlib import Path
from .paths import logs_dir

def setup_logging(level: int = logging.INFO) -> Path:
    r"""파일 로깅을 항상 구성하고, 콘솔 handler는 stderr가 있을 때만 추가한다(P13).

    PyInstaller console=False(windowed) 설치본에서 sys.stdout/sys.stderr가
    None일 수 있어 StreamHandler(None) 생성/출력이 실패한다. 파일 로그
    (%LOCALAPPDATA%\VoiceStudio\logs\voice-studio.log)는 GUI/frozen에서도
    반드시 동작해야 한다. force=True가 중복 handler 누적을 방지한다.
    """
    from .paths import ensure_app_dirs
    ensure_app_dirs()
    log_file = logs_dir() / "voice-studio.log"
    handlers: list[logging.Handler] = [
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )
    return log_file

