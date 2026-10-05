"""worker 프로세스 실행 명령 결정.

개발 실행: `python -m voice_studio.main --worker job.json`
패키지 실행(PyInstaller frozen): `VoiceStudio.exe --worker job.json`
"""

from __future__ import annotations
import sys

def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))

def worker_command(job_file: str) -> tuple[str, list[str]]:
    """(프로그램, 인자) 반환. 개발/패키지 환경 모두 같은 진입점(main.py --worker)을 쓴다."""
    if is_frozen():
        return sys.executable, ["--worker", job_file]
    return sys.executable, ["-m", "voice_studio.main", "--worker", job_file]
