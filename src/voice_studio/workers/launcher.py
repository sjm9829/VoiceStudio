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


def diagnostics_command() -> tuple[str, list[str]]:
    """Self-Diagnosis 실행 명령(P12.3-06).

    개발: python -m voice_studio.main --diagnostics --json --log
    frozen: VoiceStudio.exe --diagnostics --json --log
    frozen에서 sys.executable은 VoiceStudio.exe이므로 `-m voice_studio.diagnostics`
    방식은 사용하지 않는다(worker launcher와 동일한 dispatch 패턴).
    """
    if is_frozen():
        return sys.executable, ["--diagnostics", "--json", "--log"]
    return sys.executable, ["-m", "voice_studio.main", "--diagnostics", "--json", "--log"]
