"""worker 프로세스 실행 명령 결정.

개발 실행: `python -m voice_studio.main --worker job.json`
패키지 실행(PyInstaller frozen): `VoiceStudio.exe --worker job.json`
"""

from __future__ import annotations
import sys


def apply_no_window(qprocess) -> None:
    """QProcess로 콘솔 자식 프로세스를 띄울 때 Windows 창 생성을 막는다(P17-A).

    개발 모드에서 worker는 python.exe(콘솔 앱)이므로 검은 CMD 창이 깜빡인다.
    PySide6가 setCreateProcessArgumentsModifier를 지원하는 환경에서만
    CREATE_NO_WINDOW를 적용하고, 미지원 플랫폼/API에서는 아무것도 하지 않는다.
    stdout/stderr 수집과 종료 코드 처리에는 영향이 없다.
    """
    if sys.platform != "win32":
        return
    modifier = getattr(qprocess, "setCreateProcessArgumentsModifier", None)
    if modifier is None:
        return

    import subprocess

    def _patch(args):
        args["creationflags"] = args.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW

    modifier(_patch)

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
