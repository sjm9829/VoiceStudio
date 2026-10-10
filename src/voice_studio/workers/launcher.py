"""worker 프로세스 실행 명령 결정.

개발 실행: `python -m voice_studio.main --worker job.json`
패키지 실행(PyInstaller frozen): `VoiceStudio.exe --worker job.json`
"""

from __future__ import annotations
import sys
from pathlib import Path


def apply_no_window(qprocess) -> bool:
    """QProcess 자식의 Windows 콘솔 창 생성을 막는다(P17-A, P17-H3).

    PySide6 setCreateProcessArgumentsModifier는 6.11 바인딩에 존재하지 않아
    실제로 적용 가능한 Windows 대체는 다음 둘이다.
    1. 개발 모드: 콘솔 없는 pythonw.exe로 worker 실행(파이프 stdout/stderr 유지).
    2. frozen: 진입점 VoiceStudio.exe가 console=False(windowed)라 창이 없다.

    반환값은 적용 여부이며, 미지원 환경에서 조용히 no-op이 되지 않도록
    worker_command 단계에서 pythonw 치환으로 해결한다. modifier가 지원되는
    바인딩에서는 CREATE_NO_WINDOW를 추가로 적용한다.
    """
    if sys.platform != "win32":
        return False
    modifier = getattr(qprocess, "setCreateProcessArgumentsModifier", None)
    if modifier is None:
        return False

    import subprocess

    def _patch(args):
        args["creationflags"] = args.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW

    modifier(_patch)
    return True


def _console_less_python() -> str | None:
    """개발 모드 콘솔 없는 인터프리터(pythonw.exe). 없으면 None."""
    if sys.platform != "win32":
        return None
    exe = Path(sys.executable)
    w = exe.with_name("pythonw.exe")
    return str(w) if w.is_file() else None


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def worker_command(job_file: str) -> tuple[str, list[str]]:
    """(프로그램, 인자) 반환. 개발/패키지 모두 같은 진입점(main.py --worker).

    개발 모드 Windows에서는 pythonw.exe를 우선해 콘솔 창 생성을 원천 차단한다(P17-H3).
    pythonw가 없으면 기존과 같이 sys.executable을 사용한다.
    """
    if is_frozen():
        return sys.executable, ["--worker", job_file]
    pythonw = _console_less_python()
    program = pythonw or sys.executable
    return program, ["-m", "voice_studio.main", "--worker", job_file]


def diagnostics_command() -> tuple[str, list[str]]:
    """Self-Diagnosis 실행 명령(P12.3-06). 콘솔 억제 규칙은 worker_command과 동일.

    개발: python(w) -m voice_studio.main --diagnostics --json --log
    frozen: VoiceStudio.exe --diagnostics --json --log
    """
    if is_frozen():
        return sys.executable, ["--diagnostics", "--json", "--log"]
    pythonw = _console_less_python()
    program = pythonw or sys.executable
    return program, ["-m", "voice_studio.main", "--diagnostics", "--json", "--log"]
