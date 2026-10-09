"""Windows 콘솔 창 깜빡임 제거를 위한 공통 subprocess 실행 헬퍼(P17-A).

Windows에서 ffmpeg/ffprobe/nvidia-smi 같은 콘솔 실행 파일을 subprocess로
호출하면 검은 CMD 창이 순간 나타났다 사라진다. CREATE_NO_WINDOW 플래그로
새 창 생성을 막는다. 그 외 동작(stdout/stderr 캡처, return code, timeout,
encoding, stdin 입력)은 subprocess.run과 동일하게 유지한다.
"""

from __future__ import annotations
import subprocess
import sys


def no_window_creationflags() -> int:
    """Windows에서 CREATE_NO_WINDOW, 그 외 플랫폼에서 0을 반환한다."""
    if sys.platform == "win32":
        return subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    return 0


def run(*args, **kwargs) -> subprocess.CompletedProcess:
    """subprocess.run 래퍼. Windows 콘솔 창 생성을 막는다(P17-A).

    호출자가 creationflags를 직접 넘기면 OR로 병합한다(기존 플래그 보존).
    그 외 모든 인자는 subprocess.run에 그대로 전달된다.
    """
    kwargs["creationflags"] = kwargs.get("creationflags", 0) | no_window_creationflags()
    return subprocess.run(*args, **kwargs)
