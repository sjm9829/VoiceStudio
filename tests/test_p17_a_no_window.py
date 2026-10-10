"""P17-A 회귀 테스트: Windows 콘솔 창 깜빡임 제거(CREATE_NO_WINDOW 공통 헬퍼).

CREATE_NO_WINDOW는 win32 전용 상수이므로, 비 Windows 개발 환경에서도
테스트할 수 있게 mock.patch로 상수를 주입해 win32 경로를 검증한다.
Windows 실제 플래그 값은 0x08000000이다.
"""

from __future__ import annotations
import subprocess
import sys
from unittest import mock

import pytest

from voice_studio.infra import subprocess_runner
from voice_studio.infra.ffmpeg_adapter import RealFfmpegAdapter
from voice_studio.workers.launcher import apply_no_window

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_CONSOLE = 0x00000010


def test_no_window_creationflags_on_win32():
    with mock.patch.object(subprocess_runner.sys, "platform", "win32"), \
            mock.patch.object(subprocess, "CREATE_NO_WINDOW", CREATE_NO_WINDOW, create=True):
        assert subprocess_runner.no_window_creationflags() == CREATE_NO_WINDOW


def test_no_window_creationflags_on_other_platforms():
    for platform in ("linux", "darwin"):
        with mock.patch.object(subprocess_runner.sys, "platform", platform):
            assert subprocess_runner.no_window_creationflags() == 0


def test_run_merges_caller_creationflags():
    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(args, 0)

    with mock.patch.object(subprocess_runner.sys, "platform", "win32"), \
            mock.patch.object(subprocess, "CREATE_NO_WINDOW", CREATE_NO_WINDOW, create=True), \
            mock.patch.object(subprocess_runner.subprocess, "run", fake_run):
        subprocess_runner.run(["x"], capture_output=True, creationflags=CREATE_NEW_CONSOLE)
    assert captured["creationflags"] == CREATE_NEW_CONSOLE | CREATE_NO_WINDOW


def test_run_preserves_other_kwargs():
    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(args, 0)

    with mock.patch.object(subprocess_runner.subprocess, "run", fake_run):
        subprocess_runner.run(["x"], capture_output=True, text=True, timeout=300, input="hi")
    assert captured["capture_output"] is True
    assert captured["text"] is True
    assert captured["timeout"] == 300
    assert captured["input"] == "hi"


def test_real_ffmpeg_adapter_uses_no_window_helper():
    """ffmpeg 어댑터의 모든 subprocess 호출이 공통 헬퍼를 거쳐 일관된 creationflags를 받는다."""
    adapter = RealFfmpegAdapter.__new__(RealFfmpegAdapter)
    adapter.ffmpeg = "ffmpeg"
    adapter.ffprobe = "ffprobe"

    recorded = []

    def fake_run(*args, **kwargs):
        recorded.append(kwargs.get("creationflags"))
        return subprocess.CompletedProcess(args[0] if args else kwargs.get("args"), 1, b"", b"")

    with mock.patch.object(subprocess_runner.subprocess, "run", fake_run):
        with pytest.raises(Exception):
            adapter.probe("missing.mp3")

    expected = subprocess_runner.no_window_creationflags()
    assert recorded and all(flag == expected for flag in recorded)


class _FakeQProcess:
    def __init__(self):
        self.modifier = None

    def setCreateProcessArgumentsModifier(self, modifier):
        self.modifier = modifier


def test_apply_no_window_sets_modifier_on_win32():
    qp = _FakeQProcess()
    with mock.patch.object(sys, "platform", "win32"), \
            mock.patch.object(subprocess, "CREATE_NO_WINDOW", CREATE_NO_WINDOW, create=True):
        apply_no_window(qp)
        assert qp.modifier is not None
        args = {}
        qp.modifier(args)
        assert args["creationflags"] == CREATE_NO_WINDOW


def test_apply_no_window_noop_without_pyside6_support():
    class _NoSupport:
        pass

    qp = _NoSupport()  # setCreateProcessArgumentsModifier 없음
    apply_no_window(qp)  # 예외 없이 통과해야 한다



# ---- P17-H3: QProcess 미지원 바인딩 대비 콘솔 없는 worker 실행 ----

def test_worker_command_uses_pythonw_on_dev_windows(monkeypatch, tmp_path):
    """개발 모드 Windows에서 pythonw.exe가 있으면 worker/diagnostics 진입점이 된다."""
    from voice_studio.workers import launcher
    fake_py = tmp_path / "python.exe"
    fake_pyw = tmp_path / "pythonw.exe"
    fake_py.write_text("")
    fake_pyw.write_text("")
    monkeypatch.setattr(launcher.sys, "platform", "win32")
    monkeypatch.setattr(launcher.sys, "executable", str(fake_py))
    monkeypatch.setattr(launcher, "is_frozen", lambda: False)
    program, args = launcher.worker_command("j.json")
    assert program == str(fake_pyw)
    assert args == ["-m", "voice_studio.main", "--worker", "j.json"]
    dprogram, dargs = launcher.diagnostics_command()
    assert dprogram == str(fake_pyw)
    assert "--diagnostics" in dargs


def test_worker_command_falls_back_to_python_without_pythonw(monkeypatch, tmp_path):
    """pythonw가 없는 환경에서는 기존과 같이 sys.executable을 쓴다(회귀 방지)."""
    from voice_studio.workers import launcher
    fake_py = tmp_path / "python.exe"
    fake_py.write_text("")
    monkeypatch.setattr(launcher.sys, "platform", "win32")
    monkeypatch.setattr(launcher.sys, "executable", str(fake_py))
    monkeypatch.setattr(launcher, "is_frozen", lambda: False)
    program, args = launcher.worker_command("j.json")
    assert program == str(fake_py)


def test_worker_command_frozen_uses_sys_executable(monkeypatch):
    """frozen에서는 VoiceStudio.exe 그대로(콘솔 없는 windowed 빌드)."""
    from voice_studio.workers import launcher
    monkeypatch.setattr(launcher, "is_frozen", lambda: True)
    monkeypatch.setattr(launcher.sys, "executable", "C:/app/VoiceStudio.exe")
    program, args = launcher.worker_command("j.json")
    assert program == "C:/app/VoiceStudio.exe"
    assert args == ["--worker", "j.json"]


def test_apply_no_window_returns_supported_state():
    """지원 바인딩에선 True, 미지원에선 False를 반환해 조용한 실패를 없앤다(P17-H3)."""
    qp = _FakeQProcess()
    with mock.patch.object(sys, "platform", "win32"), \
            mock.patch.object(subprocess, "CREATE_NO_WINDOW", CREATE_NO_WINDOW, create=True):
        assert apply_no_window(qp) is True
    class _NoSupportLocal:
        pass
    assert apply_no_window(_NoSupportLocal()) is False
