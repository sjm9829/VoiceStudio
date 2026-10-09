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
