"""P18-G 안정화 회귀: Windows 안전 PID 확인, OS-native 파일 잠금, MP3 저장 원자성."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from unittest import mock

import pytest

from voice_studio.services import gguf_model_manager as gmm
from voice_studio.core import gguf as gguf_cfg


def test_pid_alive_never_uses_kill_on_windows(tmp_path, monkeypatch):
    """Windows 분기는 os.kill을 호출하지 않는다(P18-G: signal 0 위험 제거)."""
    calls = []

    def fake_kill(pid, sig):
        calls.append((pid, sig))

    monkeypatch.setattr(gmm.os, "kill", fake_kill)
    monkeypatch.setattr(gmm.sys, "platform", "win32")
    fake_k32 = mock.Mock()
    fake_k32.OpenProcess.return_value = 7
    import ctypes
    with mock.patch.object(ctypes, "windll", create=True) as _w:
        ctypes.windll = mock.Mock()
        ctypes.windll.kernel32 = fake_k32
        assert gmm._pid_alive(4321) is True
    assert calls == []
    fake_k32.OpenProcess.assert_called_once()
    fake_k32.CloseHandle.assert_called_once()


def test_pid_alive_posix_check():
    """POSIX 분기: 내 PID True, 불가 PID False."""
    assert gmm._pid_alive(os.getpid()) is True
    assert gmm._pid_alive(-5) is False


def test_file_lock_mutual_exclusion_without_kill(tmp_path, monkeypatch):
    """직렬 획득 상호배타. 경합 판정에 os.kill 부재(P18-G)."""
    lock_path = tmp_path / "x.download.lock"
    monkeypatch.setattr(gmm.os, "kill", lambda p, s: (_ for _ in ()).throw(AssertionError("no kill")))
    order = []

    def worker():
        with gmm._FileLock(lock_path):
            order.extend(("in1",))
            time.sleep(0.05)
            order.extend(("out1",))

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(0.02)
    with gmm._FileLock(lock_path):
        order.extend(("in2",))
    t.join()
    assert order.index("out1") < order.index("in2")


def test_file_lock_timeout_raises_no_fallback(tmp_path):
    """점유 중 잠금은 timeout 후 TimeoutError. 무잠금 폴백 금지."""
    lock_path = tmp_path / "y.download.lock"
    with gmm._FileLock(lock_path):
        contender = gmm._FileLock(lock_path)
        contender.TIMEOUT_S = 0.3
        with pytest.raises(TimeoutError):
            contender.__enter__()
    contender.__enter__()
    contender.__exit__()


def test_download_lock_is_stable_sibling(tmp_path, monkeypatch):
    """download 잠금은 rename 대상이 아닌 형제 경로에 둔다."""
    m = gmm.GgufModelManager(cache_dir=tmp_path / "model")
    seen = []

    def fake_inner(self, tmp, progress_cb=None):
        return self.cache_dir

    orig_init = gmm._FileLock.__init__

    def spy_init(self, path):
        orig_init(self, path)
        seen.append(Path(path))

    monkeypatch.setattr(gmm.GgufModelManager, "_download_inner", fake_inner)
    monkeypatch.setattr(gmm._FileLock, "__init__", spy_init)
    monkeypatch.setattr(gmm, "_verify_pair", lambda d: [])
    m.download()
    locks = [p for p in seen if p.name.endswith(".download.lock")]
    assert locks
    assert all("model.tmp" != p.name for p in locks)
    assert all(p.parent == tmp_path for p in locks)  # cache_dir와 형제


def test_download_reuses_completed_by_racing_process(tmp_path, monkeypatch):
    """잠금 획득 후 완성 캐시 발견 시 재검증 후 재사용(선행 다운로드 결과)."""
    d = tmp_path / "model"
    d.mkdir(parents=True)
    (d / ".complete").write_text("rev")
    (d / gguf_cfg.GGUF_MAIN_FILE).write_bytes(b"x" * 16)
    (d / gguf_cfg.GGUF_MMPROJ_FILE).write_bytes(b"y")
    # 해시 일치하도록 실제 digest로 마커 파일 대체는 복잡: verify만 mock
    m = gmm.GgufModelManager(cache_dir=d)
    inner_calls = []
    monkeypatch.setattr(gmm.GgufModelManager, "_download_inner",
                        lambda self, tmp, progress_cb=None: inner_calls.append(1) or d)
    monkeypatch.setattr(gmm, "_verify_pair", lambda dir_: [])
    assert m.download() == d
    assert inner_calls == []  # inner 미실행: 잠금 후 재검증 재사용


def test_save_mp3_atomic_publish_and_no_overwrite(qtbot, stub_context, tmp_path, monkeypatch):
    """copyfile 실패 시 임시/예약 파일만 정리되고 기존 MP3는 보존된다(P18-G)."""
    from voice_studio.ui import main_window as mw
    from voice_studio.ui.main_window import MainWindow
    # This test exercises saving only: no model preparation or modal event loop.
    monkeypatch.setattr(MainWindow, "_model_ready", lambda self: True)
    warned = {}
    informed = {}
    monkeypatch.setattr(mw.QMessageBox, "warning", lambda *a: warned.setdefault('w', a))
    monkeypatch.setattr(mw.QMessageBox, "information", lambda *a: informed.setdefault('i', a))
    win = MainWindow(stub_context)
    qtbot.addWidget(win)

    # 저장 대상 dir 준비: 기존 사용자 MP3 존재
    out_dir = tmp_path / "music"; out_dir.mkdir(parents=True)
    existing = out_dir / "홍길동_20260101_000000.mp3"
    existing.write_bytes(b"EXISTING")
    monkeypatch.setattr(win, "_save_dir", lambda: out_dir)
    win._gen_voice_name = "홍길동"
    import datetime as _dt
    win._gen_finished_at = _dt.datetime(2026, 1, 2, 3, 4, 5)

    src = tmp_path / "job" / "result.mp3"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"MP3DATA")
    win._last_output = src
    win._cleanup_job = lambda: None
    win._player_ready = lambda: False

    # 예약 경로: 덮어쓰기 없는 신규 예약만 허용
    path = win._auto_save_path()
    assert path is not None and Path(path).name == "홍길동_20260102_030405.mp3"
    assert Path(path).exists()  # 예약된 빈 파일
    assert existing.read_bytes() == b"EXISTING"

    # 실패 경로: 사전 예약을 제거해 save_mp3가 예약~실패~정리를 온전히 수행하게 함
    Path(path).unlink(missing_ok=True)
    import shutil as _shutil
    def boom(a, b):
        raise OSError("disk full")
    with monkeypatch.context() as failure_patch:
        failure_patch.setattr(mw.shutil, "copyfile", boom)
        win.save_mp3()
    assert 'w' in warned
    # 부분/빈 MP3 잔존 없음(예약 파일도 정리), 기존파일 보존
    leftovers = [p for p in out_dir.iterdir() if p.name != existing.name]
    assert leftovers == [], f"leftover: {leftovers}"
    assert existing.read_bytes() == b"EXISTING"

    # 성공 경로: 원자 게시 후 완성 MP3 존재, 임시파일 없음
    win.save_mp3()
    assert 'i' in informed
    saved = out_dir / "홍길동_20260102_030405.mp3"
    assert saved.read_bytes() == b"MP3DATA"
    assert not (out_dir / "홍길동_20260102_030405.mp3.part~").exists()


@pytest.fixture
def stub_context(tmp_path):
    import types
    from voice_studio.core.job_coordinator import JobCoordinator
    ctx = types.SimpleNamespace()
    ctx.settings = {"mp3_output_dir": str(tmp_path / "out"), "mp3_bitrate_kbps": 192}
    ctx.profile_repository = types.SimpleNamespace(root=tmp_path / "profiles")
    ctx.model_manager = types.SimpleNamespace(model_path=lambda: str(tmp_path / "models" / "snap"))
    ctx.profile_service = types.SimpleNamespace(list_profiles=lambda: [])
    ctx.audio = None
    ctx.transcriber = None
    ctx.jobs = JobCoordinator()
    return ctx
