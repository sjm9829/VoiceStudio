"""P12.3-25: 앱 수준 동시 worker 1개 제한 regression."""

import sys, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
pytest.importorskip("PySide6")

from voice_studio.core.job_coordinator import JobCoordinator


def test_coordinator_allows_only_one_worker():
    c = JobCoordinator()
    assert c.try_acquire() is True
    assert c.busy is True
    assert c.try_acquire() is False      # 두 번째 worker 금지
    c.release()
    assert c.busy is False
    assert c.try_acquire() is True       # 반납 후 재사용 가능


def test_context_exposes_job_coordinator():
    from voice_studio.app_context import dev_context
    ctx = dev_context()
    assert hasattr(ctx, "jobs")
    assert ctx.jobs.try_acquire() is True
    ctx.jobs.release()


def _mkdir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


class _FakeContext:
    def __init__(self, root: Path):
        self.settings = {"mp3_output_dir": str(root / "out")}
        self.saved = []
        self.profile_service = types.SimpleNamespace(list_profiles=lambda: [])
        self.profile_repository = types.SimpleNamespace(root=root / "profiles")
        self.model_manager = types.SimpleNamespace(model_path=lambda: "", status_text=lambda: "")
        from voice_studio.core.job_coordinator import JobCoordinator
        self.jobs = JobCoordinator()
    def save_settings(self, data):
        self.saved.append(data)


def _make_window(qtbot, tmp_path):
    from voice_studio.ui.main_window import MainWindow
    win = MainWindow(_FakeContext(tmp_path))
    qtbot.addWidget(win)
    return win


def test_narrate_start_refused_while_slot_held(qtbot, tmp_path, monkeypatch):
    win = _make_window(qtbot, tmp_path)
    assert win.context.jobs.try_acquire() is True   # register worker가 점유한 상황 흉내
    monkeypatch.setattr("voice_studio.ui.main_window.QMessageBox.warning",
                        staticmethod(lambda *a, **k: None))
    assert win._start_worker({"job_id": "n1"}) is False
    assert win.context.jobs.busy is True            # 점유 유지(반납 안 됨)
    assert win._job_dir is None                     # worker 시작 안 됨
    win.context.jobs.release()


def test_narrate_finish_releases_slot(qtbot, tmp_path, monkeypatch):
    win = _make_window(qtbot, tmp_path)
    monkeypatch.setattr("voice_studio.ui.main_window.safe_job_cache_dir",
                        lambda jid: _mkdir(tmp_path / "jobs" / jid))
    monkeypatch.setattr("voice_studio.ui.main_window.worker_command",
                        lambda f: (sys.executable, ["--worker", f]))
    assert win._start_worker({"job_id": "n1"}) is True
    assert win.context.jobs.busy is True
    win._on_worker_finished(1, 0)                   # 실패 종료
    assert win.context.jobs.busy is False
