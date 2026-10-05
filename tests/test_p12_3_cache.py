"""P12.3-18/19: 결과 캐시 lifecycle, stale jobs startup cleanup regression."""

import sys, time, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
pytest.importorskip("PySide6")

from voice_studio.core import paths as core_paths


def test_cleanup_stale_jobs_removes_only_old_job_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(core_paths, "jobs_cache_dir", lambda: tmp_path / "jobs")
    jobs = tmp_path / "jobs"; jobs.mkdir()
    old = jobs / "old-uuid"; old.mkdir(); (old / "result.mp3").write_bytes(b"x")
    fresh = jobs / "fresh-uuid"; fresh.mkdir()
    os_time = time.time() - 25 * 3600
    import os as _os
    _os.utime(old, (os_time, os_time))
    removed = core_paths.cleanup_stale_jobs(max_age_hours=24)
    assert removed == 1
    assert not old.exists() and fresh.exists()


def test_cleanup_stale_jobs_never_touches_profiles_or_settings(tmp_path, monkeypatch):
    """프로필/모델/설정 폴더는 절대 건드리지 않는다(P12.3-19)."""
    monkeypatch.setattr(core_paths, "jobs_cache_dir", lambda: tmp_path / "cache" / "jobs")
    profiles = tmp_path / "profiles"; profiles.mkdir(); (profiles / "p1").mkdir()
    settings = tmp_path / "settings.json"; settings.write_text("{}")
    removed = core_paths.cleanup_stale_jobs()
    assert removed == 0
    assert profiles.exists() and settings.exists()


class _FakeContext:
    def __init__(self, root: Path):
        self.settings = {"mp3_output_dir": str(root / "out")}
        self.saved = []
        self.profile_service = types.SimpleNamespace(list_profiles=lambda: [])
        self.profile_repository = types.SimpleNamespace(root=root / "profiles")
        self.model_manager = types.SimpleNamespace(model_path=lambda: "", status_text=lambda: "")
    def save_settings(self, data):
        self.saved.append(data)


def _mkdir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_window(qtbot, tmp_path):
    from voice_studio.ui.main_window import MainWindow
    win = MainWindow(_FakeContext(tmp_path))
    qtbot.addWidget(win)
    return win


def test_success_keeps_job_dir_until_saved_or_next_generation(qtbot, tmp_path):
    win = _make_window(qtbot, tmp_path)
    job_dir = tmp_path / "jobs" / "j1"; job_dir.mkdir(parents=True)
    (job_dir / "result.mp3").write_bytes(b"x")
    win._job_dir = job_dir
    win._result_received = True
    win._last_output = str(job_dir / "result.mp3")
    win._on_worker_finished(0, 0)
    assert win._job_dir == job_dir           # 성공: 캐시 보존
    assert (job_dir / "result.mp3").exists()
    assert win.save_btn.isEnabled()


def test_new_generation_deletes_previous_success_cache(qtbot, tmp_path, monkeypatch):
    win = _make_window(qtbot, tmp_path)
    old_dir = tmp_path / "jobs" / "old"; old_dir.mkdir(parents=True)
    (old_dir / "result.mp3").write_bytes(b"x")
    win._job_dir = old_dir
    win._last_output = str(old_dir / "result.mp3")
    payload = {"job_id": "new"}
    import voice_studio.core.paths as cp
    monkeypatch.setattr("voice_studio.ui.main_window.safe_job_cache_dir",
                        lambda jid: (tmp_path / "jobs" / jid).__class_mkdir__ if False else _mkdir(tmp_path / "jobs" / jid))
    monkeypatch.setattr("voice_studio.ui.main_window.worker_command",
                        lambda f: (sys.executable, ["--worker", f]))
    win._start_worker(payload)
    assert not old_dir.exists()              # P12.3-18: 새 생성 시작 시 이전 캐시 삭제
    assert win._last_output is None
    win._worker.kill()


def test_save_mp3_moves_pointer_and_deletes_cache(qtbot, tmp_path, monkeypatch):
    win = _make_window(qtbot, tmp_path)
    job_dir = tmp_path / "jobs" / "j1"; job_dir.mkdir(parents=True)
    src = job_dir / "result.mp3"; src.write_bytes(b"data")
    target = tmp_path / "out" / "saved.mp3"; target.parent.mkdir(parents=True)
    win._job_dir = job_dir
    win._last_output = str(src)
    monkeypatch.setattr("voice_studio.ui.main_window.QFileDialog.getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    monkeypatch.setattr("voice_studio.ui.main_window.QMessageBox.information",
                        staticmethod(lambda *a, **k: None))
    win.save_mp3()
    assert target.read_bytes() == b"data"
    assert win._last_output == str(target)   # 사용자 파일로 교체
    assert not job_dir.exists()              # 캐시 job 폴더 삭제


def test_app_close_deletes_unsaved_result_cache(qtbot, tmp_path):
    win = _make_window(qtbot, tmp_path)
    job_dir = tmp_path / "jobs" / "j1"; job_dir.mkdir(parents=True)
    (job_dir / "result.mp3").write_bytes(b"x")
    win._job_dir = job_dir
    win._last_output = str(job_dir / "result.mp3")
    win.close()
    assert not job_dir.exists()              # 저장 없이 종료 → 캐시 삭제


def test_main_startup_calls_stale_job_cleanup():
    src = (Path(__file__).resolve().parents[1] / "src" / "voice_studio" / "main.py").read_text(encoding="utf-8")
    assert "cleanup_stale_jobs()" in src
