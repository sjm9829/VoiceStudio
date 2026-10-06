"""P12.3 Final Hotfix regression: PyInstaller collection timing, frozen smoke
FFmpeg/FFprobe 검사, GPU worker slot ownership."""

import ast, sys, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packaging"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
pytest.importorskip("PySide6")

ROOT = Path(__file__).resolve().parents[1]

COLLECT_PACKAGES = ("qwen_tts", "faster_whisper", "transformers",
                    "ctranslate2", "tokenizers", "safetensors")


# ---- BLOCKER 1: Analysis 생성 전 수집 완료 계약 ----

def test_spec_collects_before_analysis_ast():
    """Analysis 생성 이후에는 a.hiddenimports/a.datas/a.binaries 수정이 없어야 한다."""
    spec = (ROOT / "packaging" / "VoiceStudio.spec").read_text(encoding="utf-8")
    tree = ast.parse(spec)
    analysis_idx = None
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "a":
            call = node.value
            assert isinstance(call, ast.Call) and call.func.id == "Analysis"
            analysis_idx = i
    assert analysis_idx is not None
    for node in tree.body[analysis_idx + 1:]:
        for sub in ast.walk(node):
            if isinstance(sub, ast.AugAssign) and isinstance(sub.target, ast.Attribute):
                assert sub.target.attr not in ("hiddenimports", "datas", "binaries"), (
                    f"Analysis 이후 a.{sub.target.attr} 수정 금지")


def test_spec_pre_analysis_collect_calls_present():
    """6개 수집 대상 패키지 설정이 Analysis 이전에 존재하고 constructor에 전달된다."""
    spec = (ROOT / "packaging" / "VoiceStudio.spec").read_text(encoding="utf-8")
    analysis_pos = spec.index("a = Analysis(")
    pre = spec[:analysis_pos]
    post = spec[analysis_pos:]
    for pkg in ("qwen_tts", "faster_whisper", "transformers"):
        assert f'collect_submodules("{pkg}")' in pre, pkg
    for pkg in ("qwen_tts", "transformers", "huggingface_hub"):
        assert f'collect_data_files("{pkg}"' in pre, pkg
    assert 'for _pkg in ("ctranslate2", "tokenizers", "safetensors"):' in pre
    assert "spec_helpers.merge_collect" in pre
    assert "binaries=binaries" in post and "datas=datas" in post
    assert "hiddenimports=hiddenimports" in post
    assert "a.hiddenimports" not in post and "a.datas +=" not in post and "a.binaries +=" not in post


def test_merge_collect_returns_full_lists():
    """collect_all 계약: datas→datas, binaries→binaries, hiddenimports→hiddenimports."""
    import spec_helpers
    datas, binaries, hiddenimports = [], [], []
    spec_helpers.merge_collect(
        "x", datas, binaries, hiddenimports,
        collector=lambda name: (["d"], ["b"], ["h"]))
    assert (datas, binaries, hiddenimports) == (["d"], ["b"], ["h"])


# ---- BLOCKER 2: frozen smoke FFmpeg/FFprobe ----

def test_smoke_no_resolve_binary_instance_call():
    src = (ROOT / "src" / "voice_studio" / "main.py").read_text(encoding="utf-8")
    assert "RealFfmpegAdapter()._resolve_binary" not in src


def test_smoke_ffmpeg_ffprobe_pass_with_fake_adapter(monkeypatch, tmp_path):
    """ffmpeg/ffprobe 속성만 있으면 smoke가 FAIL로 죽지 않는다(monkeypatch로 검증)."""
    import voice_studio.main as vm
    from voice_studio.core import paths as core_paths

    class FakeAdapter:
        def __init__(self):
            self.ffmpeg = "ffmpeg.exe"
            self.ffprobe = "ffprobe.exe"
        def probe_version(self):
            return "fake version"

    import voice_studio.infra.ffmpeg_adapter as fa
    orig = fa.RealFfmpegAdapter
    fa.RealFfmpegAdapter = FakeAdapter
    try:
        # run_smoke_test를 직접 실행한다. heavy는 개발환경 SKIP, FFmpeg/FFprobe는 OK여야 한다.
        import io, contextlib
        buf = io.StringIO()
        monkeypatch.setattr(core_paths, "logs_dir", lambda: tmp_path / "logs")
        with contextlib.redirect_stdout(buf):
            rc = vm.run_smoke_test()
        out = buf.getvalue()
        assert rc == 0, out
        assert "FFmpeg: OK" in out
        assert "FFprobe: OK" in out
    finally:
        fa.RealFfmpegAdapter = orig


# ---- BLOCKER 3: GPU worker slot ownership ----

def _mkdir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


class _FakeContext:
    def __init__(self, root: Path):
        self.settings = {"mp3_output_dir": str(root / "out")}
        self.saved = []
        self.registered = []
        self.profile_service = types.SimpleNamespace(
            list_profiles=lambda: [],
            get=lambda uuid: types.SimpleNamespace(name="n", ref_text="t"),
            rename=lambda uuid, name: None,
            register=lambda **kw: self.registered.append(kw))
        self.profile_repository = types.SimpleNamespace(root=root / "profiles")
        self.model_manager = types.SimpleNamespace(model_path=lambda: "", status_text=lambda: "")
        from voice_studio.core.job_coordinator import JobCoordinator
        self.jobs = JobCoordinator()
    def save_settings(self, data):
        self.saved.append(data)


def _make_dialog(qtbot, tmp_path, context=None):
    from voice_studio.ui.voice_editor_dialog import VoiceEditorDialog
    dlg = VoiceEditorDialog(context or _FakeContext(tmp_path))
    qtbot.addWidget(dlg)
    return dlg


class _StubQProcess:
    """register 단위 테스트용 스텁. 실제 worker 프로세스를 띄우지 않는다."""
    last_instance = None
    def __init__(self, parent=None):
        type(self).last_instance = self
        self.started_args = None
        self._out = _Signal(); self._err = _Signal(); self._fin = _Signal()
    @property
    def readyReadStandardOutput(self): return self._out
    @property
    def readyReadStandardError(self): return self._err
    @property
    def finished(self): return self._fin
    def start(self, program, args): self.started_args = (program, args)
    def terminate(self): pass
    def kill(self): pass


class _Signal:
    def __init__(self): self._cbs = []
    def connect(self, cb): self._cbs.append(cb)
    def emit(self, *a): [cb(*a) for cb in list(self._cbs)]


def _stub_qprocess(monkeypatch):
    from voice_studio.ui import voice_editor_dialog as ved
    monkeypatch.setattr(ved, "QProcess", _StubQProcess)
    # 실패 경로의 경고 dialog가 테스트를 막지 않도록 한다.
    monkeypatch.setattr(ved.QMessageBox, "warning", staticmethod(lambda *a, **k: None))


def _make_window(qtbot, tmp_path):
    from voice_studio.ui.main_window import MainWindow
    win = MainWindow(_FakeContext(tmp_path))
    qtbot.addWidget(win)
    return win


def test_a_narrate_busy_register_refused(qtbot, tmp_path, monkeypatch):
    from voice_studio.ui import voice_editor_dialog as ved
    monkeypatch.setattr(ved.QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    win = _make_window(qtbot, tmp_path)
    assert win._start_worker({"job_id": "n1"}) is True   # narrate acquire
    dlg = _make_dialog(qtbot, tmp_path, context=win.context)  # 같은 coordinator 공유
    dlg.name_edit.setText("x")
    dlg._start_register_worker()                          # 거절되어야 한다
    assert dlg._job_slot_acquired is False
    assert win.context.jobs.busy is True                  # slot 유지
    assert dlg._worker is None


def test_b_narrate_busy_close_empty_dialog_keeps_slot(qtbot, tmp_path):
    win = _make_window(qtbot, tmp_path)
    assert win.context.jobs.try_acquire() is True         # narrate 점유
    dlg = _make_dialog(qtbot, tmp_path)
    dlg.close()                                           # 아무 작업 없이 닫기
    assert win.context.jobs.busy is True                  # 핵심 regression: release 금지
    win.context.jobs.release()


def test_c_register_busy_narrate_refused(qtbot, tmp_path, monkeypatch):
    from voice_studio.ui import main_window as mw
    monkeypatch.setattr(mw.QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    dlg = _make_dialog(qtbot, tmp_path)
    dlg.name_edit.setText("x")
    monkeypatch.setattr("voice_studio.core.paths.safe_job_cache_dir",
                        lambda jid: _mkdir(tmp_path / "jobs" / jid))
    monkeypatch.setattr("voice_studio.ui.voice_editor_dialog.worker_command",
                        lambda f: (sys.executable, ["--worker", f]))
    win = _make_window(qtbot, tmp_path)
    dlg = _make_dialog(qtbot, tmp_path, context=win.context)
    dlg.name_edit.setText("x")
    monkeypatch.setattr("voice_studio.core.paths.safe_job_cache_dir",
                        lambda jid: _mkdir(tmp_path / "jobs" / jid))
    monkeypatch.setattr("voice_studio.ui.voice_editor_dialog.worker_command",
                        lambda f: (sys.executable, ["--worker", f]))
    _stub_qprocess(monkeypatch)
    dlg._start_register_worker()
    assert dlg._job_slot_acquired is True
    assert win._start_worker({"job_id": "n1"}) is False   # 거절
    assert win._job_slot_acquired is False
    assert dlg.context.jobs.busy is True
    dlg._on_register_finished(1, 0)


def test_d_register_finish_releases_slot(qtbot, tmp_path, monkeypatch):
    dlg = _make_dialog(qtbot, tmp_path)
    monkeypatch.setattr("voice_studio.core.paths.safe_job_cache_dir",
                        lambda jid: _mkdir(tmp_path / "jobs" / jid))
    monkeypatch.setattr("voice_studio.ui.voice_editor_dialog.worker_command",
                        lambda f: (sys.executable, ["--worker", f]))
    _stub_qprocess(monkeypatch)
    dlg._start_register_worker()
    assert dlg.context.jobs.busy is True
    dlg._on_register_finished(0, 0)                       # 결과 event 없음 → 실패 경로
    assert dlg.context.jobs.busy is False                 # 성공/실패 모두 반납


def test_e_register_startup_exception_releases_slot(qtbot, tmp_path, monkeypatch):
    dlg = _make_dialog(qtbot, tmp_path)
    monkeypatch.setattr("voice_studio.core.paths.safe_job_cache_dir",
                        lambda jid: _mkdir(tmp_path / "jobs" / jid))
    monkeypatch.setattr("voice_studio.ui.voice_editor_dialog.worker_command",
                        lambda f: (sys.executable, ["--worker", f]))
    def boom(job_id, **kw):
        raise OSError("job file fail")
    monkeypatch.setattr("voice_studio.ui.voice_editor_dialog.build_register_payload", boom)
    _stub_qprocess(monkeypatch)
    with pytest.raises(OSError):
        dlg._start_register_worker()
    assert dlg._job_slot_acquired is False
    assert dlg.context.jobs.busy is False


def test_f_register_close_does_not_prematurely_release(qtbot, tmp_path, monkeypatch):
    dlg = _make_dialog(qtbot, tmp_path)
    monkeypatch.setattr("voice_studio.core.paths.safe_job_cache_dir",
                        lambda jid: _mkdir(tmp_path / "jobs" / jid))
    monkeypatch.setattr("voice_studio.ui.voice_editor_dialog.worker_command",
                        lambda f: (sys.executable, ["--worker", f]))
    _stub_qprocess(monkeypatch)
    dlg._start_register_worker()
    assert dlg.context.jobs.busy is True
    dlg.close()                                           # worker 종료 요청만
    assert dlg.context.jobs.busy is True                  # premature release 없음
    dlg._on_register_finished(1, 0)                       # 실제 finished 후 반납
    assert dlg.context.jobs.busy is False


def test_mainwindow_ownership_release(qtbot, tmp_path, monkeypatch):
    win = _make_window(qtbot, tmp_path)
    monkeypatch.setattr("voice_studio.ui.main_window.safe_job_cache_dir",
                        lambda jid: _mkdir(tmp_path / "jobs" / jid))
    monkeypatch.setattr("voice_studio.ui.main_window.worker_command",
                        lambda f: (sys.executable, ["--worker", f]))
    assert win._start_worker({"job_id": "n1"}) is True
    assert win._job_slot_acquired is True
    assert win.context.jobs.busy is True
    win._on_worker_finished(1, 0)
    assert win._job_slot_acquired is False
    assert win.context.jobs.busy is False
    # 자신이 acquire하지 않았으면 release하지 않는다.
    assert win.context.jobs.try_acquire() is True
    win.close()
    assert win.context.jobs.busy is True
    win.context.jobs.release()
