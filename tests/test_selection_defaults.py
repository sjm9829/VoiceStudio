"""P12.1-04/06/18: 참조 선택 구간 regression + 등록 입력 검증(GPU 불필요, offscreen Qt)."""

import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

pytest.importorskip("PySide6", reason="PySide6 미설치 환경")


@pytest.fixture
def stub_context(tmp_path):
    ctx = types.SimpleNamespace()
    ctx.settings = {}
    ctx.profile_repository = types.SimpleNamespace(root=tmp_path / "profiles")
    ctx.model_manager = types.SimpleNamespace(model_path=lambda: str(tmp_path / "models" / "snap"))
    ctx.profile_service = types.SimpleNamespace(list_profiles=lambda: [])
    ctx.audio = None
    ctx.transcriber = None
    from voice_studio.core.job_coordinator import JobCoordinator
    ctx.jobs = JobCoordinator()
    return ctx


@pytest.fixture
def editor(qtbot, stub_context):
    from voice_studio.ui.voice_editor_dialog import VoiceEditorDialog
    dlg = VoiceEditorDialog(stub_context)
    qtbot.addWidget(dlg)
    return dlg


def test_long_file_default_selection_is_target_seconds(editor):
    """10분(600초) 파일 로드 시 실제 widget state가 0~15초여야 한다(라벨만이 아님)."""
    from voice_studio.core import config
    editor._on_peaks([0.1] * 100, 600.0)
    assert editor.duration == 600.0
    assert editor.wave.start_s == 0.0
    assert editor.wave.end_s == config.REFERENCE_TARGET_SECONDS
    assert editor.wave.end_s == 15.0


def test_short_file_selection_capped_at_duration(editor):
    editor._on_peaks([0.1] * 10, 8.0)
    assert editor.wave.start_s == 0.0
    assert editor.wave.end_s == 8.0


def test_selection_warns_at_30s(editor, monkeypatch):
    from voice_studio.ui import voice_editor_dialog as ved
    shown = []
    monkeypatch.setattr(ved.QMessageBox, "information",
                        lambda *a, **k: shown.append(a[2] if len(a) > 2 else ""))
    editor._on_selection(0.0, 31.0)
    assert shown and "30초" in shown[0]
    shown.clear()
    editor._on_selection(0.0, 40.0)  # 1회만 경고
    assert not shown
    editor._on_selection(0.0, 10.0)
    assert not shown


def _fake_worker(monkeypatch, ved, captured):
    from PySide6.QtCore import QObject, Signal

    class _Signal:
        def connect(self, fn): pass

    class FakeProcess(QObject):
        def __init__(self, parent=None):
            super().__init__()
            self.readyReadStandardOutput = _Signal()
            self.readyReadStandardError = _Signal()
            self.finished = _Signal()

        def start(self, program, args):
            captured["program"], captured["args"] = program, args
            captured["job_file"] = args[-1]

    monkeypatch.setattr(ved.QProcess, "__init__", FakeProcess.__init__)
    monkeypatch.setattr(ved.QProcess, "start", FakeProcess.start)
    # 실제 QProcess 인스턴스 대체가 어려우므로 QProcess 자체를 교체한다.
    monkeypatch.setattr(ved, "QProcess", FakeProcess)
    monkeypatch.setattr(ved, "worker_command", lambda job_file: ("prog", ["--worker", job_file]))


def _ffmpeg_stub(monkeypatch):
    import voice_studio.infra.ffmpeg_adapter as fa

    class FakeReal:
        def __init__(self, ffmpeg=None, ffprobe=None):
            self.ffmpeg, self.ffprobe = "ffmpeg", "ffprobe"

    monkeypatch.setattr(fa, "RealFfmpegAdapter", FakeReal)


def test_register_job_uses_actual_selection(editor, stub_context, tmp_path, monkeypatch):
    """핸들을 75~87초로 옮긴 뒤 등록하면 job에 실제 start/end가 전달되어야 한다."""
    from voice_studio.ui import voice_editor_dialog as ved
    from voice_studio.workers.job_schema import parse_job
    captured = {}
    _fake_worker(monkeypatch, ved, captured)
    _ffmpeg_stub(monkeypatch)
    ref = tmp_path / "ref.mp3"
    ref.write_bytes(b"x")
    editor.source_path = str(ref)
    editor.name_edit.setText("내 목소리")
    editor.transcript_edit.setPlainText("안녕하세요. 테스트 대사입니다.")
    editor.consent.setChecked(True)
    editor.wave.duration = 600.0
    editor.wave.set_selection(75.0, 87.0)

    editor.save()

    assert captured.get("job_file"), "worker가 시작되지 않았다"
    job = parse_job(json.loads(Path(captured["job_file"]).read_text(encoding="utf-8")))
    assert job.mode == "register"
    assert job.start_s == 75.0
    assert job.end_s == 87.0
    assert job.name == "내 목소리"


def test_register_validation_blocks_bad_input(editor, stub_context, tmp_path, monkeypatch):
    """이름/파일/구간 길이/대사/권한 누락 시 worker를 시작하지 않고 안내한다."""
    from voice_studio.ui import voice_editor_dialog as ved
    captured = {}
    _fake_worker(monkeypatch, ved, captured)
    _ffmpeg_stub(monkeypatch)
    shown = []
    monkeypatch.setattr(ved.QMessageBox, "warning",
                        lambda *a, **k: shown.append(a[2] if len(a) > 2 else ""))
    monkeypatch.setattr(ved.QMessageBox, "information",
                        lambda *a, **k: shown.append(a[2] if len(a) > 2 else ""))
    ref = tmp_path / "ref.mp3"
    ref.write_bytes(b"x")
    editor.source_path = str(ref)
    editor.wave.duration = 600.0
    editor.wave.set_selection(0.0, 10.0)
    editor.transcript_edit.setPlainText("대사")
    editor.consent.setChecked(True)

    # 이름 누락
    editor.save()
    assert not captured.get("job_file") and shown
    shown.clear()
    editor.name_edit.setText("이름")

    # 선택 구간이 앱 최소 길이 미만
    editor.wave.set_selection(0.0, 2.0)
    editor.save()
    assert not captured.get("job_file") and shown
    shown.clear()
    editor.wave.set_selection(0.0, 10.0)

    # 시작 >= 끝
    editor.wave.set_selection(50.0, 50.0)
    editor.save()
    assert not captured.get("job_file") and shown
    shown.clear()

    # 정상 입력 → worker 시작
    editor.wave.set_selection(10.0, 25.0)
    editor.save()
    assert captured.get("job_file")


def test_register_duplicate_name_blocked(editor, stub_context, tmp_path, monkeypatch):
    from voice_studio.ui import voice_editor_dialog as ved
    captured = {}
    _fake_worker(monkeypatch, ved, captured)
    _ffmpeg_stub(monkeypatch)
    shown = []
    monkeypatch.setattr(ved.QMessageBox, "warning",
                        lambda *a, **k: shown.append(a[2] if len(a) > 2 else ""))
    stub_context.profile_service.list_profiles = lambda: [types.SimpleNamespace(uuid="other", name="내 목소리")]
    ref = tmp_path / "ref.mp3"
    ref.write_bytes(b"x")
    editor.source_path = str(ref)
    editor.name_edit.setText("내 목소리")
    editor.transcript_edit.setPlainText("대사")
    editor.consent.setChecked(True)
    editor.wave.duration = 600.0
    editor.wave.set_selection(0.0, 10.0)
    editor.save()
    assert not captured.get("job_file") and shown


def test_register_model_missing_shows_user_message(editor, stub_context, tmp_path, monkeypatch):
    """모델 미다운로드 시 traceback 대신 사용자 메시지를 보여준다(P12.1-06)."""
    from voice_studio.ui import voice_editor_dialog as ved
    from voice_studio.core.errors import ModelNotDownloadedError
    captured = {}
    _fake_worker(monkeypatch, ved, captured)
    _ffmpeg_stub(monkeypatch)
    shown = []
    monkeypatch.setattr(ved.QMessageBox, "warning",
                        lambda *a, **k: shown.append(a[2] if len(a) > 2 else ""))

    def boom():
        raise ModelNotDownloadedError("음성 모델이 아직 받아지지 않았습니다. 설정에서 모델을 받은 뒤 다시 시도해 주세요.")
    stub_context.model_manager = types.SimpleNamespace(model_path=boom)
    ref = tmp_path / "ref.mp3"
    ref.write_bytes(b"x")
    editor.source_path = str(ref)
    editor.name_edit.setText("이름")
    editor.transcript_edit.setPlainText("대사")
    editor.consent.setChecked(True)
    editor.wave.duration = 600.0
    editor.wave.set_selection(0.0, 10.0)
    editor.save()
    assert shown and "모델" in shown[0]
    assert not captured.get("job_file")
