"""UI/worker 연결 계약 테스트(GPU 불필요, offscreen Qt)."""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

pytest.importorskip("PySide6", reason="PySide6 미설치 환경")


@pytest.fixture
def stub_context(tmp_path):
    from voice_studio.services.text_segmenter import segment_with_flags
    ctx = types.SimpleNamespace()
    ctx.settings = {"mp3_output_dir": str(tmp_path / "out"), "mp3_bitrate_kbps": 192}
    ctx.profile_repository = types.SimpleNamespace(root=tmp_path / "profiles")
    ctx.model_manager = types.SimpleNamespace(model_path=lambda: str(tmp_path / "models" / "snap"))
    ctx.profile_service = types.SimpleNamespace(list_profiles=lambda: [])
    ctx.audio = None
    ctx.transcriber = None
    return ctx


def test_main_window_builds_full_narrate_payload(qtbot, stub_context, monkeypatch):
    """MainWindow job 빌더가 worker 필수 필드를 모두 채운다(1-3 계약)."""
    from voice_studio.workers.job_schema import parse_job
    from voice_studio.ui.main_window import MainWindow
    win = MainWindow(stub_context)
    qtbot.addWidget(win)
    payload = win._build_job("p-uuid", "첫 번째 문장입니다.\n\n두 번째 문장입니다.")
    job = parse_job(payload)
    assert job.mode == "narrate"
    assert job.profile_uuid == "p-uuid"
    assert job.segments and len(job.segments) == len(job.gap_flags)
    assert job.profile_dir == str(stub_context.profile_repository.root)
    assert job.model_path.endswith("snap")
    # P12.2-04: worker 결과는 job 캐시(jobs/<job_id>/result.mp3)에 저장된다.
    assert job.output_path.endswith("/result.mp3")
    assert job.output_path.split("/")[-2] == payload["job_id"]
    assert job.bitrate_kbps in (128, 192, 256)
    assert payload["job_id"]


from voice_studio.ui.main_window import MainWindow


def test_main_window_starts_worker_with_launcher(qtbot, stub_context, tmp_path, monkeypatch):
    """생성 버튼이 worker_command(job.json)으로 QProcess를 실행한다(1-3/패키징 계약)."""
    from voice_studio.ui import main_window as mw_mod
    launched = {}
    monkeypatch.setattr(mw_mod, "worker_command", lambda job_file: ("prog", ["--worker", job_file]))

    class _Signal:
        def connect(self, fn): pass

    class FakeProcess:
        def __init__(self, parent=None):
            self.readyReadStandardOutput = _Signal()
            self.readyReadStandardError = _Signal()
            self.finished = _Signal()
        def start(self, program, args):
            launched["args"] = args
        def terminate(self): pass
        def kill(self): pass

    monkeypatch.setattr(mw_mod, "QProcess", FakeProcess)
    win = MainWindow(stub_context)
    qtbot.addWidget(win)
    win.voice_combo.addItem("테스트", "p-uuid")
    win.script_edit.setPlainText("한 문장짜리 대본입니다.")
    win.start_generation()
    assert launched.get("args", [""])[-2:] == ["--worker", launched.get("args", ["", ""])[-1]]
    assert win._last_output is None  # 새 실행 시 stale 결과 제거


def test_register_ui_delegates_to_worker_not_fake_register():
    """등록 화면은 profile_service.register(fake)를 직접 호출하지 않고 worker를 실행한다(1-5 계약)."""
    text = (Path(__file__).resolve().parents[1] / "src" / "voice_studio" / "ui" /
            "voice_editor_dialog.py").read_text(encoding="utf-8")
    assert "profile_service.register(" not in text
    assert "_start_register_worker" in text
    assert "build_register_payload" in text and "worker_command" in text
