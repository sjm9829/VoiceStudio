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
    from voice_studio.core.job_coordinator import JobCoordinator
    ctx.jobs = JobCoordinator()
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
    out = Path(job.output_path)  # OS-independent path contract
    assert out.name == "result.mp3"
    assert out.parent.name == payload["job_id"]
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


# ---- P14 UX regression ----

def test_voice_combo_preserves_selection_after_refresh(qtbot, stub_context):
    """목소리 관리 대화상자 닫은 뒤에도 선택한 목소리가 유지된다."""
    from types import SimpleNamespace
    from voice_studio.ui.main_window import MainWindow
    stub_context.profile_service = SimpleNamespace(
        list_profiles=lambda: [SimpleNamespace(name="A", uuid="u-a"),
                               SimpleNamespace(name="B", uuid="u-b")])
    win = MainWindow(stub_context)
    qtbot.addWidget(win)
    assert win.voice_combo.currentData() == "u-a"
    win.voice_combo.setCurrentIndex(1)
    win._refresh_voices()
    assert win.voice_combo.currentData() == "u-b"


def test_script_ctrl_enter_shortcut_registered(qtbot, stub_context):
    """대본 화면에 Ctrl+Return 생성 단축키가 등록돼 있다."""
    from PySide6.QtGui import QShortcut, QKeySequence
    from voice_studio.ui.main_window import MainWindow
    win = MainWindow(stub_context)
    qtbot.addWidget(win)
    shortcuts = [s for s in win.findChildren(QShortcut)
                 if s.objectName() == "scriptGenerateShortcut"]
    assert len(shortcuts) == 1
    assert shortcuts[0].key() == QKeySequence("Ctrl+Return")


def test_voice_manager_double_click_opens_edit(qtbot, stub_context):
    """목록 항목 더블 클릭이 edit_voice 슬롯으로 연결된다(중첩 exec 회피: 슬롯 기록)."""
    from types import SimpleNamespace
    from voice_studio.ui.voice_manager_dialog import VoiceManagerDialog
    stub_context.profile_service = SimpleNamespace(
        list_profiles=lambda: [SimpleNamespace(
            name="A", uuid="u-a", created_at="2025-01-01T00:00:00",
            reference_duration_ms=5000)])
    calls = []
    dlg = VoiceManagerDialog(stub_context)
    qtbot.addWidget(dlg)
    dlg.edit_voice = lambda: calls.append(1)
    assert dlg.list.count() == 1
    dlg.list.itemDoubleClicked.emit(dlg.list.item(0))
    assert calls == [1]
