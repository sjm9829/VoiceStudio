"""P12.2 단위 regression: 설정 저장 API/복원, 출력 폴더 생성, 결과 경로 계약."""

import sys, json, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
pytest.importorskip("PySide6", reason="PySide6 미설치 환경")


class FakeSettingsRepo:
    def __init__(self, path):
        self.path = path
        self.saved = None
    def load(self):
        if self.saved is not None:
            return dict(self.saved)
        return {"mp3_output_dir": "", "mp3_bitrate_kbps": 192}
    def save(self, data):
        self.saved = dict(data)
        self.path.write_text(json.dumps(self.saved, ensure_ascii=False), encoding="utf-8")


def make_context(tmp_path, with_model_manager=True):
    from voice_studio.app_context import AppContext
    ctx = AppContext.__new__(AppContext)
    ctx.settings_repo = FakeSettingsRepo(tmp_path / "settings.json")
    ctx.settings = ctx.settings_repo.load()
    if with_model_manager:
        snap = tmp_path / "snap"
        snap.mkdir(parents=True, exist_ok=True)
        ctx.model_manager = types.SimpleNamespace(
            model_path=lambda: str(snap),
            status_text=lambda: "다운로드됨", download=lambda force=False: snap)
    return ctx


def test_context_save_settings_updates_settings_and_file(tmp_path):
    """P12.2-01: save_settings 후 context.settings 갱신 + 파일 반영."""
    from voice_studio.app_context import AppContext
    assert hasattr(AppContext, "save_settings")
    ctx = make_context(tmp_path)
    ctx.save_settings({**ctx.settings, "mp3_bitrate_kbps": 256,
                       "mp3_output_dir": str(tmp_path / "가자 폴더")})
    assert ctx.settings["mp3_bitrate_kbps"] == 256
    assert ctx.settings["mp3_output_dir"] == str(tmp_path / "가자 폴더")
    on_disk = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert on_disk["mp3_bitrate_kbps"] == 256


def test_settings_dialog_restores_saved_values(qtbot, tmp_path):
    """P12.2-16: 저장된 128/192/256이 대화상자에 그대로 표시된다."""
    from voice_studio.ui.settings_dialog import SettingsDialog
    for kbps in (128, 192, 256):
        ctx = make_context(tmp_path / f"s{kbps}")
        ctx.save_settings({**ctx.settings, "mp3_bitrate_kbps": kbps})
        dlg = SettingsDialog(ctx)
        qtbot.addWidget(dlg)
        assert dlg.quality.currentData() == kbps


def test_settings_dialog_save_creates_user_output_dir(qtbot, tmp_path):
    """P12.2-03: 사용자 지정 폴더도 저장 시점에 확보된다."""
    from voice_studio.ui.settings_dialog import SettingsDialog
    ctx = make_context(tmp_path)
    target = tmp_path / "새 폴더" / "보이스"
    dlg = SettingsDialog(ctx)
    qtbot.addWidget(dlg)
    dlg.dir_edit.setText(str(target))
    dlg._save_and_close()
    assert target.is_dir()
    assert ctx.settings["mp3_output_dir"] == str(target)


def test_narrate_output_path_in_job_cache_and_dir_created(qtbot, tmp_path):
    """P12.2-03/04: 출력 폴더 자동 생성 + worker 결과는 jobs/<job_id>/result.mp3."""
    from voice_studio.ui.main_window import MainWindow
    from voice_studio.core.paths import jobs_cache_dir
    ctx = types.SimpleNamespace()
    ctx.settings = {"mp3_output_dir": str(tmp_path / "음악" / "보이스 스튜디오"),
                    "mp3_bitrate_kbps": 256}
    ctx.profile_repository = types.SimpleNamespace(root=tmp_path / "profiles")
    ctx.model_manager = types.SimpleNamespace(model_path=lambda: str(tmp_path / "snap"))
    ctx.profile_service = types.SimpleNamespace(list_profiles=lambda: [])
    ctx.audio = ctx.transcriber = None
    win = MainWindow(ctx)
    qtbot.addWidget(win)
    win.voice_combo.addItem("테스트", "p-uuid")
    payload = win._build_job("p-uuid", "한 문장짜리 대본입니다.")
    out = Path(payload["output_path"])
    assert out.name == "result.mp3"
    assert out.parent.parent == jobs_cache_dir()
    assert out.parent.name == payload["job_id"]
    # 기본 출력 폴더가 없어도 생성 시작 전에 만들어진다(한글 경로 포함)
    assert Path(ctx.settings["mp3_output_dir"]).is_dir()


def test_success_cleanup_keeps_result_file(qtbot, tmp_path, monkeypatch):
    """P12.2-04: 성공 시 job 폴더(result.mp3) 보존, 실패/취소 시 삭제."""
    from voice_studio.ui import main_window as mw_mod
    from voice_studio.ui.main_window import MainWindow
    ctx = types.SimpleNamespace()
    ctx.settings = {"mp3_output_dir": str(tmp_path / "out"), "mp3_bitrate_kbps": 192}
    ctx.profile_repository = types.SimpleNamespace(root=tmp_path / "profiles")
    ctx.model_manager = types.SimpleNamespace(model_path=lambda: "")
    ctx.profile_service = types.SimpleNamespace(list_profiles=lambda: [])
    win = MainWindow(ctx)
    qtbot.addWidget(win)
    from voice_studio.core.paths import safe_job_cache_dir
    jd = safe_job_cache_dir("job-keep")
    jd.mkdir(parents=True, exist_ok=True)
    (jd / "result.mp3").write_bytes(b"MP3")
    win._job_dir = jd
    win._cleanup_job(keep_output=True)
    assert (jd / "result.mp3").is_file()
    jd2 = safe_job_cache_dir("job-drop")
    jd2.mkdir(parents=True, exist_ok=True)
    win._job_dir = jd2
    win._cleanup_job()
    assert not jd2.exists()


def test_missing_model_generation_shows_user_message_no_nameerror(qtbot, tmp_path, monkeypatch):
    """P12.2-02: 모델 미다운로드 상태에서 생성 → NameError 없이 사용자 안내."""
    from voice_studio.ui import main_window as mw_mod
    from voice_studio.ui.main_window import MainWindow
    from voice_studio.core.errors import ModelNotDownloadedError
    ctx = types.SimpleNamespace()
    ctx.settings = {"mp3_output_dir": str(tmp_path / "out"), "mp3_bitrate_kbps": 192}
    ctx.profile_repository = types.SimpleNamespace(root=tmp_path / "profiles")
    def raise_missing():
        raise ModelNotDownloadedError()
    ctx.model_manager = types.SimpleNamespace(model_path=raise_missing)
    ctx.profile_service = types.SimpleNamespace(list_profiles=lambda: [])
    shown = []
    class FakeBox:
        def __init__(self, *a, **k): pass
        def information(self, *a): shown.append(("info", a[1]))
        def warning(self, *a): shown.append(("warn", a[1]))
    monkeypatch.setattr(mw_mod.QMessageBox, "information", lambda *a, **k: shown.append(("info", a[2])))
    monkeypatch.setattr(mw_mod.QMessageBox, "warning", lambda *a, **k: shown.append(("warn", a[2])))
    win = MainWindow(ctx)
    qtbot.addWidget(win)
    win.voice_combo.addItem("테스트", "p-uuid")
    win.script_edit.setPlainText("대본입니다.")
    win.start_generation()
    assert shown == [("warn", "음성 모델이 아직 받아지지 않았습니다. 설정에서 모델을 받아 주세요.")]
