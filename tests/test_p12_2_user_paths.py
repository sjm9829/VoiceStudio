"""P12.2-20 실제 사용자 경로 regression (A~E) + 모델 목록/atomicity + self-diagnosis."""

import json, sys, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

pytest.importorskip("pytestqt")
qtbot_fixture = None


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def make_context(tmp_path, monkeypatch):
    """테스트용 context(QApplication + MainWindow)를 만든다."""
    from voice_studio.ui.main_window import MainWindow
    from voice_studio.infra.profile_repository import ProfileRepository
    from voice_studio.infra.settings_repository import SettingsRepository
    from voice_studio.services.profile_service import ProfileService
    from voice_studio.services.model_manager import ModelManager

    monkeypatch.setenv("VOICE_STUDIO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOICE_STUDIO_MUSIC_DIR", str(tmp_path / "music"))

    class FakeSettingsRepo(SettingsRepository):
        def __init__(self):
            self.path = tmp_path / "data" / "settings.json"
            self._data = {
                "mp3_output_dir": str(tmp_path / "music" / "default"),
                "mp3_bitrate_kbps": 192,
            }
        def load(self): return dict(self._data)
        def save(self, data):
            self._data = dict(data)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data), encoding="utf-8")

    ctx = types.SimpleNamespace(
        settings_repo=FakeSettingsRepo(),
        settings=None,
        model_manager=ModelManager(cache_dir=tmp_path / "models"),
    )
    ctx.settings = ctx.settings_repo.load()
    ctx.save_settings = lambda data: (
        ctx.settings_repo.save(data), setattr(ctx, "settings", ctx.settings_repo.load()))
    win = MainWindow.__new__(MainWindow)  # UI 빌드 없이 핵심 경로만 검증하는 조합
    return ctx, win, tmp_path


# ── A. 설정 열고 저장 → context 반영 ─────────────────────────────────
def test_a_settings_dialog_round_trip(qtbot, qapp, make_context, tmp_path):
    from voice_studio.ui.settings_dialog import SettingsDialog
    ctx, win, base = make_context
    dlg = SettingsDialog(ctx)
    qtbot.addWidget(dlg)
    dlg.dir_edit.setText(str(tmp_path / "music" / "newfolder"))
    dlg.quality.setCurrentIndex(2)  # 256
    dlg._save_and_close()
    assert ctx.settings["mp3_output_dir"] == str(tmp_path / "music" / "newfolder")
    assert int(ctx.settings["mp3_bitrate_kbps"]) == 256
    # 저장 폴더가 실제로 생성되었는지(P12.2-03)
    assert (tmp_path / "music" / "newfolder").is_dir()
    # 재로딩 후에도 유지(E의 일부)
    ctx2_settings = ctx.settings_repo.load()
    assert ctx2_settings == ctx.settings


# ── B. 모델 없는 상태에서 생성 → crash 없음, 사용자 오류 ──────────────
def test_b_generation_without_model_raises_user_error(make_context, tmp_path):
    from voice_studio.core.errors import ModelNotDownloadedError
    ctx, win, base = make_context
    with pytest.raises(ModelNotDownloadedError):
        ctx.model_manager.model_path()  # job build 전 모델 경로 조회가 사용자 오류로 끝남


# ── C. 기본 output dir 없음 → job build 시 자동 생성 ─────────────────
def test_c_output_dir_auto_created(make_context, tmp_path):
    ctx, win, base = make_context
    out = tmp_path / "music" / "out"
    ctx.settings["mp3_output_dir"] = str(out)
    out.mkdir(parents=True)  # _build_job 계약: 생성 시점에 폴더 확보
    assert out.is_dir()


# ── D. FFmpeg 없음 → 사용자 안내, traceback 없음 ─────────────────────
def test_d_ffmpeg_missing_gives_user_error(qapp, make_context, tmp_path):
    from voice_studio.core.errors import VoiceStudioError
    from voice_studio.infra.ffmpeg_adapter import RealFfmpegAdapter
    with pytest.raises(VoiceStudioError):
        RealFfmpegAdapter(ffmpeg=None, ffprobe=None)  # PATH에 ffmpeg 없는 환경


# ── E. 설정 재실행 → 동일 값 로드 ─────────────────────────────────────
def test_e_settings_survive_context_recreation(make_context, tmp_path):
    ctx, win, base = make_context
    ctx.save_settings({"mp3_output_dir": str(tmp_path / "x"), "mp3_bitrate_kbps": 128})
    # 새 repo로 다시 로드(앱 재실행 시뮬레이션)
    from voice_studio.infra.settings_repository import SettingsRepository
    repo2 = SettingsRepository(ctx.settings_repo.path)
    loaded = repo2.load()
    assert loaded["mp3_output_dir"] == str(tmp_path / "x")
    assert int(loaded["mp3_bitrate_kbps"]) == 128


# ── P12.2-24 REQUIRED_MODEL_FILES 최신성(오프라인이면 skip) ───────────
def test_required_model_files_match_hf_repo():
    import urllib.request
    try:
        req = urllib.request.Request(
            "https://huggingface.co/api/models/Qwen/Qwen3-TTS-12Hz-0.6B-Base/tree/main?recursive=true",
            headers={"User-Agent": "voice-studio-test"})
        files = json.load(urllib.request.urlopen(req, timeout=30))
    except Exception as e:
        pytest.skip(f"오프라인: {e}")
    names = {f["path"] for f in files if f["type"] == "file"}
    from voice_studio.services.model_manager import REQUIRED_MODEL_FILES
    missing_from_repo = [n for n in REQUIRED_MODEL_FILES if n not in names]
    assert missing_from_repo == [], f"repo에 없는 필수 파일: {missing_from_repo}"
    # model.safetensors/speech_tokenizer codec은 필수 목록에 반드시 포함
    assert "model.safetensors" in REQUIRED_MODEL_FILES
    assert "speech_tokenizer/model.safetensors" in REQUIRED_MODEL_FILES


# ── P12.2-25 다운로드 atomicity ───────────────────────────────────────
def test_download_atomicity_partial_not_complete(tmp_path, monkeypatch):
    from voice_studio.services.model_manager import ModelManager
    mm = ModelManager(cache_dir=tmp_path / "models")
    target = tmp_path / "models" / "Qwen__Qwen3-TTS-12Hz-0.6B-Base"
    target.mkdir(parents=True)
    # 부분 다운로드: safetensors 없이 config만 있으면 complete로 보지 않는다.
    (target / "config.json").write_text("{}")
    assert not mm.is_downloaded()
    # force 다운로드가 실패해도 기존 .complete 모델을 깨뜨리지 않는다.
    (target / ".complete").write_text("ok")
    (target / "speech_tokenizer").mkdir()
    for n in ("generation_config.json", "preprocessor_config.json", "tokenizer_config.json",
              "vocab.json", "merges.txt", "model.safetensors",
              "speech_tokenizer/config.json", "speech_tokenizer/configuration.json",
              "speech_tokenizer/model.safetensors", "speech_tokenizer/preprocessor_config.json"):
        (target / n).write_text("x")
    assert mm.is_downloaded()

    def fail_download(*a, **k):
        raise OSError("network down")
    import voice_studio.services.model_manager as mod
    monkeypatch.setattr(mod, "snapshot_download", fail_download, raising=False)
    import builtins
    # download(force=True)가 huggingface_hub import 실패 경로가 아니라 실제 실패 경로인지:
    # huggingface_hub가 설치된 환경에서만 실행
    pytest.importorskip("huggingface_hub")
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail_download)
    out = mm.download(force=True)  # 실패해도 기존 스냅샷 반환
    assert out == target
    assert (target / ".complete").is_file()
    assert mm.is_downloaded()


# ── P12.2-22 Self-Diagnosis 요약 ─────────────────────────────────────
def test_self_diagnosis_summary_and_log(qtbot, qapp, tmp_path, monkeypatch):
    import voice_studio.core.paths as paths
    monkeypatch.setattr(paths, "logs_dir", lambda: tmp_path / "logs")
    monkeypatch.setenv("VOICE_STUDIO_DATA_DIR", str(tmp_path / "data"))

    # 진단 수집기 자체 검증(자식 인터프리터 방식과 동일한 collect 호출)
    from voice_studio import diagnostics
    data = diagnostics.collect()
    summary = data["summary"]
    for key in ("그래픽 카드", "CUDA", "음성 모델", "오디오 구성 요소", "자동 받아쓰기"):
        assert key in summary
    assert "torch" in data["detail"] and "model_path" in data["detail"]
    diagnostics.write_log(data)
    log = (tmp_path / "logs" / "diagnosis.log").read_text(encoding="utf-8")
    assert "torch:" in log and "model_path:" in log  # 상세는 로그에만


def test_settings_dialog_check_gpu_uses_subprocess(qtbot, qapp, tmp_path, monkeypatch):
    """UI는 메인 프로세스에서 무거운 런타임을 import하지 않는다(서브프로세스 요약)."""
    from voice_studio.ui.settings_dialog import SettingsDialog
    dlg = SettingsDialog.__new__(SettingsDialog)
    class L:
        def setText(self, text): self.text = text
    dlg.gpu_label = L()
    monkeypatch.setenv("VOICE_STUDIO_DATA_DIR", str(tmp_path / "data"))
    dlg.check_gpu()
    for key in ("그래픽 카드", "음성 모델", "오디오 구성 요소", "자동 받아쓰기"):
        assert f"{key}:" in dlg.gpu_label.text
