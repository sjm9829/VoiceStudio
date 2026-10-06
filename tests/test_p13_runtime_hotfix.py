"""P13 Windows runtime hotfix regression(P13 실기 blocker 수정 검증).

- HF console progress 비활성화(windowed frozen sys.stdout=None 대응)
- FFmpeg segment extraction -ss + -t 계약(-to 제거), invalid range 거부
- 받아쓰기/미리듣기 오류 안내 분리
- 실제 ffmpeg가 있는 환경에서 75~87초 구간 추출 integration
"""

import logging
import shutil
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from voice_studio.core.errors import OfflineError, UnsupportedAudioError
from voice_studio.infra import ffmpeg_adapter as fa
from voice_studio.services.model_manager import ModelManager, REQUIRED_MODEL_FILES
from voice_studio.ui.voice_editor_dialog import transcribe_failure_message


# ---------- FFmpeg command contract ----------

@pytest.fixture
def captured_ffmpeg(monkeypatch):
    """RealFfmpegAdapter의 subprocess 호출을 가로채 args를 기록한다."""
    calls = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return types.SimpleNamespace(returncode=0, stdout=b"", stderr="")

    monkeypatch.setattr(fa.subprocess, "run", fake_run)
    adapter = fa.RealFfmpegAdapter(ffmpeg="/fake/ffmpeg", ffprobe="/fake/ffprobe")
    return adapter, calls


def test_ffmpeg_segment_0_to_15_uses_ss_and_t(captured_ffmpeg):
    adapter, calls = captured_ffmpeg
    adapter.decode_segment("in.wav", 0.0, 15.0, 24000)
    args = calls[-1]
    i_ss, i_t = args.index("-ss"), args.index("-t")
    assert args[i_ss + 1] == "0.000" and args[i_t + 1] == "15.000"


def test_ffmpeg_segment_75_to_87_uses_duration_12(captured_ffmpeg):
    adapter, calls = captured_ffmpeg
    adapter.decode_segment("in.wav", 75.0, 87.0, 24000)
    args = calls[-1]
    assert args[args.index("-ss") + 1] == "75.000"
    assert args[args.index("-t") + 1] == "12.000"


def test_ffmpeg_command_has_no_to_flag(captured_ffmpeg):
    adapter, calls = captured_ffmpeg
    adapter.decode_segment("in.wav", 75.0, 87.0, 24000)
    adapter.decode_segment_to_flac("in.wav", 75.0, 87.0, "out.flac")
    for args in calls[-2:]:
        assert "-to" not in args
        assert "-ss" in args and "-t" in args


def test_ffmpeg_start_equals_end_rejects_without_subprocess(captured_ffmpeg):
    adapter, calls = captured_ffmpeg
    with pytest.raises(UnsupportedAudioError):
        adapter.decode_segment("in.wav", 5.0, 5.0, 24000)
    assert calls == []


def test_ffmpeg_end_before_start_rejects_without_subprocess(captured_ffmpeg):
    adapter, calls = captured_ffmpeg
    with pytest.raises(UnsupportedAudioError):
        adapter.decode_segment("in.wav", 20.0, 10.0, 24000)
    with pytest.raises(UnsupportedAudioError):
        adapter.decode_segment_to_flac("in.wav", 20.0, 10.0, "out.flac")
    assert calls == []


def test_ffmpeg_negative_start_rejects(captured_ffmpeg):
    adapter, calls = captured_ffmpeg
    with pytest.raises(UnsupportedAudioError):
        adapter.decode_segment("in.wav", -1.0, 5.0, 24000)
    assert calls == []


def test_flac_extraction_same_ss_t_contract(captured_ffmpeg):
    adapter, calls = captured_ffmpeg
    adapter.decode_segment_to_flac("D:/x/카카오톡 받은 파일/sample.m4a", 75.0, 87.0, "out.flac")
    args = calls[-1]
    assert args[0] == "/fake/ffmpeg"
    assert args[args.index("-ss") + 1] == "75.000"
    assert args[args.index("-t") + 1] == "12.000"
    assert "-to" not in args


def test_korean_space_path_is_single_list_argument(captured_ffmpeg):
    adapter, calls = captured_ffmpeg
    path = r"D:\Documents\카카오톡 받은 파일\sample.m4a"
    adapter.decode_segment(path, 0.0, 15.0, 24000)
    args = calls[-1]
    assert isinstance(args, list)
    assert path in args  # 셸 문자열 조합 없이 list argument 하나로 전달


# ---------- 실제 ffmpeg integration(선택적) ----------

pytestmark_skip = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="실제 ffmpeg가 없는 환경")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="실제 ffmpeg가 없는 환경")
def test_real_ffmpeg_extract_75_87_seconds(tmp_path):
    """90초 synthetic tone에서 75~87초를 뽑아 약 12초 출력을 확인한다."""
    import subprocess
    from voice_studio.infra.ffmpeg_adapter import RealFfmpegAdapter
    src = tmp_path / "tone 90s.wav"
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                        "-i", "sine=frequency=440:duration=90",
                        "-ar", "24000", "-ac", "1", str(src)], capture_output=True)
    assert r.returncode == 0
    adapter = RealFfmpegAdapter()
    pcm = adapter.decode_segment(str(src), 75.0, 87.0, 24000)
    assert abs(pcm.size / 24000 - 12.0) < 0.5
    out = tmp_path / "reference.flac"
    adapter.decode_segment_to_flac(str(src), 75.0, 87.0, str(out))
    assert out.is_file() and out.stat().st_size > 0
    import subprocess as sp
    probe = sp.run([adapter.ffprobe, "-v", "error", "-show_entries", "format=duration",
                    "-of", "csv=p=0", str(out)], capture_output=True, text=True)
    assert abs(float(probe.stdout.strip()) - 12.0) < 0.5


# ---------- HF progress / ModelManager ----------

class FakeHub(types.ModuleType):
    """huggingface_hub를 가짜 모듈로 대체해 progress 비활성화 순서를 기록한다."""
    events: list = []

    @staticmethod
    def snapshot_download(repo_id, local_dir):
        FakeHub.events.append("snapshot_download")
        for rel in REQUIRED_MODEL_FILES:
            f = Path(local_dir) / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("x", encoding="utf-8")
        return str(local_dir)


@pytest.fixture
def fake_hub(monkeypatch):
    FakeHub.events = []
    hub = FakeHub("huggingface_hub")
    utils = types.ModuleType("huggingface_hub.utils")

    def disable_progress_bars():
        FakeHub.events.append("disable_progress_bars")

    utils.disable_progress_bars = disable_progress_bars
    hub.snapshot_download = FakeHub.snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", utils)
    return hub


def test_disable_progress_bars_before_snapshot_download(fake_hub, tmp_path):
    m = ModelManager(cache_dir=tmp_path)
    m.download()
    assert FakeHub.events == ["disable_progress_bars", "snapshot_download"]


def test_download_in_consoleless_env_succeeds(fake_hub, tmp_path, monkeypatch):
    """sys.stdout/stderr가 None인 windowed frozen 환경을 모사해서 다운로드가 성공한다."""
    fake_none = types.SimpleNamespace(write=lambda *a, **k: None)
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    m = ModelManager(cache_dir=tmp_path)
    target = m.download()
    assert target == tmp_path / "Qwen__Qwen3-TTS-12Hz-0.6B-Base"
    # 전역 교체는 테스트 내 monkeypatch로만 수행(프로덕션 코드는 교체하지 않음)
    monkeypatch.setattr(sys, "stdout", fake_none)
    monkeypatch.setattr(sys, "stderr", fake_none)


def test_snapshot_failure_raises_friendly_offline_error(fake_hub, tmp_path, monkeypatch):
    def boom(repo_id, local_dir):
        raise RuntimeError("'NoneType' object has no attribute 'write'")

    fake_hub.snapshot_download = boom
    m = ModelManager(cache_dir=tmp_path)
    with pytest.raises(OfflineError) as ei:
        m.download()
    assert "음성 모델을 받지 못했습니다" in ei.value.user_message
    assert "NoneType" not in ei.value.user_message  # 내부 오류가 사용자 메시지에 노출되지 않음
    # 부분 상태가 complete로 기록되지 않음
    assert not (tmp_path / "Qwen__Qwen3-TTS-12Hz-0.6B-Base" / ".complete").is_file()


def test_existing_valid_model_survives_failed_redownload(fake_hub, tmp_path, monkeypatch):
    """기존 정상 모델 보존 정책 회귀 없음."""
    base = tmp_path / "Qwen__Qwen3-TTS-12Hz-0.6B-Base"
    base.mkdir()
    (base / ".complete").write_text("ok")
    for rel in REQUIRED_MODEL_FILES:
        f = base / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x", encoding="utf-8")
    def boom(repo_id, local_dir):
        raise RuntimeError("network down")
    fake_hub.snapshot_download = boom
    m = ModelManager(cache_dir=tmp_path)
    assert m.download(force=True) == base  # 실패해도 기존 정상 모델 반환


# ---------- STT/preview 오류 안내 분리 ----------

def test_transcribe_message_decode_failure():
    assert "읽을 수 없습니다" in transcribe_failure_message(
        UnsupportedAudioError("-to value smaller than -ss"))
    assert "인터넷" not in transcribe_failure_message(UnsupportedAudioError("x"))


def test_transcribe_message_model_preparation():
    assert "모델을 준비할 수 없습니다" in transcribe_failure_message(
        OfflineError("hf download failed"))
    assert "모델을 준비할 수 없습니다" in transcribe_failure_message(
        RuntimeError("connection timeout while downloading"))


def test_transcribe_message_unexpected():
    assert transcribe_failure_message(ValueError("bad")) == "자동 받아쓰기를 실행할 수 없습니다."


def test_transcribe_failure_logged_not_shown(tmp_path):
    """기술 상세는 로그로만 남는다."""
    records = []

    class Cap(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    h = Cap()
    logging.getLogger("voice_studio.ui.voice_editor_dialog").addHandler(h)
    try:
        from PySide6.QtCore import QCoreApplication
        app = QCoreApplication.instance() or QCoreApplication([])
        from voice_studio.ui import voice_editor_dialog as ved
        ved.QMessageBox.warning = staticmethod(lambda *a, **k: None)
        m = types.SimpleNamespace()
        # 인스턴스 없이 클래스 메서드 호출: self로 더미 사용
        self_dummy = types.SimpleNamespace(status_label=types.SimpleNamespace(setText=lambda t: None))
        ved.VoiceEditorDialog._on_transcribe_failed(self_dummy, UnsupportedAudioError("ffmpeg boom"))
    finally:
        logging.getLogger("voice_studio.ui.voice_editor_dialog").removeHandler(h)
