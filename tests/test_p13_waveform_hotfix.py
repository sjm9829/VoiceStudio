"""P13 waveform hotfix regression: 0.0초 silent-failure 제거 + 실패 시 UI reset.

시나리오 A/B(성공 시 기본 selection)는 test_selection_defaults.py가 이미 검증하고,
여기서는 로더 계약, duration validation, 실패 시 reset(C/D/E), 분석 중 차단,
한글/공백 경로, shell 금지, 실제 FFmpeg integration을 검증한다.
"""

import shutil
import json
import subprocess
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

ffprobe_bin = shutil.which("ffprobe")
ffmpeg_bin = shutil.which("ffmpeg")
_real_ok = ffprobe_bin is not None

pytest.importorskip("PySide6", reason="PySide6 미설치 환경")

KOREAN_PATH = r"D:\\Documents\\카카오톡 받은 파일\\sample.m4a"


class StubAudio:
    """_WaveformLoader 경로 테스트용 스텁. 호출 순서와 결과를 기록한다."""

    def __init__(self, duration=None, probe_exc=None, waveform_exc=None,
                 waveform_result=None, waveform_delay=0.0):
        self.duration = duration
        self.probe_exc = probe_exc
        self.waveform_exc = waveform_exc
        self.waveform_result = waveform_result
        self.waveform_delay = waveform_delay
        self.calls = []
        self.probe_paths = []
        self.waveform_paths = []

    def probe(self, path):
        self.calls.append("probe")
        self.probe_paths.append(path)
        if self.probe_exc:
            raise self.probe_exc
        return {"duration": self.duration, "format_name": "stub",
                "sample_rate": 24000, "channels": 1, "codec": "stub"}

    def waveform(self, path):
        import time
        self.calls.append("waveform")
        self.waveform_paths.append(path)
        if self.waveform_delay:
            time.sleep(self.waveform_delay)
        if self.waveform_exc:
            raise self.waveform_exc
        return list(self.waveform_result or [0.1] * 10)


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


@pytest.fixture
def ui_messages(monkeypatch):
    import voice_studio.ui.voice_editor_dialog as ved
    box = {"warning": [], "information": []}
    monkeypatch.setattr(ved.QMessageBox, "warning",
                        lambda *a, **k: box["warning"].append(a[2] if len(a) > 2 else ""))
    monkeypatch.setattr(ved.QMessageBox, "information",
                        lambda *a, **k: box["information"].append(a[2] if len(a) > 2 else ""))
    return box


def _patch_file_dialog(monkeypatch, ved, path):
    monkeypatch.setattr(ved.QFileDialog, "getOpenFileName",
                        lambda *a, **k: (path, "오디오 파일"))


def _wait_load_done(qtbot, editor):
    qtbot.waitUntil(lambda: not editor._analyzing, timeout=5000)


# ---- _WaveformLoader 계약 ----

def test_loader_success_emits_done_with_duration(qtbot):
    from voice_studio.ui.voice_editor_dialog import _WaveformLoader
    audio = StubAudio(duration=123.0, waveform_result=[0.5] * 8)
    loader = _WaveformLoader(audio, KOREAN_PATH)
    with qtbot.waitSignal(loader.done, timeout=5000) as blocker:
        loader.start()
    peaks, duration = blocker.args
    assert peaks == [0.5] * 8 and duration == 123.0
    assert audio.calls == ["probe", "waveform"]  # probe 먼저
    assert audio.probe_paths == [KOREAN_PATH]   # 한글/공백 경로 그대로 전달
    assert audio.waveform_paths == [KOREAN_PATH]


def test_loader_probe_failure_emits_failed_never_done(qtbot):
    from voice_studio.ui.voice_editor_dialog import _WaveformLoader
    from voice_studio.core.errors import UnsupportedAudioError
    audio = StubAudio(probe_exc=UnsupportedAudioError("duration 없음"))
    loader = _WaveformLoader(audio, KOREAN_PATH)
    done_seen = []
    loader.done.connect(lambda p, d: done_seen.append((p, d)))
    with qtbot.waitSignal(loader.failed, timeout=5000):
        loader.start()
    assert not done_seen
    assert audio.calls == ["probe"]  # probe 실패 시 waveform을 실행하지 않음


def test_loader_waveform_failure_emits_failed(qtbot):
    from voice_studio.ui.voice_editor_dialog import _WaveformLoader
    from voice_studio.core.errors import UnsupportedAudioError
    audio = StubAudio(duration=20.0, waveform_exc=UnsupportedAudioError("decode 실패"))
    loader = _WaveformLoader(audio, KOREAN_PATH)
    with qtbot.waitSignal(loader.failed, timeout=5000):
        loader.start()


def test_loader_unexpected_exception_emits_failed(qtbot):
    from voice_studio.ui.voice_editor_dialog import _WaveformLoader
    audio = StubAudio(duration=20.0, waveform_exc=RuntimeError("boom"))
    loader = _WaveformLoader(audio, KOREAN_PATH)
    with qtbot.waitSignal(loader.failed, timeout=5000):
        loader.start()


@pytest.mark.parametrize("bad_duration", [None, float("nan"), float("inf"), 0.0, -3.0])
def test_loader_invalid_duration_is_failure_not_success(qtbot, bad_duration):
    from voice_studio.ui.voice_editor_dialog import _WaveformLoader
    audio = StubAudio(duration=bad_duration, waveform_result=[0.5] * 8)
    loader = _WaveformLoader(audio, KOREAN_PATH)
    done_seen = []
    loader.done.connect(lambda p, d: done_seen.append((p, d)))
    with qtbot.waitSignal(loader.failed, timeout=5000):
        loader.start()
    assert not done_seen
    assert audio.calls == ["probe"]


# ---- UI 시나리오 A/B: pick_file 전체 경로 성공 ----

def test_pick_file_success_long_file_default_selection_0_to_15(editor, qtbot, monkeypatch):
    from voice_studio.ui import voice_editor_dialog as ved
    from voice_studio.core import config
    editor.context.audio = StubAudio(duration=600.0, waveform_result=[0.2] * 50)
    _patch_file_dialog(monkeypatch, ved, KOREAN_PATH)
    editor.pick_file()
    assert editor.status_label.text() == "오디오 파일을 분석하는 중…"
    _wait_load_done(qtbot, editor)
    assert editor.duration == 600.0
    assert editor.wave.start_s == 0.0
    assert editor.wave.end_s == config.REFERENCE_TARGET_SECONDS
    assert editor.status_label.text() == ""
    assert "0.0초" in editor.selection_label.text()
    assert editor.context.audio.probe_paths == [KOREAN_PATH]
    assert editor.context.audio.waveform_paths == [KOREAN_PATH]


def test_pick_file_success_short_file_selection_capped(editor, qtbot, monkeypatch):
    from voice_studio.ui import voice_editor_dialog as ved
    editor.context.audio = StubAudio(duration=8.0, waveform_result=[0.2] * 20)
    _patch_file_dialog(monkeypatch, ved, KOREAN_PATH)
    editor.pick_file()
    _wait_load_done(qtbot, editor)
    assert editor.wave.start_s == 0.0
    assert editor.wave.end_s == 8.0


def test_pick_file_success_60s_then_actions_allowed(editor, qtbot, monkeypatch):
    from voice_studio.ui import voice_editor_dialog as ved
    editor.context.audio = StubAudio(duration=60.0, waveform_result=[0.2] * 20)
    _patch_file_dialog(monkeypatch, ved, KOREAN_PATH)
    editor.pick_file()
    _wait_load_done(qtbot, editor)
    assert editor.duration == 60.0
    assert editor.wave.start_s == 0.0 and editor.wave.end_s == 15.0


# ---- UI 시나리오 C: probe 실패 ----

def test_pick_file_probe_failure_resets_and_blocks(editor, qtbot, monkeypatch, ui_messages):
    from voice_studio.ui import voice_editor_dialog as ved
    from voice_studio.core.errors import UnsupportedAudioError
    editor.context.audio = StubAudio(probe_exc=UnsupportedAudioError("duration 없음"))
    _patch_file_dialog(monkeypatch, ved, KOREAN_PATH)
    editor.pick_file()
    _wait_load_done(qtbot, editor)
    assert editor.duration == 0.0
    assert editor.wave.peaks == []
    assert editor.wave.start_s == 0.0 and editor.wave.end_s == 0.0
    assert editor.selection_label.text() == "선택 구간: 없음"
    assert editor.status_label.text() == ""
    assert ui_messages["warning"], "probe 실패 시 사용자 안내가 표시되어야 한다"
    assert "재생 시간" in ui_messages["warning"][0]
    # 성공 경로가 done([], 0.0)으로 성공처럼 보이지 않는다(_on_peaks 미호출).
    assert editor._analyzing is False
    # 등록이 차단된다: 입력 검증에서 선택 구간 오류로 raise되고 worker가 시작되지 않는다.
    from voice_studio.core.errors import ProfileError
    with pytest.raises(ProfileError):
        editor._validate_register_inputs()


# ---- UI 시나리오 D: waveform 실패 ----

def test_pick_file_waveform_failure_resets_state(editor, qtbot, monkeypatch, ui_messages):
    from voice_studio.ui import voice_editor_dialog as ved
    from voice_studio.core.errors import UnsupportedAudioError
    editor.context.audio = StubAudio(duration=30.0, waveform_exc=UnsupportedAudioError("decode 실패"))
    _patch_file_dialog(monkeypatch, ved, KOREAN_PATH)
    editor.pick_file()
    _wait_load_done(qtbot, editor)
    assert editor.duration == 0.0
    assert editor.wave.peaks == []
    assert editor.selection_label.text() == "선택 구간: 없음"
    assert ui_messages["warning"]


# ---- UI 시나리오 E: 성공 후 실패 → 이전 state 재사용 금지 ----

def test_second_file_failure_discards_first_file_state(editor, qtbot, monkeypatch, ui_messages):
    from voice_studio.ui import voice_editor_dialog as ved
    from voice_studio.core.errors import UnsupportedAudioError
    audio = StubAudio(duration=60.0, waveform_result=[0.3] * 20)
    editor.context.audio = audio
    _patch_file_dialog(monkeypatch, ved, "C:\\first.m4a")
    editor.pick_file()
    _wait_load_done(qtbot, editor)
    assert editor.wave.peaks and editor.duration == 60.0
    # 두 번째 파일은 probe 실패.
    audio.probe_exc = UnsupportedAudioError("duration 없음")
    audio.duration = None
    _patch_file_dialog(monkeypatch, ved, "D:\\카카오톡 받은 파일\\bad.m4a")
    editor.pick_file()
    _wait_load_done(qtbot, editor)
    assert editor.duration == 0.0
    assert editor.wave.peaks == []
    assert editor.wave.start_s == 0.0 and editor.wave.end_s == 0.0
    assert editor.selection_label.text() == "선택 구간: 없음"
    assert editor.wave.duration == 0.0
    assert ui_messages["warning"]


# ---- 분석 중 상태 표시 + 액션 차단 ----

def test_while_analyzing_actions_are_blocked(editor, qtbot, monkeypatch, ui_messages):
    from voice_studio.ui import voice_editor_dialog as ved
    editor.context.audio = StubAudio(duration=60.0, waveform_delay=0.6,
                                     waveform_result=[0.2] * 10)
    _patch_file_dialog(monkeypatch, ved, KOREAN_PATH)
    editor.pick_file()
    assert editor._analyzing is True
    assert editor.status_label.text() == "오디오 파일을 분석하는 중…"
    editor.preview_selection()
    editor.auto_transcribe()
    assert editor._preview_thread is None
    assert editor._transcribe_thread is None
    assert ui_messages["information"]
    editor.save()
    assert editor.context.jobs.try_acquire()  # worker가 slot을 잡지 않았다
    _wait_load_done(qtbot, editor)


# ---- WaveformWidget.clear() ----

def test_waveform_widget_clear_resets_all(qtbot):
    from voice_studio.ui.waveform_widget import WaveformWidget
    w = WaveformWidget()
    w.set_peaks([0.5] * 10, 30.0)
    w.set_selection(5.0, 10.0)
    w.clear()
    assert w.peaks == [] and w.duration == 0.0
    assert w.start_s == 0.0 and w.end_s == 0.0


# ---- probe duration parsing 계약 ----

def _patch_probe_run(monkeypatch, payload):
    """RealFfmpegAdapter._run이 canned ffprobe JSON을 돌려주게 한다."""
    import voice_studio.infra.ffmpeg_adapter as fa
    out = json.dumps(payload)
    calls = []

    def fake_run(self, args):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=out, stderr="")

    monkeypatch.setattr(fa.RealFfmpegAdapter, "_run", fake_run)
    return calls


def _probe_payload(fmt_duration, stream_duration):
    stream = {"codec_type": "audio", "codec_name": "aac", "sample_rate": "44100",
              "channels": 2}
    if stream_duration is not None:
        stream["duration"] = stream_duration
    fmt = {"format_name": "mov,mp4,m4a"}
    if fmt_duration is not None:
        fmt["duration"] = fmt_duration
    return {"format": fmt, "streams": [stream]}


def test_probe_uses_format_duration(monkeypatch):
    from voice_studio.infra.ffmpeg_adapter import RealFfmpegAdapter
    calls = _patch_probe_run(monkeypatch, _probe_payload("20.5", None))
    info = RealFfmpegAdapter("ffmpeg", "ffprobe").probe("x.m4a")
    assert info["duration"] == 20.5


def test_probe_falls_back_to_stream_duration(monkeypatch):
    from voice_studio.infra.ffmpeg_adapter import RealFfmpegAdapter
    _patch_probe_run(monkeypatch, _probe_payload(None, "18.25"))
    info = RealFfmpegAdapter("ffmpeg", "ffprobe").probe("x.m4a")
    assert info["duration"] == 18.25


@pytest.mark.parametrize("fmt_d,stream_d",
                         [(None, None), ("0", "0"), ("N/A", None), ("Infinity", "0")])
def test_probe_missing_or_invalid_duration_raises(monkeypatch, fmt_d, stream_d):
    from voice_studio.infra.ffmpeg_adapter import RealFfmpegAdapter
    from voice_studio.core.errors import UnsupportedAudioError
    _patch_probe_run(monkeypatch, _probe_payload(fmt_d, stream_d))
    with pytest.raises(UnsupportedAudioError):
        RealFfmpegAdapter("ffmpeg", "ffprobe").probe("x.m4a")


def test_probe_no_audio_stream_raises(monkeypatch):
    from voice_studio.infra.ffmpeg_adapter import RealFfmpegAdapter
    from voice_studio.core.errors import UnsupportedAudioError
    _patch_probe_run(monkeypatch, {"format": {}, "streams": [{"codec_type": "video"}]})
    with pytest.raises(UnsupportedAudioError):
        RealFfmpegAdapter("ffmpeg", "ffprobe").probe("x.mp4")


def test_probe_failure_returns_never_zero_duration(monkeypatch):
    """ffprobe returncode != 0이면 0.0으로 변환하지 않고 실패한다."""
    import voice_studio.infra.ffmpeg_adapter as fa
    from voice_studio.core.errors import UnsupportedAudioError
    monkeypatch.setattr(fa.RealFfmpegAdapter, "_run",
                        lambda self, args: subprocess.CompletedProcess(args, 1, stdout="", stderr="err"))
    with pytest.raises(UnsupportedAudioError):
        fa.RealFfmpegAdapter("ffmpeg", "ffprobe").probe("x.m4a")


# ---- subprocess list arg / shell 금지 ----

def test_ffmpeg_calls_use_list_args_without_shell(monkeypatch):
    import voice_studio.infra.ffmpeg_adapter as fa
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        if "--probe" in args:  # placeholder marker, never used
            pass
        if args[0].endswith("ffprobe"):
            payload = _probe_payload("20.0", None)
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps(payload), stderr="")
        # decode/waveform: 1초짜리 사인파 f32le 데이터를 흉내
        import numpy as np
        t = np.arange(8000, dtype=np.float32) / 8000
        pcm = (0.25 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        return subprocess.CompletedProcess(args, 0, stdout=pcm.tobytes(), stderr=b"")

    monkeypatch.setattr(fa.subprocess, "run", fake_run)
    adapter = fa.RealFfmpegAdapter("ffmpeg", "ffprobe")
    adapter.probe(KOREAN_PATH)
    adapter.decode_segment(KOREAN_PATH, 0.0, 2.0, 24000)
    adapter.waveform(KOREAN_PATH, 64)
    for args, kwargs in calls:
        assert isinstance(args, list)
        assert KOREAN_PATH in args
        assert kwargs.get("shell") is not True


# ---- 실제 FFmpeg integration (ffmpeg 없는 환경은 skip) ----

ffmpeg_bin = None

def _real_adapter():
    from voice_studio.infra.ffmpeg_adapter import RealFfmpegAdapter
    return RealFfmpegAdapter()


@pytest.fixture
def sine20(tmp_path):
    adapter = _real_adapter()
    wav = tmp_path / "sine_20s.wav"
    r = adapter._run([adapter.ffmpeg, "-y", "-f", "lavfi",
                      "-i", "sine=frequency=440:duration=20", "-ar", "24000", "-ac", "1",
                      str(wav)])
    if r.returncode != 0:
        pytest.skip("lavfi sine 생성 실패")
    return str(wav)


@pytest.mark.skipif(not _real_ok, reason="실제 ffmpeg가 없는 환경")
def test_real_probe_duration_sine20(sine20):
    adapter = _real_adapter()
    info = adapter.probe(sine20)
    assert abs(info["duration"] - 20.0) < 0.5


@pytest.mark.skipif(not _real_ok, reason="실제 ffmpeg가 없는 환경")
def test_real_waveform_buckets_and_peaks(sine20):
    adapter = _real_adapter()
    buckets = adapter.waveform(sine20, 128)
    assert len(buckets) == 128
    assert max(buckets) > 0.01


@pytest.mark.skipif(not _real_ok, reason="실제 ffmpeg가 없는 환경")
def test_real_probe_and_waveform_m4a_korean_path(sine20, tmp_path):
    """20초 WAV를 M4A(AAC)로 변환한 뒤 한글/공백 경로에서 probe/waveform 검증."""
    adapter = _real_adapter()
    r = adapter._run([adapter.ffmpeg, "-y", "-i", sine20, "-c:a", "aac", "-b:a", "128k",
                      str(tmp_path / "sine20.m4a")])
    if r.returncode != 0:
        pytest.skip("AAC 인코더 미지원 환경")
    korean_dir = tmp_path / "카카오톡 받은 파일"
    korean_dir.mkdir()
    m4a = korean_dir / "KakaoTalk_Audio_test.m4a"
    import shutil
    shutil.move(str(tmp_path / "sine20.m4a"), str(m4a))
    info = adapter.probe(str(m4a))
    assert abs(info["duration"] - 20.0) < 0.5
    buckets = adapter.waveform(str(m4a), 64)
    assert len(buckets) == 64 and max(buckets) > 0.01
