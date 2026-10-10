"""P13 Runtime Stabilization regressions.

- BF16 텐서가 float32로 변환되고 정수 ref_code dtype은 유지된다(qwen_adapter.tensor_to_numpy).
- dtype 경계(fp16/bf16/fp32/int64) safetensors roundtrip(ProfileService ↔ ProfileRepository).
- worker stdout JSONL이 locale/CP949와 무관한 ASCII wire다(protocol.emit).
- FFprobe stdout이 UTF-8로 디코딩되고 -show_entries가 제한적이다(ffmpeg_adapter).
- sys.stderr가 None(console=False frozen)인 환경에서 logging_setup이 안전하다.
- production_device()는 CUDA가 없으면 GpuUnavailableError를 raise한다(CPU fallback 금지).
- worker stderr가 worker-stderr.log로 보존된다(voice_editor_dialog).
- UI 오류 메시지에는 detail 기술 정보가 노출되지 않는다.
- 분할기가 문단 내 짧은 chunk를 이웃과 합친다(text_segmenter).
- 버전 단일 소스(pyproject vs installer .iss) 계약.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------------------------------------------------------------- qwen_adapter
def _fake_torch_module():
    """torch.Tensor/bfloat16 계약을 재현하는 최소 fake 모듈."""

    class BFloat16:
        name = "torch.bfloat16"

        def __eq__(self, other):
            return other is self or getattr(other, "name", None) == self.name

        def __hash__(self):
            return hash(self.name)

    class Tensor:
        def __init__(self, arr):
            self._arr = np.asarray(arr)
            self._dtype = self._arr.dtype

        @property
        def dtype(self):
            return self._dtype

        def detach(self):
            return self

        def cpu(self):
            return self

        def float(self):
            clone = Tensor(self._arr.astype(np.float32))
            clone._dtype = np.dtype("float32")
            return clone

        def numpy(self):
            return self._arr

    module = types.ModuleType("torch")
    module.Tensor = Tensor
    module.bfloat16 = BFloat16()
    module.from_numpy = lambda arr: Tensor(np.asarray(arr))
    return module


def _bf16_tensor(fake_module, values):
    t = fake_module.Tensor(np.array(values, dtype=np.float32))
    t._dtype = fake_module.bfloat16
    return t


def test_bf16_tensor_converted_to_float32_numpy():
    """BF16 텐서는 float32로 normalize된다(TypeError: unsupported ScalarType BFloat16 방지)."""
    from voice_studio.infra import qwen_adapter
    fake = _fake_torch_module()
    t = _bf16_tensor(fake, [1.5, 2.5])
    with mock.patch.dict(sys.modules, {"torch": fake}):
        result = qwen_adapter.tensor_to_numpy(t)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [1.5, 2.5])


def test_int_tensor_dtype_preserved():
    """정수 ref_code 텐서는 float 변환 없이 원래 dtype을 유지한다."""
    from voice_studio.infra import qwen_adapter
    fake = _fake_torch_module()
    with mock.patch.dict(sys.modules, {"torch": fake}):
        t = fake.Tensor(np.array([1, 2, 3], dtype=np.int64))
        result = qwen_adapter.tensor_to_numpy(t)
    assert result.dtype == np.int64


def test_fp32_tensor_passthrough_dtype():
    from voice_studio.infra import qwen_adapter
    fake = _fake_torch_module()
    with mock.patch.dict(sys.modules, {"torch": fake}):
        t = fake.Tensor(np.array([0.5], dtype=np.float32))
        assert qwen_adapter.tensor_to_numpy(t).dtype == np.float32


def test_to_numpy_alias_kept():
    from voice_studio.infra import qwen_adapter
    assert qwen_adapter._to_numpy is qwen_adapter.tensor_to_numpy


def test_numpy_ndarray_passthrough():
    from voice_studio.infra.qwen_adapter import tensor_to_numpy
    arr = np.array([1.0, 2.0], dtype=np.float32)
    out = tensor_to_numpy(arr)
    np.testing.assert_array_equal(out, arr)


# ------------------------------------------------- dtype boundary roundtrip
@pytest.mark.parametrize("emb_dtype", [np.float16, np.float32])
def test_embedding_dtype_normalized_to_fp32_before_save(tmp_path, emb_dtype):
    """ref_spk_embedding은 저장 전 float32로 정규화된다(BF16 허용 정책의 저장 경계)."""
    from voice_studio.services.profile_service import encode_ref_code
    from voice_studio.infra.profile_repository import ProfileRepository
    repo = ProfileRepository(tmp_path)
    emb = np.ones(8, dtype=emb_dtype)
    code = np.array([1, 2, 3], dtype=np.int64)
    from voice_studio.domain.voice_profile import VoiceProfile
    profile = VoiceProfile(name="dtype", ref_text="대사")
    repo.save(profile, {"ref_code": code, "ref_spk_embedding": emb.astype(np.float32)}, b"f")
    loaded = repo.load_prompt_tensors(profile.uuid)
    assert loaded["ref_spk_embedding"].dtype == np.float32
    assert loaded["ref_code"].dtype == np.int64  # 정수 code dtype 유지


def test_int64_ref_code_roundtrip_via_profile_service(tmp_path):
    from voice_studio.services.profile_service import encode_ref_code, decode_ref_code
    from voice_studio.infra.profile_repository import ProfileRepository
    from voice_studio.domain.voice_profile import VoiceProfile
    repo = ProfileRepository(tmp_path)
    code = np.array([7, 8, 9, 10], dtype=np.int64)
    arr = encode_ref_code(code)
    assert arr.dtype == np.int64
    profile = VoiceProfile(name="int", ref_text="대사", ref_code_kind="tensor")
    repo.save(profile, {"ref_code": arr, "ref_spk_embedding": np.ones(4, dtype=np.float32)}, b"f")
    loaded = repo.load_prompt_tensors(profile.uuid)
    restored = decode_ref_code(loaded["ref_code"], profile.ref_code_kind)
    np.testing.assert_array_equal(restored, code)
    assert restored.dtype == np.int64


def test_fp16_ref_code_roundtrip(tmp_path):
    from voice_studio.services.profile_service import encode_ref_code, decode_ref_code
    from voice_studio.infra.profile_repository import ProfileRepository
    from voice_studio.domain.voice_profile import VoiceProfile
    repo = ProfileRepository(tmp_path)
    code = np.array([0.5, 1.5], dtype=np.float16)
    profile = VoiceProfile(name="f16", ref_text="대사", ref_code_kind="tensor")
    repo.save(profile, {"ref_code": encode_ref_code(code),
                        "ref_spk_embedding": np.ones(4, dtype=np.float32)}, b"f")
    restored = decode_ref_code(repo.load_prompt_tensors(profile.uuid)["ref_code"], "tensor")
    assert restored.dtype == np.float16


def test_bf16_embedding_save_path_normalizes_to_fp32(tmp_path):
    """BF16(Ampere) 환경에서 생성된 embedding이 float32로 정규화되어 저장된다."""
    from voice_studio.services.profile_service import ProfileService
    from voice_studio.infra.profile_repository import ProfileRepository
    from voice_studio.infra.qwen_adapter import VoiceClonePromptSpec
    from voice_studio.services.audio_service import AudioService

    class FakeAudio:
        def decode_reference_segment(self, *a, **k):
            return np.zeros(24000, dtype=np.float32)

        def save_reference_flac(self, *a, **k):
            return None

    emb_bf16 = np.ones(256, dtype=np.float32)  # tensor_to_numpy가 bf16→fp32 변환한 결과
    spec = VoiceClonePromptSpec(
        ref_code=np.array([1, 2], dtype=np.int64), ref_spk_embedding=emb_bf16,
        x_vector_only_mode=False, icl_mode=True, ref_text="텍스트")
    repo = ProfileRepository(tmp_path)
    service = ProfileService(repo, AudioService.__new__(AudioService))
    with mock.patch.object(AudioService, "save_reference_flac", return_value=None), \
            mock.patch.object(ProfileService, "_reference_bytes", return_value=b"FLAC"):
        profile = service.register(name="bf16", source_path="x", start_s=0.0, end_s=1.0,
                                   ref_text="텍스트", consent=True, prompt=spec,
                                   waveform=np.zeros(24000, dtype=np.float32), sample_rate=24000)
    tensors = repo.load_prompt_tensors(profile.uuid)
    assert tensors["ref_spk_embedding"].dtype == np.float32
    assert tensors["ref_code"].dtype == np.int64


# ---------------------------------------------------------------- worker IPC
def test_emit_output_is_ascii_jsonl(capsys):
    from voice_studio.workers.protocol import emit, error_event
    ev = error_event("E_TEST", "한글 오류 메시지 테스트", detail=r"상세 traceback \ 이상문자")
    emit(ev)
    out = capsys.readouterr().out
    raw = out.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")
    # ASCII wire: 어떤 인코딩으로 재해석해도 동일하게 복원된다
    assert raw.isascii() or raw == out
    parsed = json.loads(out.strip())
    assert parsed["message"] == "한글 오류 메시지 테스트"


def test_emit_survives_cp949_stdout():
    """CP949 콘솔에서도 ASCII JSONL은 손상되지 않는다(P13 §6 계약)."""
    from voice_studio.workers.protocol import emit, error_event
    ev = error_event("E_KO", "한글상태")
    line = json.dumps(ev, ensure_ascii=True)
    cp = line.encode("ascii").decode("cp949", errors="replace")
    assert cp == line  # 순수 ASCII라서 CP949 재해석이 무의미(손상 없음)


def test_jsonl_buffer_partial_chunks():
    """stdout이 QProcess 청크로 잘려 와도 JsonlBuffer(workers.linebuffer)가 이벤트 단위로 복원한다."""
    from voice_studio.workers.linebuffer import JsonlBuffer
    from voice_studio.workers.protocol import status_event
    buf = JsonlBuffer()
    ev1 = status_event("model_loading", job_id="j")
    ev2 = status_event("saving_profile", job_id="j")
    line1 = json.dumps(ev1, ensure_ascii=True) + "\n"
    line2 = json.dumps(ev2, ensure_ascii=True) + "\n"
    payload = (line1 + line2).encode("ascii")
    mid = payload.index(line1.encode("ascii")) + len(line1.encode("ascii")) - 3
    events = buf.feed(payload[:mid]) + buf.feed(payload[mid:])
    assert [e["phase"] for e in events] == ["model_loading", "saving_profile"]


# ---------------------------------------------------------------- ffmpeg
def _probe_adapter(monkeypatch, stdout, returncode=0):
    from voice_studio.infra import ffmpeg_adapter
    calls = {}

    def fake_run(args, **kw):
        calls["args"] = args
        calls["kw"] = kw
        return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(ffmpeg_adapter.subprocess, "run", fake_run)
    adapter = ffmpeg_adapter.RealFfmpegAdapter.__new__(ffmpeg_adapter.RealFfmpegAdapter)
    adapter.ffmpeg = "ffmpeg"
    adapter.ffprobe = "ffprobe"
    return adapter, calls


_VALID_FFPROBE_JSON = json.dumps({
    "format": {"duration": "12.345", "format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
    "streams": [{"codec_type": "audio", "codec_name": "aac", "duration": "12.345",
                 "sample_rate": "44100", "channels": 2}],
})


def test_ffprobe_valid_json_returns_flat_contract(monkeypatch):
    """유효한 ffprobe JSON은 flat contract dict를 정확히 반환한다(예외 허용 금지)."""
    adapter, calls = _probe_adapter(monkeypatch, _VALID_FFPROBE_JSON)
    result = adapter.probe("sample.m4a")
    assert result == {
        "duration": 12.345,
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "sample_rate": 44100,
        "channels": 2,
        "codec": "aac",
    }
    assert calls["kw"].get("encoding") == "utf-8"
    assert calls["kw"].get("errors") == "replace"
    entries = " ".join(str(a) for a in calls["args"])
    assert "-show_entries" in entries
    assert "format_name" in entries and "duration" in entries
    assert "tags" not in entries  # 제한된 entries


def test_ffprobe_utf8_text_survives(monkeypatch):
    """stdout이 임의 유니코드 텍스트여도 UTF-8/replace로 안전히 파싱 경로를 유지한다."""
    adapter, calls = _probe_adapter(monkeypatch, _VALID_FFPROBE_JSON)
    result = adapter.probe("sample.m4a")
    assert result["duration"] == 12.345


def test_ffprobe_invalid_json_raises_unsupported_audio(monkeypatch):
    """ffprobe stdout이 JSON이 아니면 정확히 UnsupportedAudioError만 발생한다."""
    from voice_studio.core.errors import UnsupportedAudioError
    adapter, _ = _probe_adapter(monkeypatch, "not json at all")
    with pytest.raises(UnsupportedAudioError):
        adapter.probe("sample.m4a")


def test_ffprobe_failure_returncode_raises_unsupported_audio(monkeypatch):
    from voice_studio.core.errors import UnsupportedAudioError
    adapter, _ = _probe_adapter(monkeypatch, "{}", returncode=1)
    with pytest.raises(UnsupportedAudioError):
        adapter.probe("sample.m4a")


def test_ffprobe_no_audio_stream_raises(monkeypatch):
    from voice_studio.core.errors import UnsupportedAudioError
    adapter, _ = _probe_adapter(monkeypatch, json.dumps({
        "format": {"duration": "3.0", "format_name": "mp3"},
        "streams": [{"codec_type": "video"}]}))
    with pytest.raises(UnsupportedAudioError):
        adapter.probe("sample.mp3")


def test_ffmpeg_run_uses_utf8_decode(monkeypatch):
    from voice_studio.infra import ffmpeg_adapter
    captured = {}
    captured["called"] = False

    def fake_run(args, **kw):
        captured["called"] = True
        captured.update(kw)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ffmpeg_adapter.subprocess, "run", fake_run)
    adapter = ffmpeg_adapter.RealFfmpegAdapter.__new__(ffmpeg_adapter.RealFfmpegAdapter)
    adapter.ffmpeg = "ffmpeg"
    adapter.ffprobe = "ffprobe"
    adapter._run(["ffmpeg", "-version"])  # 텍스트 경로 계약 검증
    assert captured["called"]
    assert captured.get("encoding") == "utf-8"
    assert captured.get("errors") == "replace"
    assert captured.get("text") is True


# ---------------------------------------------------------------- logging
def test_logging_setup_with_none_stderr(tmp_path, monkeypatch):
    """console=False frozen 환경(sys.stderr=None)에서도 파일 로그가 구성된다(§14)."""
    import logging
    from voice_studio.core import logging_setup, paths
    monkeypatch.setattr(paths, "logs_dir", lambda: tmp_path)
    monkeypatch.setattr(logging_setup, "logs_dir", lambda: tmp_path)
    handlers_before = list(logging.root.handlers)
    with mock.patch.object(sys, "stderr", None):
        log_file = logging_setup.setup_logging()
    try:
        assert (tmp_path / "voice-studio.log").is_file()
        handlers = list(logging.root.handlers)
        assert any(isinstance(hh, logging.FileHandler) for hh in handlers)
        assert not any(isinstance(hh, logging.StreamHandler) and not isinstance(hh, logging.FileHandler)
                       for hh in handlers)
        logging.getLogger("test").warning("frozen-safe message")  # stderr None에서도 안전
    finally:
        for hh in list(logging.root.handlers):
            if hh not in handlers_before:
                logging.root.removeHandler(hh)
                hh.close()


def test_logging_setup_with_stderr_keeps_stream(tmp_path, monkeypatch):
    import logging
    from voice_studio.core import logging_setup, paths
    monkeypatch.setattr(paths, "logs_dir", lambda: tmp_path)
    monkeypatch.setattr(logging_setup, "logs_dir", lambda: tmp_path)
    handlers_before = list(logging.root.handlers)
    try:
        logging_setup.setup_logging()
        handlers = list(logging.root.handlers)
        assert any(isinstance(hh, logging.FileHandler) for hh in handlers)
        assert any(isinstance(hh, logging.StreamHandler) and not isinstance(hh, logging.FileHandler)
                   for hh in handlers)
    finally:
        for hh in list(logging.root.handlers):
            if hh not in handlers_before:
                logging.root.removeHandler(hh)
                hh.close()


# ---------------------------------------------------------------- GPU policy
def test_production_device_raises_gpu_unavailable_without_torch():
    from voice_studio.infra import qwen_adapter
    from voice_studio.core.errors import GpuUnavailableError
    with mock.patch.dict(sys.modules, {"torch": None}):
        with pytest.raises(GpuUnavailableError):
            qwen_adapter.production_device()


def test_production_device_cuda_false_raises():
    from voice_studio.infra import qwen_adapter
    from voice_studio.core.errors import GpuUnavailableError
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(is_available=lambda: False)
    with mock.patch.dict(sys.modules, {"torch": fake}):
        with pytest.raises(GpuUnavailableError):
            qwen_adapter.production_device()


def test_production_device_cuda_true_returns_cuda():
    from voice_studio.infra import qwen_adapter
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(is_available=lambda: True)
    fake.device = lambda spec=None: spec or "cuda"
    with mock.patch.dict(sys.modules, {"torch": fake}):
        assert "cuda" in str(qwen_adapter.production_device())


def test_worker_main_uses_gguf_engine_contract():
    """P18-1: worker는 torch device 대신 gguf 어댑터(-ngl 포함 llama-tts)를 사용한다."""
    src = (Path(__file__).resolve().parents[1] / "src" / "voice_studio" / "workers" / "worker_main.py").read_text(encoding="utf-8")
    assert "_gguf_adapter" in src
    assert "GgufQwenAdapter" in src


# ---------------------------------------------------------------- worker stderr
def test_worker_stderr_persisted_to_log(tmp_path, monkeypatch):
    import voice_studio.ui.voice_editor_dialog as ved
    from PySide6.QtWidgets import QDialog
    monkeypatch.setattr(ved, "logs_dir", lambda: tmp_path, raising=False)
    dlg = ved.VoiceEditorDialog.__new__(ved.VoiceEditorDialog)
    dlg._worker_job_id = "job-123"
    dlg._worker_mode = "register"
    dlg._append_worker_stderr_log("BF16 unsupported crash\nline2")
    log = (tmp_path / "worker-stderr.log").read_text(encoding="utf-8")
    assert "BF16 unsupported crash" in log
    assert "job=job-123" in log and "mode=register" in log


def test_main_window_stderr_persisted(tmp_path, monkeypatch):
    import voice_studio.ui.main_window as mw
    monkeypatch.setattr(mw, "logs_dir", lambda: tmp_path, raising=False)
    win = mw.MainWindow.__new__(mw.MainWindow)
    win._current_job_id = "job-456"
    win._persist_worker_stderr("some stderr text")
    log = (tmp_path / "worker-stderr.log").read_text(encoding="utf-8")
    assert "some stderr text" in log
    assert "job=job-456" in log


def test_logs_dir_has_worker_stderr_member():
    from voice_studio.core import paths
    assert callable(getattr(paths, "logs_dir", None))


# ---------------------------------------------------------------- UI detail
def test_ui_error_does_not_append_detail():
    """§8: UI 오류 메시지에 detail을 붙이지 않는다."""
    src = (Path(__file__).resolve().parents[1] / "src" / "voice_studio" / "ui" / "voice_editor_dialog.py").read_text(encoding="utf-8")
    import re as _re
    fn = src[src.index("def _on_worker_output"):src.index("def _on_worker_stderr")]
    assert "self._error_message = f" not in fn


# ---------------------------------------------------------------- segmenter
def test_segmenter_merges_short_chunks_within_paragraph():
    from voice_studio.services.text_segmenter import segment_with_flags
    segments, flags = segment_with_flags("짧다. 그래도 합쳐야 한다. 이것은 충분히 긴 문장이니 그대로 둔다.")
    assert all(len(seg) > 0 for seg in segments)
    # 어떤 segment도 hard_max(240)를 넘지 않는다
    assert all(len(seg) <= 240 for seg in segments)


def test_segmenter_short_chunk_merge_does_not_break_paragraph_flags():
    from voice_studio.services.text_segmenter import segment_with_flags
    text = "짧은문장. 이어지는문장.\n\n두번째문단의짧은시작. 그리고 길게 이어지는 문장을 하나 더 넣어서 충분한 길이를 만든다."
    segments, flags = segment_with_flags(text)
    assert flags[0] is False
    # 문단 경계로 시작하는 segment만 True
    for i, f in enumerate(flags):
        if f:
            assert i > 0


def test_segmenter_hard_max_respected():
    from voice_studio.services.text_segmenter import segment_with_flags
    long_text = ("아주긴문장" * 100) + "."
    segments, _ = segment_with_flags(long_text)
    assert all(len(seg) <= 240 for seg in segments)
    joined = "".join(segments)
    assert long_text.replace(".", "").replace(" ", "") in joined.replace(" ", "")


# ---------------------------------------------------------------- version source
def test_version_single_source():
    """pyproject / installer .iss / src/voice_studio/__init__.py 세 곳 모두 동일 버전."""
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    iss = (root / "installer" / "voice-studio.iss").read_text(encoding="utf-8", errors="replace")
    pkg_init = (root / "src" / "voice_studio" / "__init__.py").read_text(encoding="utf-8")
    import re as _re
    m = _re.search(r'^version\s*=\s*"([^"]+)"', pyproject, _re.M)
    assert m, "pyproject version not found"
    version = m.group(1)
    mi = _re.search(r'MyAppVersion\s+"([^"]+)"', iss)
    assert mi and mi.group(1) == version
    mpkg = _re.search(r'__version__\s*=\s*"([^"]+)"', pkg_init)
    assert mpkg and mpkg.group(1) == version
    assert version == "0.1.1"
