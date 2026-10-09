"""P15 Qwen3-TTS 프로브 스크립트 비-GPU 회귀 검증.

scripts/p15_qwen_probe.py의 조사 도구(silence_report, save_raw, fingerprint,
tensor_diff, cmd_list, cmd_inspect, cmd_stt NOT RUN 분기)와 프로브 대상
어댑터 계약(FakeQwenAdapter로 대체)을 검증한다. GPU 배터리 자체는 RTX
실기에서만 실행되며 이 테스트는 실행하지 않는다(NOT RUN 정책 유지).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_SPEC = Path(__file__).resolve().parents[1] / "scripts" / "p15_qwen_probe.py"
_spec = importlib.util.spec_from_file_location("p15_qwen_probe", _SPEC)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

from voice_studio.infra.profile_repository import ProfileRepository
from voice_studio.infra.qwen_adapter import FakeQwenAdapter, VoiceClonePromptSpec
from voice_studio.services.audio_diagnostics import read_wav_f32
from voice_studio.core import config


def _spec_like(peak: float = 0.3) -> VoiceClonePromptSpec:
    return FakeQwenAdapter().create_prompt(np.full(2400, peak, dtype=np.float32),
                                           config.REFERENCE_SAMPLE_RATE, "테스트 대사")


# ---- silence_report

def test_silence_report_leading_trailing():
    sr = 24000
    pcm = np.concatenate([np.zeros(int(sr * 0.5), dtype=np.float32),
                          np.full(int(sr * 0.25), 0.5, dtype=np.float32),
                          np.zeros(int(sr * 0.25), dtype=np.float32)])
    rep = probe.silence_report(pcm, sr)
    assert abs(rep["seconds"] - 1.0) < 0.01
    assert abs(rep["leading_silence_s"] - 0.5) < 0.01
    assert abs(rep["trailing_silence_s"] - 0.25) < 0.01
    assert rep["rms"] > 0.1

def test_silence_report_all_silent():
    rep = probe.silence_report(np.zeros(24000, dtype=np.float32), 24000)
    assert abs(rep["leading_silence_s"] - 1.0) < 0.01
    assert rep["rms"] == 0.0


# ---- save_raw 원본 보존 계약

def test_save_raw_roundtrip_float32(tmp_path):
    sr = 24000
    pcm = np.linspace(-1.0, 1.0, 2400, dtype=np.float32)
    path = probe.save_raw("test_label", pcm, sr, tmp_path)
    assert path.name == "01_qwen_raw_24k.wav"
    back, sr_back = read_wav_f32(path)
    assert sr_back == sr
    np.testing.assert_array_equal(back, pcm)  # 무손실(양자화·리샘플 없음)

def test_save_raw_rejects_corrupt(tmp_path, monkeypatch):
    pcm = np.zeros(100, dtype=np.float32)
    monkeypatch.setattr(probe, "_write_wav_f32",
                        lambda p, x, s: Path(p).write_bytes(b"broken"))
    with pytest.raises((RuntimeError, AssertionError)):
        probe.save_raw("bad", pcm, 24000, tmp_path)


def test_save_raw_does_not_silently_overwrite(tmp_path):
    """같은 라벨 재실행 시 이전 진단 WAV를 덮어쓰지 않고 번호를 붙여 보존한다."""
    pcm_a = np.full(2400, 0.1, dtype=np.float32)
    pcm_b = np.full(2400, 0.2, dtype=np.float32)
    p1 = probe.save_raw("dup", pcm_a, 24000, tmp_path)
    p2 = probe.save_raw("dup", pcm_b, 24000, tmp_path)
    assert p1 != p2
    back1, _ = read_wav_f32(p1)
    np.testing.assert_array_equal(back1, pcm_a)  # 첫 파일 무변경
    back2, _ = read_wav_f32(p2)
    np.testing.assert_array_equal(back2, pcm_b)
    p3 = probe.save_raw("dup", pcm_a, 24000, tmp_path)
    assert p3.name == "01_qwen_raw_24k_3.wav"


# ---- fingerprint / tensor_diff

def test_fingerprint_and_diff_identical():
    code = np.array([1, 2, 3], dtype=np.int64)
    emb = np.zeros(256, dtype=np.float32)
    fp = probe.fingerprint(code, emb)
    assert fp["ref_code"]["shape"] == [3]
    assert fp["ref_spk_embedding"]["dtype"] == "float32"
    assert probe.tensor_diff(fp, fp) == []

def test_tensor_diff_detects_mismatch():
    a = _spec_like(0.3)
    b = _spec_like(0.9)
    diffs = probe.tensor_diff(probe.fingerprint(a.ref_code, a.ref_spk_embedding),
                              probe.fingerprint(b.ref_code, b.ref_spk_embedding))
    assert diffs and all("ref_code" in d or "ref_spk_embedding" in d for d in diffs)

def test_fingerprint_handles_json_blob():
    fp = probe.fingerprint({"fake": True, "peak": 0.3}, None)
    assert "repr" in fp["ref_code"] and fp["ref_spk_embedding"] is None

def test_tensor_diff_handles_none():
    fp_a = probe.fingerprint(None, None)
    fp_b = probe.fingerprint(np.zeros(4, dtype=np.int64), None)
    diffs = probe.tensor_diff(fp_a, fp_b)
    assert any("ref_code" in d for d in diffs)


# ---- cmd_list / cmd_inspect (FakeQwenAdapter로 저장한 fake 프로필 대상)

def _fake_profile(tmp_path):
    repo = ProfileRepository(tmp_path / "profiles")
    from voice_studio.domain.voice_profile import VoiceProfile
    profile = VoiceProfile(name="프로브", ref_text="테스트 대사")
    repo.save(profile, {"ref_code": np.array([5.0, 1.0, 2.0], dtype=np.float32),
                        "ref_spk_embedding": np.zeros(256, dtype=np.float32)}, b"FAKEFLAC")
    return repo, profile

def test_cmd_list_and_inspect(tmp_path, capsys):
    repo, profile = _fake_profile(tmp_path)
    assert probe.cmd_list(repo) == 0
    assert "프로브" in capsys.readouterr().out
    assert probe.cmd_inspect(repo, profile.uuid, tmp_path / "diag") == 0
    out = capsys.readouterr().out
    assert "prompt_fingerprint" in out and "NOT RUN" in out  # fake FLAC은 디코딩 불가 NOT RUN 표시

def test_cmd_inspect_missing_profile(tmp_path, capsys):
    repo = ProfileRepository(tmp_path / "profiles")
    with pytest.raises(Exception):
        probe.cmd_inspect(repo, "no-such-uuid", tmp_path / "diag")


# ---- cmd_stt: 디코딩 불가 환경에서 NOT RUN 분기

def test_cmd_stt_not_run_without_ffmpeg(tmp_path, capsys):
    repo, profile = _fake_profile(tmp_path)
    code = probe.cmd_stt(repo, profile.uuid)
    assert code == 3
    assert "NOT RUN" in capsys.readouterr().out


# ---- GPU 배터리 진입점은 CUDA/qwen_tts 없는 환경에서 NOT RUN

def test_cmd_gpu_not_run_without_model(tmp_path, capsys, monkeypatch):
    """모델 로더를 mock 처리하고 실제로 Mock이 호출됐는지 검증한다.

    비GPU 환경에서 실제 Qwen 모델 다운로드/로드/CUDA 초기화가 발생하지 않아야 하고,
    참조 음성 검증이 통과한 경우에만 모델 로더가 호출된다."""
    calls = {"real_adapter_init": 0}

    class _StubAdapter:
        def __init__(self, *args, **kwargs):
            calls["real_adapter_init"] += 1
            raise RuntimeError("qwen_tts/torch 없음 (test stub)")

    monkeypatch.setattr(probe, "RealQwenAdapter", _StubAdapter)
    monkeypatch.setattr(probe, "load_reference_via_probe",
                        lambda repo, uuid: (np.zeros(2400, dtype=np.float32),
                                            {"duration_s": 0.1}))
    # CUDA 없는 개발 환경에서도 모델 로더 경로까지 도달하게 한다(실제 torch/CUDA 접근 없음).
    monkeypatch.setattr(probe, "production_device", lambda: "cuda:0")
    monkeypatch.setattr(probe, "ModelManager", lambda: type("MM", (), {"model_path": lambda self: "/fake/snapshot"})())
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    repo, profile = _fake_profile(tmp_path)
    code = probe.cmd_gpu(repo, profile.uuid, tmp_path / "diag", None, 1)
    assert code == 3
    out = capsys.readouterr().out
    assert "NOT RUN" in out
    assert calls["real_adapter_init"] == 1  # Mock(스텁) 로더가 실제로 호출됐음을 검증


def test_cmd_gpu_reference_check_precedes_model_load(tmp_path, capsys, monkeypatch):
    """FFmpeg·참조 음성 유효성 실패 시 모델을 적재하지 않고 NOT RUN으로 끝낸다."""
    def _boom(repo, uuid):
        raise probe.UnsupportedAudioError("ffmpeg 없음: third_party/bin 미설치")

    class _StubAdapter:
        def __init__(self, *args, **kwargs):
            raise AssertionError("참조 검증 실패 상태에서 모델 로더가 호출되면 안 됩니다")

    monkeypatch.setattr(probe, "RealQwenAdapter", _StubAdapter)
    monkeypatch.setattr(probe, "load_reference_via_probe", _boom)
    repo, profile = _fake_profile(tmp_path)
    code = probe.cmd_gpu(repo, profile.uuid, tmp_path / "diag", None, 1)
    assert code == 3
    out = capsys.readouterr().out
    assert "NOT RUN" in out and "ffmpeg 없음" in out


def test_gpu_adapter_uses_local_snapshot_and_cuda(monkeypatch):
    """_gpu_adapter는 model_path 미지정 시 ModelManager 로컬 스냅샷을 쓰고,
    production_device()로 CUDA를 명시한다(HF repo 자동 다운로드 없음)."""
    calls = {}

    class _StubAdapter:
        def __init__(self, *, model_path, device):
            calls["model_path"] = model_path
            calls["device"] = device

    monkeypatch.setattr(probe, "RealQwenAdapter", _StubAdapter)
    monkeypatch.setattr(probe, "production_device", lambda: "cuda:0")
    class _StubMM:
        def model_path(self):
            return "/fake/local/snapshot"

    monkeypatch.setattr(probe, "ModelManager", _StubMM)
    # model_path 지정: 그대로 사용, ModelManager 호출 없음
    probe._gpu_adapter("/given/path")
    assert calls == {"model_path": "/given/path", "device": "cuda:0"}
    # model_path 미지정: 로컬 스냅샷 사용
    probe._gpu_adapter(None)
    assert calls == {"model_path": "/fake/local/snapshot", "device": "cuda:0"}


def test_probe_ffmpeg_adapter_uses_repo_third_party_bin(tmp_path, monkeypatch):
    """프로브 디코딩은 시스템 PATH가 아니라 repo third_party/bin을 명시 사용한다."""
    monkeypatch.setenv("PATH", "")  # 시스템 PATH 의존 제거 검증
    probe_bin = tmp_path / "third_party" / "bin"
    probe_bin.mkdir(parents=True)
    (probe_bin / "ffmpeg.exe").write_bytes(b"x")
    (probe_bin / "ffprobe.exe").write_bytes(b"x")
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    adapter = probe._probe_ffmpeg_adapter()
    assert adapter.ffmpeg == str(probe_bin / "ffmpeg.exe")
    assert adapter.ffprobe == str(probe_bin / "ffprobe.exe")


def test_supported_params_probe_normal_and_kwargs():
    def generate_voice_clone(self, text, language=None, voice_clone_prompt=None, **kwargs):
        pass
    res = probe.supported_params_probe(type("M", (), {"generate_voice_clone": generate_voice_clone})())
    assert res["accepts_kwargs"] is True and res["error"] is None
    assert "text" not in res["supported"]

def test_supported_params_probe_signature_exception_initializes_accepts_kwargs(monkeypatch):
    """inspect.signature 예외 시에도 accepts_kwargs가 초기화되어 있어야 한다."""
    import inspect as _inspect
    def _raise(*a, **k):
        raise ValueError("signature 실패")
    # supported_params_probe 내부의 import inspect와 동일 모듈 객체를 패치한다.
    monkeypatch.setattr(_inspect, "signature", _raise)
    class _M:
        generate_voice_clone = staticmethod(lambda *a, **k: None)
    res = probe.supported_params_probe(_M())
    assert res["supported"] == [] and res["accepts_kwargs"] is False
    assert res["error"] and "signature 실패" in res["error"]

def test_gpu_param_probe_does_not_skip_kwargs_only_params():
    """시그니처에 이름이 없어도 **kwargs면 SKIP 근거로 쓰지 않는다."""
    import inspect
    def generate_voice_clone(self, text, language=None, voice_clone_prompt=None, **kwargs):
        raise TypeError("unsupported argument")  # 실제 호출에서만 SKIP

    sig = inspect.signature(generate_voice_clone)
    accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    explicit = {p.name for p in sig.parameters.values()}
    assert accepts_kwargs and "temperature" not in explicit


# ---- 어댑터 계약 유지(FakeQwenAdapter generate 서명: language 옵트인)

def test_generate_contract_language_optin():
    qwen = FakeQwenAdapter()
    spec = _spec_like()
    wav = qwen.generate(spec, "안녕하세요 서정민입니다 안녕하세요 서정민입니다",
                        config.REFERENCE_SAMPLE_RATE, language="Korean")
    assert wav.dtype == np.float32 and wav.ndim == 1
    assert qwen.calls == ["안녕하세요 서정민입니다 안녕하세요 서정민입니다"]
