"""P12.2 worker/dtype regression: 모델 경로 필수, dtype 정책, GPU cleanup finally."""

import sys, json, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

torch = pytest.importorskip("torch", reason="torch 미설치 환경")
from voice_studio.infra.qwen_adapter import preferred_dtype, RealQwenAdapter
from voice_studio.workers.worker_main import _require_model_path, _release_gpu


def test_dtype_policy_fp16_for_turing(monkeypatch):
    """P12.2-10: RTX 2070 SUPER(CC 7.5) 기본 dtype은 torch.float16."""
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: False)
    assert preferred_dtype("cuda:0") is torch.float16


def test_dtype_policy_bf16_when_supported(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    assert preferred_dtype("cuda:0") is torch.bfloat16


def test_dtype_policy_cpu_not_forced():
    """CPU에서는 dtype을 강제하지 않는다(P12.2-09)."""
    assert preferred_dtype("cpu") is None


def test_adapter_uses_dtype_object_not_string(monkeypatch):
    """P12.2-09: kwargs["dtype"]는 torch 객체(str 아님)로 전달된다."""
    captured = {}
    class FakeModel:
        @staticmethod
        def from_pretrained(source, **kwargs):
            captured.update(kwargs)
            return "model"
    _inject_fake_qwen_tts(monkeypatch, FakeModel)
    RealQwenAdapter(model_path="/tmp/x", device="cuda:0")
    assert isinstance(captured.get("dtype"), torch.dtype)
    assert not isinstance(captured.get("dtype"), str)


def test_adapter_flash_attention_failure_falls_back(monkeypatch):
    """P12.2-11: flash_attention 실패 시 표준 attention으로 1회 fallback."""
    calls = []
    class FakeModel:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append(kwargs.get("attn_implementation"))
            if kwargs.get("attn_implementation") == "flash_attention_2":
                raise RuntimeError("flash_attention_2 not supported on this GPU")
            return "model"
    _inject_fake_qwen_tts(monkeypatch, FakeModel)
    RealQwenAdapter(model_path="/tmp/x", device="cuda:0", flash_attention=True)
    assert calls == ["flash_attention_2", None]


def test_worker_requires_model_path(tmp_path):
    """P12.2-26: model_path 비었거나 없으면 E_MODEL_NOT_DOWNLOADED, HF 다운로드 금지."""
    import types as t
    from voice_studio.core.errors import VoiceStudioError
    job = t.SimpleNamespace(job_id="j", mode="narrate", model_path="")
    with pytest.raises(VoiceStudioError) as e:
        _require_model_path(job)
    assert e.value.code == "E_MODEL_NOT_DOWNLOADED"
    job.model_path = str(tmp_path / "nope")
    with pytest.raises(VoiceStudioError) as e:
        _require_model_path(job)
    assert e.value.code == "E_MODEL_NOT_DOWNLOADED"
    snap = tmp_path / "snap"; snap.mkdir()
    job.model_path = str(snap)
    assert _require_model_path(job) == str(snap)


def test_worker_gpu_cleanup_in_finally_structure():
    """P12.2-18: worker main이 성공/실패 무관하게 _release_gpu를 finally로 호출하는지 구조 검증."""
    src = (Path(__file__).resolve().parents[1] / "src" / "voice_studio" / "workers" /
           "worker_main.py").read_text(encoding="utf-8")
    assert "finally:\n            # 성공/실패 무관하게" in src
    assert "_release_gpu()" in src.split("finally:", 1)[1]
    assert "def _write_failure_log" in src  # P12.2-23 진단 로그 존재
    assert "worker_logs_dir" in src


def test_worker_failure_log_written_to_logs_dir(tmp_path, monkeypatch):
    """P12.2-23: 실패 시 traceback이 logs 폴더 파일에 기록된다."""
    import voice_studio.workers.worker_main as wm
    monkeypatch.setattr(wm, "worker_logs_dir", lambda: tmp_path / "logs")
    try:
        raise RuntimeError("boom")
    except RuntimeError as e:
        wm._write_failure_log(types.SimpleNamespace(job_id="j1", mode="narrate"), e)
    content = (tmp_path / "logs" / "worker-failure.log").read_text(encoding="utf-8")
    assert "job_id=j1" in content and "RuntimeError: boom" in content


def test_release_gpu_safe_without_cuda():
    """_release_gpu는 CUDA 없는 환경에서도 예외를 던지지 않는다."""
    _release_gpu()
