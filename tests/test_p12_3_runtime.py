"""P12.3 runtime regression: model_path 강제, HF fallback 차단, 오류 생성자 계약."""

import ast, sys, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from voice_studio.core.errors import ModelNotDownloadedError, VoiceStudioError
from voice_studio.workers.worker_main import _require_model_path


def _job(**kw):
    base = dict(job_id="j1", mode="narrate", profile_dir=str(Path.cwd()), model_path="")
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_register_empty_model_path_raises_not_downloaded(tmp_path, monkeypatch, capsys):
    """P12.3-23: register model_path="" → ModelNotDownloadedError, qwen adapter 생성 안 됨."""
    import voice_studio.infra.qwen_adapter as qa
    constructed = []
    class _Sentinel:
        def __init__(self, *a, **k):
            constructed.append(True)
    monkeypatch.setattr(qa, "RealQwenAdapter", _Sentinel, raising=False)
    import voice_studio.workers.worker_main as wm
    monkeypatch.setattr(wm, "RealQwenAdapter", _Sentinel, raising=False)
    with pytest.raises(ModelNotDownloadedError):
        wm.run_register(_job(mode="register"))
    assert not constructed


def test_narrate_empty_model_path_raises_not_downloaded(tmp_path, monkeypatch):
    """P12.3-23: narrate model_path="" → ModelNotDownloadedError, HF fallback 없음."""
    import voice_studio.infra.qwen_adapter as qa
    constructed = []
    class _Sentinel:
        def __init__(self, *a, **k):
            constructed.append(True)
    monkeypatch.setattr(qa, "RealQwenAdapter", _Sentinel, raising=False)
    import voice_studio.workers.worker_main as wm
    monkeypatch.setattr(wm, "RealQwenAdapter", _Sentinel, raising=False)
    with pytest.raises(ModelNotDownloadedError):
        wm.run_narrate(_job(mode="narrate"))
    assert not constructed  # HF repo ID fallback으로 어댑터가 만들어지지 않았다


def test_missing_model_dir_raises_for_both_modes(tmp_path):
    from voice_studio.workers.worker_main import _require_model_path
    for mp in ("", "  ", str(tmp_path / "missing-model")):
        with pytest.raises(ModelNotDownloadedError):
            _require_model_path(_job(model_path=mp))


def test_worker_error_event_uses_user_message_not_detail():
    """worker 실패 이벤트는 UI에 user_message를 주고 detail은 로그로 남긴다."""
    src = (Path(__file__).resolve().parents[1] / "src" / "voice_studio" / "workers" /
           "worker_main.py").read_text(encoding="utf-8")
    assert "emit(error_event(e.code, e.user_message, detail=e.detail))" in src


def test_qwen_adapter_blocks_repo_fallback_by_default():
    """P12.3-10: production 기본은 HF repo ID fallback 금지(qwen_tts 미설치 환경에서도 검증)."""
    import voice_studio.infra.qwen_adapter as qa
    with pytest.raises(ModelNotDownloadedError):
        qa.RealQwenAdapter(model_path=None)
    src = (Path(__file__).resolve().parents[1] / "src" / "voice_studio" /
           "infra" / "qwen_adapter.py").read_text(encoding="utf-8")
    assert "allow_repo_fallback: bool = False" in src
    # 명시적 허용 시에만 source가 repo ID가 될 수 있다(구조 확인).
    assert "source = model_path or model_id" in src


def test_no_two_positional_VoiceStudioError_calls():
    """P12.3-24: VoiceStudioError(...) positional 인수 2개 이상 사용처가 없다."""
    bad = []
    for p in (Path(__file__).resolve().parents[1] / "src" / "voice_studio").rglob("*.py"):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == "VoiceStudioError":
                if len(node.args) >= 2:
                    bad.append(str(p))
    assert bad == []
