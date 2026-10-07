"""P13 stabilization self-review: 검증 코드 자체의 blocker 회귀.

- scripts/p13_runtime_e2e.py가 존재하는 실제 API만 사용하는지 정적 계약.
- scripts/validate_runtime_gpu_windows.bat가 실제 존재하고 ASCII-only인지.
- ffmpeg_adapter 오타(excxc) 부재와 정확한 예외 계약.
- worker result 이벤트 계약(protocol.result_event → output_path).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _e2e_source() -> str:
    return (ROOT / "scripts" / "p13_runtime_e2e.py").read_text(encoding="utf-8")


def test_e2e_does_not_import_production_device_from_ffmpeg_adapter():
    """production_device는 qwen_adapter에만 존재한다. ffmpeg_adapter import에 섞이면 안 된다."""
    src = _e2e_source()
    for line in src.splitlines():
        if "ffmpeg_adapter import" in line:
            assert "production_device" not in line


def test_e2e_uses_flat_probe_contract():
    """probe 계약은 flat dict(duration/format_name/...)이므로 probe["format"] 접근이 없어야 한다."""
    src = _e2e_source()
    assert 'probe["format"]' not in src
    assert "probe['format']" not in src
    assert 'probe["duration"]' in src


def test_e2e_uses_output_path_result_contract():
    """protocol.result_event는 output_path 키를 사용한다."""
    src = _e2e_source()
    assert 'res2["output_path"]' in src
    assert 'res2["path"]' not in src


def test_e2e_does_not_call_repository_load_prompt_spec():
    """load_prompt_spec는 ProfileService 메서드다. ProfileRepository 직접 호출 금지."""
    src = _e2e_source()
    assert "repo.load_prompt_spec" not in src
    assert "ProfileService" in src and "load_prompt_spec" in src


def test_e2e_uses_model_manager_model_path_first():
    """모델 경로는 ModelManager().model_path() 계약을 우선 사용해야 한다."""
    src = _e2e_source()
    assert "ModelManager().model_path()" in src
    assert "huggingface" in src  # HF 캐시 fallback은 유지


def test_e2e_profile_isolation_is_temp_and_unique():
    """E2E 프로필은 temp 데이터 루트로 격리되고 이름에 unique suffix를 쓴다."""
    src = _e2e_source()
    assert "mkdtemp" in src
    assert "VOICE_STUDIO_DATA_DIR" in src
    assert "LOCALAPPDATA" in src
    assert 'f"P13 E2E {uuidlib.uuid4().hex[:8]}"' in src


def test_e2e_supports_frozen_and_source_worker_modes():
    src = _e2e_source()
    assert "--app-exe" in src
    assert '"--worker"' in src or "'--worker'" in src
    assert "-m" in src and "voice_studio.main" in src


def test_e2e_includes_stt_runtime_step():
    src = _e2e_source()
    assert "FasterWhisperTranscriber" in src
    assert "--skip-stt" in src


def test_e2e_module_import_and_help_succeed():
    import importlib.util
    spec = importlib.util.spec_from_file_location("p13_runtime_e2e", ROOT / "scripts" / "p13_runtime_e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import subprocess as sp
    r = sp.run([sys.executable, str(ROOT / "scripts" / "p13_runtime_e2e.py"), "--help"],
               capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr
    assert "--app-exe" in r.stdout and "--skip-stt" in r.stdout


def test_validate_runtime_gpu_windows_bat_exists_ascii_only():
    bat = ROOT / "scripts" / "validate_runtime_gpu_windows.bat"
    assert bat.is_file(), "validate_runtime_gpu_windows.bat가 repository에 존재해야 한다"
    text = bat.read_text(encoding="ascii")
    assert text.isascii()
    assert "check_cuda.py" in text
    assert "tests\\test_gpu_real.py -m gpu" in text or "test_gpu_real.py -m gpu" in text
    assert "p13_runtime_e2e.py" in text
    # 사용자 audio/ref-text는 인자로 전달(hardcode 금지)
    assert "%*" in text


def test_all_bat_scripts_ascii_only():
    for bat in (ROOT / "scripts").glob("*.bat"):
        bat.read_text(encoding="ascii")  # non-ascii면 UnicodeDecodeError로 실패


def test_ffmpeg_adapter_no_excxc_typo():
    src = (ROOT / "src" / "voice_studio" / "infra" / "ffmpeg_adapter.py").read_text(encoding="utf-8")
    assert "excxc" not in src
    assert "from exc" in src


def test_worker_result_event_contract_is_output_path():
    from voice_studio.workers.protocol import result_event
    ev = result_event("out/x.mp3", job_id="j", result_type="audio")
    assert ev["output_path"] == "out/x.mp3"
    assert "path" not in ev


def test_jsonl_buffer_lives_in_linebuffer_module():
    """JsonlBuffer는 workers.linebuffer에 있다(protocol에 두면 안 된다)."""
    from voice_studio.workers.linebuffer import JsonlBuffer
    import voice_studio.workers.protocol as protocol
    assert not hasattr(protocol, "JsonlBuffer") or protocol.JsonlBuffer is JsonlBuffer
