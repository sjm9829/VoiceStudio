"""P12.2-21 smoke-test 모드 / P12.2-05 frozen ffmpeg 탐색 / build 스크립트 계약."""

import sys, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

def test_smoke_test_mode_writes_log(tmp_path, monkeypatch):
    from voice_studio import main as m
    monkeypatch.setattr("voice_studio.core.paths.ensure_app_dirs", lambda: {})
    monkeypatch.setattr("voice_studio.core.paths.logs_dir", lambda: tmp_path / "logs")
    monkeypatch.setattr(m, "run_smoke_test", m.run_smoke_test)  # 실제 함수 유지
    code = m.run_smoke_test()
    log = (tmp_path / "logs" / "smoke-test.log").read_text(encoding="utf-8")
    assert "SMOKE_OK" in log
    assert code == 0


def test_smoke_test_mode_runs_before_qt(tmp_path, monkeypatch):
    """--smoke-test는 QApplication 생성 전에 끝나야 한다(앱 실행 없음)."""
    src = (Path(__file__).resolve().parents[1] / "src" / "voice_studio" / "main.py").read_text(encoding="utf-8")
    assert src.index('if "--smoke-test" in args:') < src.index("QApplication(args)")


def test_frozen_ffmpeg_search_covers_internal_bin(monkeypatch):
    """P12.2-05: frozen에서 exe 옆 bin/과 _internal/bin/ 모두 탐색한다."""
    from voice_studio.infra import ffmpeg_adapter as fa
    root = Path(__file__).resolve().parents[1]
    fake_exe = tmp_exedir = None
    # sys.frozen + sys.executable 스텁: exe 옆 bin에 ffmpeg만 있을 때 찾는지
    d = root / ".tmp_test_bin"
    (d / "_internal" / "bin").mkdir(parents=True, exist_ok=True)
    (d / "_internal" / "bin" / "ffmpeg.exe").write_bytes(b"x")
    import sys as _sys
    monkeypatch.setattr(_sys, "frozen", True, raising=False)
    monkeypatch.setattr(_sys, "executable", str(d / "VoiceStudio.exe"))
    try:
        assert fa._resolve_binary("ffmpeg", None) == str(d / "_internal" / "bin" / "ffmpeg.exe")
    finally:
        monkeypatch.setattr(_sys, "frozen", False, raising=False)
        import shutil as _sh; _sh.rmtree(d, ignore_errors=True)


def test_spec_bundles_ffmpeg_and_notice():
    spec = (Path(__file__).resolve().parents[1] / "packaging" / "VoiceStudio.spec").read_text(encoding="utf-8")
    assert 'collect_data_files("qwen_tts"' in spec           # P12.2-07
    assert 'collect_all("ctranslate2")' in spec and 'collect_all("tokenizers")' in spec  # P12.2-08
    assert '"ffmpeg.exe"' in spec and '"ffprobe.exe"' in spec  # P12.2-05
    assert "FFMPEG_NOTICE.txt" in spec                        # P12.2-06


def test_build_script_cuda_tag_parameterized():
    """P12.2-12: CUDA_TAG 환경변수로 조합 교체 가능 + 실패 시 빌드 중단."""
    bat = (Path(__file__).resolve().parents[1] / "scripts" / "build_windows.bat").read_text(encoding="utf-8")
    assert "CUDA_TAG" in bat
    assert "check_cuda.py || goto :err" in bat
    assert "TORCH_BEFORE_QWEN_TTS" in bat and "TORCH_AFTER_QWEN_TTS" in bat
    assert "pip check" in bat


def test_check_cuda_validates_cuda_and_qwen_tts():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "check_cuda.py").read_text(encoding="utf-8")
    assert "is_bf16_supported" in src            # P12.2-13/GPU 정정
    assert "from qwen_tts import Qwen3TTSModel" in src
    assert "CHECK_CUDA_FAILED" in src            # 실패 시 빌드 중단 가능한 exit code
