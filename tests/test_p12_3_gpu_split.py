"""GPU/빌드 역할 분리 regression(P12.3 Final Hotfix 후속).

기본 smoke는 GPU 없는 환경에서도 성공해야 하고, --require-gpu에서만 CUDA/GPU가
mandatory이다. build_windows.bat는 check_cuda.py를 필수 호출하지 않으며, GPU
validation script는 check_cuda.py와 --require-gpu 스모크를 호출한다.
"""

import sys, types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]


def _install_fake_gpu_stack(monkeypatch, cuda_available: bool):
    """torch/torchaudio/qwen_tts를 가짜 모듈로 대체해 heavy_checks를 구동한다."""
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "2.4.0"
    fake_torch.version = types.SimpleNamespace(cuda="12.6")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        get_device_name=lambda i: "NVIDIA GeForce RTX 2070 SUPER",
        get_device_capability=lambda i: (7, 5),
        is_bf16_supported=lambda: False,
    )
    fake_audio = types.ModuleType("torchaudio")
    fake_audio.__version__ = "2.4.0"
    fake_qwen = types.ModuleType("qwen_tts")
    fake_qwen.Qwen3TTSModel = type("Qwen3TTSModel", (), {})
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torchaudio", fake_audio)
    monkeypatch.setitem(sys.modules, "qwen_tts", fake_qwen)


def _fake_ffmpeg(monkeypatch):
    """frozen/require-gpu에서 FFmpeg 실패를 피하기 위한 어댑터 스텁."""
    from voice_studio.infra import ffmpeg_adapter as fa

    class _FakeAdapter:
        ffmpeg = "ffmpeg.exe"
        ffprobe = "ffprobe.exe"
        def probe_version(self):
            return "fake ffmpeg"

    monkeypatch.setattr(fa, "RealFfmpegAdapter", _FakeAdapter)


def _run_smoke(monkeypatch, tmp_path, argv):
    from voice_studio import main as m
    monkeypatch.setattr("voice_studio.core.paths.ensure_app_dirs", lambda: {})
    monkeypatch.setattr("voice_studio.core.paths.logs_dir", lambda: tmp_path / "logs")
    return m.run_smoke_test(argv)


def _smoke_log(tmp_path):
    return (tmp_path / "logs" / "smoke-test.log").read_text(encoding="utf-8")


def test_default_smoke_succeeds_without_gpu(tmp_path, monkeypatch, capsys):
    """GGUF packaging smoke does not require torch or an attached GPU."""
    import voice_studio.heavy_smoke as hs
    monkeypatch.setattr(hs, "_check_gguf_runtime", lambda: None)
    monkeypatch.setattr(hs, "_check_llama_engine", lambda: None)
    code = _run_smoke(monkeypatch, tmp_path, ["VoiceStudio.exe", "--smoke-test"])
    assert code == 0
    assert "SMOKE_OK" in _smoke_log(tmp_path)
    out = capsys.readouterr().out
    assert "GGUF runtime imports: OK" in out
    assert "CUDA available: SKIP" in out
    assert "GPU: SKIP" in out


def test_frozen_smoke_succeeds_without_gpu(tmp_path, monkeypatch, capsys):
    import voice_studio.heavy_smoke as hs
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    _fake_ffmpeg(monkeypatch)
    monkeypatch.setattr(hs, "_check_gguf_runtime", lambda: None)
    monkeypatch.setattr(hs, "_check_llama_engine", lambda: None)
    assert _run_smoke(monkeypatch, tmp_path, ["VoiceStudio.exe", "--smoke-test"]) == 0
    assert "CUDA available: SKIP" in capsys.readouterr().out


def test_frozen_smoke_fails_when_engine_missing(tmp_path, monkeypatch):
    import voice_studio.heavy_smoke as hs
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    _fake_ffmpeg(monkeypatch)
    monkeypatch.setattr(hs, "_check_gguf_runtime", lambda: None)
    def missing():
        raise FileNotFoundError("llama-tts.exe")
    monkeypatch.setattr(hs, "_check_llama_engine", missing)
    assert _run_smoke(monkeypatch, tmp_path, ["VoiceStudio.exe", "--smoke-test"]) == 1
    assert "SMOKE_FAILED" in _smoke_log(tmp_path)


def test_strict_gpu_smoke_fails_if_driver_missing(tmp_path, monkeypatch):
    import voice_studio.heavy_smoke as hs
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    _fake_ffmpeg(monkeypatch)
    monkeypatch.setattr(hs, "_check_gguf_runtime", lambda: None)
    monkeypatch.setattr(hs, "_check_llama_engine", lambda: None)
    def no_cuda():
        raise RuntimeError("CUDA driver missing")
    monkeypatch.setattr(hs, "_cuda_device_count", no_cuda)
    assert _run_smoke(monkeypatch, tmp_path,
                      ["VoiceStudio.exe", "--smoke-test", "--require-gpu"]) == 1


def test_strict_gpu_smoke_succeeds_with_driver(tmp_path, monkeypatch):
    import voice_studio.heavy_smoke as hs
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    _fake_ffmpeg(monkeypatch)
    monkeypatch.setattr(hs, "_check_gguf_runtime", lambda: None)
    monkeypatch.setattr(hs, "_check_llama_engine", lambda: None)
    monkeypatch.setattr(hs, "_cuda_device_count", lambda: 1)
    assert _run_smoke(monkeypatch, tmp_path,
                      ["VoiceStudio.exe", "--smoke-test", "--require-gpu"]) == 0


def test_build_script_does_not_require_check_cuda():
    bat = (ROOT / "scripts" / "build_windows.bat").read_text(encoding="utf-8")
    assert "check_cuda.py || goto :err" not in bat
    assert "check_runtime_packages.py || goto :err" in bat
    assert "--smoke-test ||" in bat  # 기본 CPU-safe smoke


def test_check_runtime_packages_never_asserts_cuda_available():
    src = (ROOT / "scripts" / "check_runtime_packages.py").read_text(encoding="utf-8")
    assert "torch.version.cuda" not in src
    # GPU 존재 검사 코드가 없어야 한다(주석 언급 제외: 실제 호출 라인 기준).
    for line in src.splitlines():
        code = line.split("#")[0]
        assert "cuda.is_available" not in code, line
        assert "nvidia-smi" not in code, line


def test_validate_gpu_script_calls_check_cuda_and_require_gpu():
    src = (ROOT / "scripts" / "validate_gpu_windows.bat").read_text(encoding="utf-8")
    assert "check_cuda.py" in src
    assert "--smoke-test --require-gpu" in src
    assert "GPU_VALIDATION_OK" in src


def test_heavy_checks_require_gpu_signature():
    from voice_studio import heavy_smoke as hs
    import inspect
    sig = inspect.signature(hs.heavy_checks)
    assert "require_gpu" in sig.parameters
