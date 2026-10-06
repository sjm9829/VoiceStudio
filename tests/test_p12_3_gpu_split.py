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
    _install_fake_gpu_stack(monkeypatch, cuda_available=False)
    code = _run_smoke(monkeypatch, tmp_path, ["VoiceStudio.exe", "--smoke-test"])
    assert code == 0
    log = _smoke_log(tmp_path)
    out = capsys.readouterr().out
    assert "SMOKE_OK" in log
    # CUDA/GPU 2개 항목이 GPU 없이 SKIP 처리되어 성공이다(FAIL 아님).
    assert "CUDA available: SKIP" in out
    assert "GPU: SKIP" in out
    assert ": FAIL" not in out


def test_default_smoke_fails_on_torch_import_error(tmp_path, monkeypatch):
    """frozen에서 torch/qwen import 실패는 기본 smoke에서도 FAIL이다(SKIP 아님)."""
    import sys as _sys
    monkeypatch.setattr(_sys, "frozen", True, raising=False)
    import voice_studio.heavy_smoke as hs

    def _failing_heavy(checks, require_gpu=False):
        checks.append(("Torch import", False, "ImportError: dll not found"))
        checks.append(("qwen_tts import", False, "ImportError: no module"))
        checks.append(("CUDA available", True, "SKIP (...)"))
        checks.append(("GPU", True, "SKIP (...)"))

    monkeypatch.setattr(hs, "heavy_checks", _failing_heavy)
    code = _run_smoke(monkeypatch, tmp_path, ["VoiceStudio.exe", "--smoke-test"])
    assert code == 1
    log = _smoke_log(tmp_path)
    assert "SMOKE_FAILED" in log
    assert "Torch import" in log


def test_require_gpu_fails_when_cuda_unavailable(tmp_path, monkeypatch):
    _install_fake_gpu_stack(monkeypatch, cuda_available=False)
    code = _run_smoke(monkeypatch, tmp_path, ["VoiceStudio.exe", "--smoke-test", "--require-gpu"])
    assert code == 1
    log = _smoke_log(tmp_path)
    assert "SMOKE_FAILED" in log
    assert "CUDA" in log


def test_require_gpu_succeeds_when_cuda_available(tmp_path, monkeypatch):
    _install_fake_gpu_stack(monkeypatch, cuda_available=True)
    _fake_ffmpeg(monkeypatch)
    code = _run_smoke(monkeypatch, tmp_path, ["VoiceStudio.exe", "--smoke-test", "--require-gpu"])
    assert code == 0
    assert "SMOKE_OK" in _smoke_log(tmp_path)


def test_build_script_does_not_require_check_cuda():
    bat = (ROOT / "scripts" / "build_windows.bat").read_text(encoding="utf-8")
    assert "check_cuda.py || goto :err" not in bat
    assert "check_runtime_packages.py || goto :err" in bat
    assert "--smoke-test ||" in bat  # 기본 CPU-safe smoke


def test_check_runtime_packages_never_asserts_cuda_available():
    src = (ROOT / "scripts" / "check_runtime_packages.py").read_text(encoding="utf-8")
    assert "torch.version.cuda" in src
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
