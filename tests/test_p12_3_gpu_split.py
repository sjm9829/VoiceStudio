"""GPU/빌드 역할 분리 regression(P12.3 Final Hotfix 후속, P18 GGUF 계약).

기본 smoke는 GPU 없는 환경에서도 성공해야 하고, --require-gpu에서만 CUDA/GPU가
mandatory이다. heavy 검사는 GGUF 런타임 계약(llama-tts/CUDA DLL/nvidia-smi)으로
동작하며 torch/torchaudio/qwen_tts는 더 이상 검사 대상이 아니다.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]


def _fake_nvidia_smi(monkeypatch, available: bool = True, name: str = "NVIDIA GeForce RTX 2070 SUPER"):
    """heavy_smoke의 nvidia-smi 조회를 가짜로 대체한다(PyTorch 없이 CUDA 검증)."""
    from voice_studio import heavy_smoke as hs

    def _run(*args, **kwargs):
        out = types.SimpleNamespace(
            returncode=0 if available else 1,
            stdout=name if available else "",
            stderr="" if available else "no devices",
        )
        return out

    monkeypatch.setattr(hs.shutil, "which", lambda _: "/usr/bin/nvidia-smi" if available else None)
    monkeypatch.setattr(hs.subprocess, "run", _run)


def _fake_engine(monkeypatch, tmp_path: Path, with_dll: bool = True):
    """gguf_engine_dir를 임시 엔진 디렉터로 대체한다."""
    from voice_studio.core import paths as paths_mod
    engine = tmp_path / "engine"
    engine.mkdir(parents=True, exist_ok=True)
    (engine / ("llama-tts.exe" if sys.platform == "win32" else "llama-tts")).write_bytes(b"MZ")
    if with_dll and sys.platform == "win32":
        (engine / "cudart64_12.dll").write_bytes(b"x")
    monkeypatch.setattr(paths_mod, "gguf_engine_dir", lambda: engine)
    return engine


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
    _fake_nvidia_smi(monkeypatch, available=False)
    code = _run_smoke(monkeypatch, tmp_path, ["VoiceStudio.exe", "--smoke-test"])
    assert code == 0
    log = _smoke_log(tmp_path)
    out = capsys.readouterr().out
    assert "SMOKE_OK" in log
    # CUDA/GPU 2개 항목이 GPU 없이 SKIP 처리되어 성공이다(FAIL 아님).
    assert "CUDA available: SKIP" in out
    assert "GPU: SKIP" in out
    assert ": FAIL" not in out


def test_default_smoke_fails_on_gguf_import_error(tmp_path, monkeypatch):
    """frozen에서 GGUF 어댑터 import 실패는 기본 smoke에서도 FAIL이다(SKIP 아님)."""
    import sys as _sys
    monkeypatch.setattr(_sys, "frozen", True, raising=False)
    import voice_studio.heavy_smoke as hs

    def _failing_heavy(checks, require_gpu=False):
        checks.append(("GGUF runtime import", False, "ImportError: dll not found"))
        checks.append(("CUDA available", True, "SKIP (...)"))
        checks.append(("GPU", True, "SKIP (...)"))

    monkeypatch.setattr(hs, "heavy_checks", _failing_heavy)
    code = _run_smoke(monkeypatch, tmp_path, ["VoiceStudio.exe", "--smoke-test"])
    assert code == 1
    log = _smoke_log(tmp_path)
    assert "SMOKE_FAILED" in log
    assert "GGUF runtime import" in log


def test_frozen_smoke_requires_bundled_engine(tmp_path, monkeypatch, capsys):
    """frozen에서 llama-tts 실행 파일/DLL 누락은 FAIL이어야 한다(설치본 무결성)."""
    import sys as _sys
    monkeypatch.setattr(_sys, "frozen", True, raising=False)
    _fake_ffmpeg(monkeypatch)
    _fake_nvidia_smi(monkeypatch, available=False)
    # 엔진 디렉터 자체를 비워 frozen 무결성 FAIL을 유도한다.
    engine = tmp_path / "empty-engine"
    engine.mkdir()
    from voice_studio.core import paths as paths_mod
    monkeypatch.setattr("voice_studio.core.paths.gguf_engine_dir", lambda: engine)
    code = _run_smoke(monkeypatch, tmp_path, ["VoiceStudio.exe", "--smoke-test"])
    assert code == 1
    out = capsys.readouterr().out
    assert "GGUF llama-tts engine: FAIL" in out
    assert "SMOKE_FAILED" in _smoke_log(tmp_path)


def test_frozen_smoke_succeeds_with_engine_and_cuda_dll(tmp_path, monkeypatch, capsys):
    import sys as _sys
    if sys.platform != "win32":
        _fake_engine(monkeypatch, tmp_path)
    else:
        _fake_engine(monkeypatch, tmp_path, with_dll=True)
    _fake_ffmpeg(monkeypatch)
    _fake_nvidia_smi(monkeypatch, available=False)
    monkeypatch.setattr(_sys, "frozen", True, raising=False)
    code = _run_smoke(monkeypatch, tmp_path, ["VoiceStudio.exe", "--smoke-test"])
    assert code == 0, capsys.readouterr().out
    out = capsys.readouterr().out
    assert "GGUF llama-tts engine: OK" in out
    assert "CUDA available: SKIP (frozen)" in out
    assert "SMOKE_OK" in _smoke_log(tmp_path)


def test_frozen_smoke_fails_without_cuda_dll_on_windows(tmp_path, monkeypatch, capsys):
    """Windows frozen에서 llama-tts는 있어도 CUDA DLL이 없으면 FAIL이다."""
    import sys as _sys
    if sys.platform != "win32":
        return  # 비Windows에서는 DLL 검사 자체가 없다(정책: Windows 번들 계약).
    monkeypatch.setattr(_sys, "frozen", True, raising=False)
    _fake_engine(monkeypatch, tmp_path, with_dll=False)
    _fake_ffmpeg(monkeypatch)
    _fake_nvidia_smi(monkeypatch, available=False)
    code = _run_smoke(monkeypatch, tmp_path, ["VoiceStudio.exe", "--smoke-test"])
    assert code == 1
    assert "CUDA 런타임 DLL 없음" in capsys.readouterr().out


def test_frozen_smoke_succeeds_without_gpu(tmp_path, monkeypatch, capsys):
    """sys.frozen=True + GPU 없음 + 기본 smoke는 SKIP 2개로 성공이어야 한다(P13 후속 회귀)."""
    import sys as _sys
    monkeypatch.setattr(_sys, "frozen", True, raising=False)
    _fake_ffmpeg(monkeypatch)
    _fake_nvidia_smi(monkeypatch, available=False)
    _fake_engine(monkeypatch, tmp_path)
    code = _run_smoke(monkeypatch, tmp_path, ["VoiceStudio.exe", "--smoke-test"])
    assert code == 0
    log = _smoke_log(tmp_path)
    out = capsys.readouterr().out
    assert "SMOKE_OK" in log
    assert "CUDA available: SKIP (frozen)" in out
    assert "GPU: SKIP (frozen)" in out
    assert ": FAIL" not in out


def test_frozen_smoke_require_gpu_still_fails_without_gpu(tmp_path, monkeypatch, capsys):
    """frozen + --require-gpu + GPU 없음은 여전히 FAIL(exit 1)이어야 한다."""
    import sys as _sys
    monkeypatch.setattr(_sys, "frozen", True, raising=False)
    _fake_ffmpeg(monkeypatch)
    _fake_nvidia_smi(monkeypatch, available=False)
    if sys.platform == "win32":
        _fake_engine(monkeypatch, tmp_path)
    code = _run_smoke(monkeypatch, tmp_path, ["VoiceStudio.exe", "--smoke-test", "--require-gpu"])
    assert code == 1
    assert "SMOKE_FAILED" in _smoke_log(tmp_path)


def test_require_gpu_fails_when_cuda_unavailable(tmp_path, monkeypatch):
    _fake_nvidia_smi(monkeypatch, available=False)
    code = _run_smoke(monkeypatch, tmp_path, ["VoiceStudio.exe", "--smoke-test", "--require-gpu"])
    assert code == 1
    log = _smoke_log(tmp_path)
    assert "SMOKE_FAILED" in log
    assert "CUDA" in log


def test_require_gpu_succeeds_when_cuda_available(tmp_path, monkeypatch):
    _fake_nvidia_smi(monkeypatch, available=True)
    _fake_ffmpeg(monkeypatch)
    _fake_engine(monkeypatch, tmp_path)
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
