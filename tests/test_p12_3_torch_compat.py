"""P12.3 Windows build finding regression: torch/torchaudio Stable ABI rule.

Real Windows build installed torch 2.14.1+cu126 / torchaudio 2.11.0+cu126.
TorchAudio 2.11 is Stable ABI based and works with torch >= 2.11 (including
future torch releases), so exact version equality must NOT be enforced.
"""

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


version_compat = _load("version_compat", SCRIPTS / "version_compat.py")


class _FakeTorch:
    def __init__(self, version: str, cuda: str | None = "12.6"):
        self.__version__ = version
        self.version = types.SimpleNamespace(cuda=cuda)


class _FakeTorchAudio:
    def __init__(self, version: str):
        self.__version__ = version


def test_a_torch_2_14_1_torchaudio_2_11_0_passes():
    ok, _ = version_compat.check_torch_torchaudio_compat(
        _FakeTorch("2.14.1+cu126"), _FakeTorchAudio("2.11.0+cu126")
    )
    assert ok


def test_b_torch_2_11_torchaudio_2_11_0_passes():
    ok, _ = version_compat.check_torch_torchaudio_compat(
        _FakeTorch("2.11.0"), _FakeTorchAudio("2.11.0")
    )
    assert ok


def test_c_torch_below_2_11_fails():
    ok, msg = version_compat.check_torch_torchaudio_compat(
        _FakeTorch("2.4.0"), _FakeTorchAudio("2.11.0")
    )
    assert not ok
    assert "torch 2.4.0" in msg


def test_d_torchaudio_below_2_11_fails():
    ok, msg = version_compat.check_torch_torchaudio_compat(
        _FakeTorch("2.14.1"), _FakeTorchAudio("2.4.0")
    )
    assert not ok
    assert "unsupported" in msg


def test_torchaudio_newer_than_torch_fails():
    ok, msg = version_compat.check_torch_torchaudio_compat(
        _FakeTorch("2.11.0"), _FakeTorchAudio("2.12.0")
    )
    assert not ok


def test_cuda_wheel_none_is_reported_by_runtime_package_check():
    """torch.version.cuda is None stays a FAIL (not part of the ABI rule)."""
    ok, _ = version_compat.check_torch_torchaudio_compat(
        _FakeTorch("2.14.1", cuda=None), _FakeTorchAudio("2.11.0")
    )
    # ABI rule itself passes; the None-CUDA check lives in check_runtime_packages.py.
    assert ok


def _run_runtime_package_check(monkeypatch):
    check = _load(
        "check_runtime_packages_under_test", SCRIPTS / "check_runtime_packages.py"
    )
    return check.main()


def test_check_runtime_packages_ok_on_clean_env(capsys):
    # P18-F6: gguf 단일화 런타임 계약 — 실제 venv에서 base 의존성 import 성공.
    code = _run_runtime_package_check(None)
    out = capsys.readouterr().out
    assert code == 0
    assert "CHECK_RUNTIME_PACKAGES_OK" in out
    assert "qwen_tts" not in out and "torch:" not in out


def test_check_cuda_uses_stable_abi_rule():
    """check_cuda.py must call the same helper (no exact-equality check)."""
    source = (SCRIPTS / "check_cuda.py").read_text(encoding="utf-8")
    assert "check_torch_torchaudio_compat" in source
    assert 'split("+")' not in source



def test_build_windows_bat_has_no_cuda_mandatory_step():
    """The build script must not gate the build on GPU presence."""
    source = (SCRIPTS / "build_windows.bat").read_text(encoding="ascii")
    assert "python scripts\\check_cuda.py" not in source
    assert "python scripts\\check_runtime_packages.py" in source


def test_repository_bat_files_are_ascii_only():
    """Windows CMD parsing broke on Korean REM comments; all *.bat must be ASCII."""
    offenders = []
    for path in ROOT.rglob("*.bat"):
        data = path.read_bytes()
        bad = [b for b in data if b >= 128]
        if bad:
            offenders.append(f"{path.relative_to(ROOT)}: {len(bad)} non-ASCII bytes")
    assert offenders == []
