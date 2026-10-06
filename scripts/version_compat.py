"""torch/torchaudio version compatibility helper (Stable ABI policy).

TorchAudio 2.11 is built on the PyTorch Stable ABI: it requires PyTorch 2.11
or newer and also works with future PyTorch releases. Exact version equality
between torch and torchaudio is NOT required (P12.3 build finding: real
Windows build installs torch 2.14.1+cu126 / torchaudio 2.11.0+cu126).
"""

from __future__ import annotations

from packaging.version import Version

MIN_TORCH = Version("2.11")
MIN_TORCHAUDIO = Version("2.11")


def _base_version(raw: str) -> Version:
    return Version(raw.split("+")[0])


def check_torch_torchaudio_compat(torch, torchaudio) -> tuple[bool, str]:
    """Return (ok, message) under the Stable ABI compatibility rule."""
    torch_version = _base_version(torch.__version__)
    audio_version = _base_version(torchaudio.__version__)
    if torch_version < MIN_TORCH:
        return False, (
            f"torch {torch.__version__} < 2.11 required by the Stable ABI rule"
        )
    if audio_version < MIN_TORCHAUDIO:
        return False, (
            f"torchaudio {torchaudio.__version__} < 2.11 is unsupported"
        )
    if audio_version > torch_version:
        return False, (
            f"torchaudio {torchaudio.__version__} is newer than torch "
            f"{torch.__version__}; torchaudio must run on torch >= 2.11"
        )
    return True, (
        f"torch {torch.__version__} / torchaudio {torchaudio.__version__} "
        "(Stable ABI: exact equality not required)"
    )
