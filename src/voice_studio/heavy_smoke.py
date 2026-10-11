"""GGUF-only frozen smoke: no torch/qwen_tts dependency or model-weight load.

Default smoke validates packaged imports and llama.cpp binary presence without
requiring a GPU. --require-gpu additionally probes the Windows CUDA driver.
Real inference/VRAM quality still requires a separate Windows E2E.
"""

from __future__ import annotations

import ctypes
import importlib
import sys

GPU_OPTIONAL_CHECKS = ("CUDA available", "GPU")
RUNTIME_MODULES = (
    "numpy",
    "safetensors",
    "huggingface_hub",
    "faster_whisper",
    "ctranslate2",
)


def _check_gguf_runtime() -> None:
    for module in RUNTIME_MODULES:
        importlib.import_module(module)


def _check_llama_engine() -> None:
    from voice_studio.services.gguf_model_manager import GgufModelManager

    manager = GgufModelManager()
    if not manager.is_engine_ready():
        raise FileNotFoundError(f"llama-tts engine missing: {manager.engine_binary()}")


def _cuda_device_count() -> int:
    """Use the Windows driver API; do not pull PyTorch back into GGUF builds."""
    if sys.platform != "win32":
        raise RuntimeError("CUDA driver probing is supported on Windows only")
    driver = ctypes.WinDLL("nvcuda.dll")
    code = driver.cuInit(0)
    if code != 0:
        raise RuntimeError(f"cuInit failed: {code}")
    count = ctypes.c_int()
    code = driver.cuDeviceGetCount(ctypes.byref(count))
    if code != 0 or count.value < 1:
        raise RuntimeError(f"No CUDA GPU (cuDeviceGetCount={code}, count={count.value})")
    return int(count.value)


def heavy_checks(checks: list, require_gpu: bool = False) -> None:
    """Append (name, ok, detail) records for GGUF packaging diagnostics."""

    def _add(name: str, fn) -> None:
        try:
            fn()
            checks.append((name, True, ""))
        except Exception as exc:
            checks.append((name, False, f"{type(exc).__name__}: {exc}"))

    _add("GGUF runtime imports", _check_gguf_runtime)
    _add("llama-tts engine", _check_llama_engine)

    if not require_gpu:
        # Deliberately do not probe CUDA during packaging; build hosts may be CPU-only.
        checks.append(("CUDA available", False, "SKIP (optional for packaging smoke)"))
        checks.append(("GPU", False, "SKIP (requires --require-gpu)"))
        return

    _add("CUDA available", _cuda_device_count)

    def _gpu() -> None:
        print(f"CUDA devices: {_cuda_device_count()}")

    _add("GPU", _gpu)
