"""--smoke-test의 heavy 런타임 검사(P18 GGUF 계약).

P18부터 유일 백엔드는 Qwen3-TTS 1.7B Q8_0 GGUF(llama.cpp llama-tts)이며
torch/torchaudio/qwen_tts 0.6B 런타임은 패키징에서 제거되었다. 따라서 heavy
검사도 GGUF 런타임 계약으로 교체한다:

- GGUF 어댑터/상수 import(항상 필수)
- llama-tts 실행 파일 + CUDA 런타임 DLL 존재(frozen에선 필수, dev에선 SKIP 가능)
- CUDA driver/GPU는 nvidia-smi 기반(PyTorch 의존 없음). --require-gpu에서만 fatal

기본 smoke의 목적은 "패키징이 망가지지 않았는지 확인"이므로 CUDA/GPU는 optional이다.
require_gpu=True(P13, --require-gpu)에서만 CUDA/GPU 실패가 fatal이 된다.
"""

from __future__ import annotations
import os
import shutil
import subprocess
import sys
from pathlib import Path

GPU_OPTIONAL_CHECKS = ("CUDA available", "GPU")

# llama-b11540 win-cuda 번들이 포함하는 필수 CUDA 런타임 DLL 접두어.
_REQUIRED_CUDA_DLL_PREFIXES = ("cudart64", "cublas64", "cublasLt64")


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _engine_dir() -> Path:
    from voice_studio.core.paths import gguf_engine_dir
    return gguf_engine_dir()


def _engine_binary_name() -> str:
    return "llama-tts.exe" if os.name == "nt" else "llama-tts"


def heavy_checks(checks: list, require_gpu: bool = False) -> None:
    """checks 리스트에 (name, ok, detail)을 추가한다. 예외로 실패를 기록한다.

    require_gpu=False: CUDA available/GPU와 dev 환경의 엔진 번들 검사는 SKIP으로 기록된다.
    require_gpu=True: CUDA/GPU/엔진 실패는 fatal(ok=False).
    """

    def _add(name: str, fn, optional_dev: bool = False) -> None:
        try:
            fn()
            checks.append((name, True, ""))
        except Exception as e:
            skippable = (
                (name in GPU_OPTIONAL_CHECKS or (optional_dev and not _is_frozen()))
                and not require_gpu
            )
            if skippable:
                checks.append((name, False, f"SKIP ({type(e).__name__}: {e})"))
            else:
                checks.append((name, False, f"{type(e).__name__}: {e}"))

    def _gguf_contract():
        from voice_studio.core import gguf as gguf_cfg  # noqa: F401
        from voice_studio.infra import gguf_adapter  # noqa: F401
        # 고정 revision 계약 상수 존재 확인(임의 latest 금지).
        assert gguf_cfg.GGUF_MODEL_REVISION, "GGUF_MODEL_REVISION 누락"
        assert gguf_cfg.LLAMA_RELEASE_TAG, "LLAMA_RELEASE_TAG 누락"

    def _gguf_engine():
        engine = _engine_dir()
        binary = engine / _engine_binary_name()
        assert binary.is_file(), f"llama-tts 실행 파일 없음: {binary}"
        # CUDA 런타임 DLL(LLAMA_CUDA 번들) 존재 확인. cuBLASLt는 일부 패키징에서
        # 생략 가능하므로 cudart/cublas 중 하나 이상만 요구한다.
        if os.name == "nt":
            dlls = [p.name for p in engine.glob("*.dll")]
            assert dlls, f"엔진 DLL 없음: {engine}"
            has_core = any(n.startswith(_REQUIRED_CUDA_DLL_PREFIXES) for n in dlls)
            assert has_core, f"CUDA 런타임 DLL 없음: {engine}"

    def _cuda():
        _nvidia_smi()
        name = _gpu_name()
        print("GPU:", name)

    def _gpu():
        _nvidia_smi()
        name = _gpu_name()
        print("GPU:", name)

    def _nvidia_smi() -> str:
        exe = shutil.which("nvidia-smi")
        if not exe and os.name == "nt":
            fallback = Path(r"C:\Windows\System32\nvidia-smi.exe")
            exe = str(fallback) if fallback.is_file() else None
        if not exe:
            raise AssertionError("nvidia-smi를 찾을 수 없음(CUDA driver 미설치?)")
        try:
            out = subprocess.run([exe, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=15)
        except OSError as e:
            raise AssertionError(f"nvidia-smi 실행 실패: {e}") from e
        if out.returncode != 0 or not out.stdout.strip():
            raise AssertionError(f"nvidia-smi 실패: {out.stderr.strip() or out.returncode}")
        return out.stdout.strip()

    def _gpu_name() -> str:
        exe = shutil.which("nvidia-smi") or (
            r"C:\Windows\System32\nvidia-smi.exe" if os.name == "nt" else None)
        out = subprocess.run([exe, "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
        return (out.stdout.splitlines() or ["GPU"])[0].strip()

    _add("GGUF runtime import", _gguf_contract)
    # frozen: 번들 엔진 필수. dev: engine 미준비가 정상이므로 SKIP.
    _add("GGUF llama-tts engine", _gguf_engine, optional_dev=True)
    _add("CUDA available", _cuda)
    _add("GPU", _gpu)
