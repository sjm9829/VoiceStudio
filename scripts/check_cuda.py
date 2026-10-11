"""CUDA/런타임 진단(P18 GGUF 계약).

빌드/검증 스크립트가 실패(exit != 0)로 중단할 수 있다.
검사 항목: nvidia-smi(CUDA driver/GPU 이름), GGUF 런타임 import 가능 여부,
llama-tts 엔진 실재, 필수 CUDA DLL 존재. PyTorch/qwen_tts 의존은 없다(P18).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

REQUIRED_DLL_PREFIXES = ("cudart64", "cublas64", "cublasLt64")


def _engine_dir() -> Path | None:
    try:
        from voice_studio.core.paths import gguf_engine_dir

        return Path(gguf_engine_dir())
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] engine dir 조회 실패: {e}")
        return None


def main() -> int:
    ok = True
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
        if r.returncode != 0:
            print(f"[FAIL] nvidia-smi exit {r.returncode}: {r.stderr.strip()[:200]}")
            ok = False
        else:
            print("[OK] nvidia-smi:", " | ".join(r.stdout.strip().splitlines()[:2]) or "(정보 없음)")
    except FileNotFoundError:
        print("[FAIL] nvidia-smi 없음(CUDA driver 미설치?)")
        ok = False
    except subprocess.TimeoutExpired:
        print("[FAIL] nvidia-smi 응답 없음")
        ok = False

    try:
        from voice_studio.core import gguf as gguf_cfg
        from voice_studio.services import gguf_model_manager  # noqa: F401

        print(f"[OK] GGUF 런타임 import (revision {gguf_cfg.GGUF_MODEL_REVISION[:12]}, "
              f"llama {gguf_cfg.LLAMA_RELEASE_TAG})")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] GGUF 런타임 import: {e}")
        return 1

    engine_dir = _engine_dir()
    exe_name = "llama-tts.exe" if sys.platform == "win32" else "llama-tts"
    exe = (engine_dir / exe_name) if engine_dir else None
    if exe is None or not exe.is_file():
        print(f"[WARN] llama-tts 엔진 미설치(앱에서 자동 다운로드): {exe}")
    else:
        print(f"[OK] llama-tts 엔진: {exe}")
        if sys.platform == "win32":
            dlls = sorted(engine_dir.glob("*.dll")) if engine_dir else []
            names = {p.name for p in dlls}
            missing = [pre for pre in REQUIRED_DLL_PREFIXES
                       if not any(n.startswith(pre) for n in names)]
            if missing:
                print(f"[WARN] CUDA DLL 접두어 누락: {missing} (앱이 엔진 설치 시 채움)")
            else:
                print(f"[OK] CUDA DLL {len(dlls)}개 존재")

    if not ok:
        print("CHECK_CUDA_FAILED")
        return 1
    print("CHECK_CUDA_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
