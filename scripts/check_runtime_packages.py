"""빌드 PC용 런타임 패키지 검사(GPU/빌드 역할 분리).

빌드 머신에는 NVIDIA GPU가 없어도 된다. 검사 범위는 패키지 설치 무결성뿐이다:
torch/torchaudio import, CUDA wheel 여부(torch.version.cuda), torch/torchaudio
버전 family 일치, qwen_tts/Qwen3TTSModel import.

GPU가 실제로 사용 가능한지는 실패 조건이 아니다. 실제 GPU 검증은
P13 대상 PC에서 scripts/check_cuda.py와 frozen --require-gpu 스모크로 수행한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))



def main() -> int:
    """P18-F6: GGUF 단일 백엔드 프로덕션 런타임 계약 검사(torch/qwen_tts 제거)."""
    ok = True
    for mod in (
        "numpy",
        "safetensors",
        "huggingface_hub",
        "faster_whisper",
        "ctranslate2",
        "PySide6.QtCore",
    ):
        try:
            import importlib
            importlib.import_module(mod)
            print(f"[OK] {mod}")
        except Exception as e:
            print(f"[FAIL] {mod} import: {e}")
            ok = False
    if not ok:
        print("CHECK_RUNTIME_PACKAGES_FAILED")
        return 1
    print("CHECK_RUNTIME_PACKAGES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
