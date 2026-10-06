"""빌드 PC용 런타임 패키지 검사(GPU/빌드 역할 분리).

빌드 머신에는 NVIDIA GPU가 없어도 된다. 검사 범위는 패키지 설치 무결성뿐이다:
torch/torchaudio import, CUDA wheel 여부(torch.version.cuda), torch/torchaudio
버전 family 일치, qwen_tts/Qwen3TTSModel import.

GPU가 실제로 사용 가능한지는 실패 조건이 아니다. 실제 GPU 검증은
P13 대상 PC에서 scripts/check_cuda.py와 frozen --require-gpu 스모크로 수행한다.
"""

from __future__ import annotations


def main() -> int:
    ok = True
    try:
        import torch  # type: ignore
    except ImportError as e:
        print(f"[FAIL] torch import: {e}")
        return 1
    print("torch:", torch.__version__)
    try:
        import torchaudio  # type: ignore
        print("torchaudio:", torchaudio.__version__)
    except ImportError as e:
        print(f"[FAIL] torchaudio import: {e}")
        return 1
    if torchaudio.__version__.split("+")[0] != torch.__version__.split("+")[0]:
        print("[FAIL] torch/torchaudio 버전 family 불일치 - CUDA wheel 조합 확인 필요")
        ok = False
    if torch.version.cuda is None:
        print("[FAIL] CUDA wheel이 아닙니다(torch.version.cuda is None).")
        ok = False
    else:
        print("torch.version.cuda:", torch.version.cuda, "(CUDA wheel 확인)")
        # GPU 존재 여부는 검사하지 않는다(빌드 머신에 GPU 없어도 성공).
    try:
        import qwen_tts  # noqa: F401
        from qwen_tts import Qwen3TTSModel  # noqa: F401
        print("[OK] qwen_tts/Qwen3TTSModel import 성공")
    except Exception as e:
        print(f"[FAIL] qwen_tts import: {e}")
        ok = False
    if not ok:
        print("CHECK_RUNTIME_PACKAGES_FAILED")
        return 1
    print("CHECK_RUNTIME_PACKAGES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
