"""--smoke-test의 heavy 런타임 검사(P12.3-17, GPU/빌드 역할 분리).

main.py의 정적 heavy import 계약(test_production_context)을 지키기 위해
torch/torchaudio/qwen_tts import 검사를 이 별도 모듈로 분리한다. --smoke-test 실행 시에만
main.py가 이 모듈을 함수 내부에서 지연 import한다.

기본 smoke의 목적은 "패키징이 망가지지 않았는지 확인"이므로 CUDA/GPU는 optional이다.
require_gpu=True(P13, --require-gpu)에서만 CUDA/GPU 실패가 fatal이 된다.
"""

from __future__ import annotations

GPU_OPTIONAL_CHECKS = ("CUDA available", "GPU")


def heavy_checks(checks: list, require_gpu: bool = False) -> None:
    """checks 리스트에 (name, ok, detail)을 추가한다. 예외로 실패를 기록한다.

    require_gpu=False: CUDA available/GPU는 unavailable이면 SKIP(ok=True, detail=SKIP)으로 기록.
    require_gpu=True: CUDA/GPU 실패는 fatal(ok=False).
    """

    def _add(name: str, fn) -> None:
        try:
            fn()
            checks.append((name, True, ""))
        except Exception as e:
            if name in GPU_OPTIONAL_CHECKS and not require_gpu:
                checks.append((name, True, f"SKIP ({type(e).__name__}: {e})"))
            else:
                checks.append((name, False, f"{type(e).__name__}: {e}"))

    def _torch():
        import torch  # noqa: F401

    def _torchaudio():
        import torchaudio  # noqa: F401

    def _cuda():
        import torch
        assert torch.cuda.is_available(), "CUDA 사용 불가"

    def _gpu():
        import torch
        assert torch.cuda.is_available(), "CUDA 사용 불가"
        name = torch.cuda.get_device_name(0)
        print("GPU:", name)
        policy = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        print("dtype policy:", policy)

    def _qwen():
        import qwen_tts  # noqa: F401

    def _qwen_model():
        from qwen_tts import Qwen3TTSModel  # noqa: F401

    _add("Torch import", _torch)
    _add("Torchaudio import", _torchaudio)
    _add("CUDA available", _cuda)
    _add("GPU", _gpu)
    _add("qwen_tts import", _qwen)
    _add("Qwen3TTSModel import", _qwen_model)
