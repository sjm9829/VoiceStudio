"""--smoke-test의 heavy 런타임 검사(P12.3-17).

main.py의 정적 heavy import 계약(test_production_context)을 지키기 위해
torch/torchaudio/qwen_tts import 검사를 이 별도 모듈로 분리한다. --smoke-test 실행 시에만
main.py가 이 모듈을 함수 내부에서 지연 import한다.
"""

from __future__ import annotations


def heavy_checks(checks: list) -> None:
    """checks 리스트에 (name, ok, detail)을 추가한다. 예외로 실패를 기록한다."""

    def _add(name: str, fn) -> None:
        try:
            fn()
            checks.append((name, True, ""))
        except Exception as e:
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
