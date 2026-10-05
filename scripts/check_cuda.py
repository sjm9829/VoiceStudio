"""CUDA/그래픽 카드 진단(개발 모드 전용)."""

import subprocess, sys

def main() -> int:
    try:
        import torch  # type: ignore
        print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("device:", torch.cuda.get_device_name(0))
    except ImportError:
        print("torch 미설치")
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        print("nvidia-smi exit:", r.returncode)
    except FileNotFoundError:
        print("nvidia-smi 없음")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
