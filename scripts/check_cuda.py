"""CUDA/PyTorch/모델 런타임 진단(P12.2-13).

빌드 스크립트(build_windows.bat)가 이 스크립트의 실패(exit != 0)로 빌드를 중단한다.
검사 항목: torch/torchaudio import/CUDA 사용 가능/GPU 이름·CC/torch CUDA 버전/
bf16 지원/qwen_tts import/Qwen3TTSModel import/nvidia-smi(P12.3-14).
목표 GPU는 RTX 2070 SUPER(Turing, CC 7.5)이며 기본 dtype은 float16이다.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from version_compat import check_torch_torchaudio_compat  # noqa: E402


def main() -> int:
    ok = True
    try:
        import torch  # type: ignore
    except ImportError as e:
        print(f"[FAIL] torch import: {e}")
        return 1
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    try:
        import torchaudio  # type: ignore
        print("torchaudio:", torchaudio.__version__)
        compat_ok, compat_msg = check_torch_torchaudio_compat(torch, torchaudio)
        if compat_ok:
            print("[OK] torch/torchaudio:", compat_msg)
        else:
            print(f"[FAIL] {compat_msg}")
            ok = False
    except ImportError as e:
        print(f"[FAIL] torchaudio import: {e}")
        return 1
    if not torch.cuda.is_available():
        print("[FAIL] CUDA를 사용할 수 없습니다. CUDA wheel과 드라이버를 확인하십시오.")
        return 1
    try:
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        cc = major * 10 + minor
        print("device:", name, "cc:", cc)
        bf16 = torch.cuda.is_bf16_supported()
        print("bf16 supported:", bf16)
        # 목표 GPU(RTX 2070 SUPER, cc 75) 정보 출력: FP16 기본 정책과 일치하는지 안내
        if cc < 70:
            print(f"[FAIL] CC {cc} < 7.0 (Turing 이상 필요)")
            ok = False
        if bf16:
            print("[INFO] bf16 지원 GPU: dtype 정책이 bfloat16을 선택합니다.")
        else:
            print("[INFO] bf16 미지원 GPU(Turing 등): dtype 정책이 float16을 선택합니다.")
        if "2070 SUPER" in name:
            print("[INFO] 목표 GPU(RTX 2070 SUPER) 확인: 기본 dtype은 float16.")
    except Exception as e:
        print(f"[FAIL] GPU 정보 조회 실패: {e}")
        ok = False
    # qwen_tts 실제 import 검증(P12.2-13: 빌드 시점에 worker 의존성 확인)
    try:
        from qwen_tts import Qwen3TTSModel  # noqa: F401
        print("[OK] qwen_tts import 성공")
    except Exception as e:
        print(f"[FAIL] qwen_tts import: {e}")
        ok = False
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
        print("nvidia-smi exit:", r.returncode)
        if r.returncode != 0:
            ok = False
    except FileNotFoundError:
        print("[WARN] nvidia-smi 없음(드라이버 PATH 확인 필요)")
    if not ok:
        print("CHECK_CUDA_FAILED")
        return 1
    print("CHECK_CUDA_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
