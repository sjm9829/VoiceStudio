"""설치 후 Self-Diagnosis 수집기(P12.2-22).

메인 프로세스 구조 계약(torch/qwen_tts/faster_whisper 비import)을 지키기 위해
UI는 이 모듈을 자식 인터프리터(--json)로 실행하고 결과 JSON만 받는다.
사용자용 요약 문장은 UI에, 세부 기술 정보는 logs/diagnosis.log에 기록한다.
"""

from __future__ import annotations
import datetime
import json
import subprocess


def _gpu_lines() -> dict[str, str]:
    lines: dict[str, str] = {}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        gpu = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
        lines["그래픽 카드"] = f"NVIDIA {gpu}" if gpu else "NVIDIA 그래픽 카드를 찾을 수 없습니다."
        lines["CUDA"] = "사용 가능" if gpu else "사용 불가(NVIDIA 카드 없음)"
    except FileNotFoundError:
        lines["그래픽 카드"] = "NVIDIA 그래픽 카드를 사용할 수 없습니다."
        lines["CUDA"] = "사용 불가"
    return lines


def _audio_line() -> str:
    from .infra.ffmpeg_adapter import RealFfmpegAdapter
    try:
        RealFfmpegAdapter().probe_version()
        return "정상"
    except Exception:
        return "확인 필요 (설치 폴더의 ffmpeg를 찾지 못했습니다)"


def _whisper_line() -> str:
    try:
        import faster_whisper  # noqa: F401
        return "사용 가능"
    except Exception:
        return "직접 대사 입력으로 사용 가능"


def _model_line() -> str:
    from .services.model_manager import ModelManager
    try:
        return "설치됨" if ModelManager().is_downloaded() else "아직 받지 않음 (설정에서 받을 수 있습니다)"
    except Exception:
        return "아직 받지 않음 (설정에서 받을 수 있습니다)"


def collect() -> dict:
    """UI 요약(summary)과 로그용 상세(detail)를 함께 만든다."""
    summary = _gpu_lines()
    summary["음성 모델"] = _model_line()
    summary["오디오 구성 요소"] = _audio_line()
    summary["자동 받아쓰기"] = _whisper_line()
    detail: dict[str, str] = dict(summary)
    try:
        import torch
        detail["torch"] = torch.__version__
        detail["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            detail["gpu_name"] = torch.cuda.get_device_name(0)
            detail["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    except Exception as e:
        detail["torch"] = f"미설치({type(e).__name__})"
    try:
        import qwen_tts  # noqa: F401
        detail["qwen_tts_import"] = "ok"
    except Exception as e:
        detail["qwen_tts_import"] = f"실패({type(e).__name__})"
    try:
        from .services.model_manager import ModelManager
        detail["model_path"] = ModelManager().model_path()
    except Exception:
        detail["model_path"] = "없음"
    return {"summary": summary, "detail": detail}


def write_log(data: dict) -> None:
    from .core.paths import logs_dir
    d = logs_dir()
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "diagnosis.log", "a", encoding="utf-8") as fh:
        fh.write(f"--- {datetime.datetime.now().isoformat(timespec='seconds')} ---\n")
        fh.write("\n".join(f"{k}: {v}" for k, v in data["detail"].items()) + "\n")


def main() -> int:
    try:
        data = collect()
        print(json.dumps(data, ensure_ascii=False))
        if "--log" in sys.argv:
            write_log(data)
        return 0
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False))
        return 1


import sys  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
