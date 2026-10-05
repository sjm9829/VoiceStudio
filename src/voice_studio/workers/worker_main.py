"""음성 작업 worker 프로세스 진입점.

- job JSON 파일 경로를 argv[1]로 받는다.
- 모델을 로드하고 작업을 수행한 뒤 stdout으로 JSONL 이벤트를 출력한다.
- 성공/취소/오류와 무관하게 반드시 종료해 CUDA context를 소멸시킨다.
- 메인(PySide6) 프로세스는 이 모듈을 import하지 않고 QProcess로 실행한다.
"""

from __future__ import annotations
import json, os, sys, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/

from voice_studio.core import config                                    # noqa: E402
from voice_studio.core.errors import VoiceStudioError                    # noqa: E402
from voice_studio.workers.protocol import (status_event, progress_event,  # noqa: E402
                                           error_event, result_event)
from voice_studio.services.audio_service import AudioService, concat_pcm  # noqa: E402
from voice_studio.services.text_segmenter import segment_with_flags       # noqa: E402

def emit(ev: dict) -> None:
    sys.stdout.write(json.dumps(ev, ensure_ascii=False) + "\n")
    sys.stdout.flush()

def run_register(job: dict) -> None:
    """참조 구간 분석 → 프롬프트 생성 → 프로필 저장 → worker 종료."""
    emit(status_event("model_loading"))
    from voice_studio.infra.ffmpeg_adapter import RealFfmpegAdapter
    from voice_studio.infra.profile_repository import ProfileRepository
    from voice_studio.infra.qwen_adapter import RealQwenAdapter, VoiceClonePromptSpec
    audio = AudioService(RealFfmpegAdapter())
    pcm = audio.decode_reference_segment(job["source_path"], job["start_s"], job["end_s"])
    emit(status_event("generating"))
    qwen = RealQwenAdapter()
    spec = qwen.create_prompt(pcm, config.REFERENCE_SAMPLE_RATE, job["ref_text"])
    from voice_studio.services.profile_service import ProfileService
    service = ProfileService(ProfileRepository(), audio, qwen)
    profile = service.register(
        name=job["name"], source_path=job["source_path"], start_s=job["start_s"],
        end_s=job["end_s"], ref_text=job["ref_text"], consent=True, prompt=spec,
        waveform=pcm, sample_rate=config.REFERENCE_SAMPLE_RATE)
    emit(result_event(profile.uuid))

def run_narrate(job: dict) -> None:
    """모델 1회 로드 → chunk 순차 생성 → 무음 이어붙이기 → MP3 1회 인코딩."""
    import numpy as np
    emit(status_event("model_loading"))
    from voice_studio.infra.ffmpeg_adapter import RealFfmpegAdapter
    from voice_studio.infra.qwen_adapter import RealQwenAdapter, VoiceClonePromptItem  # noqa
    audio = AudioService(RealFfmpegAdapter())
    qwen = RealQwenAdapter()
    # 저장된 프로필에서 공식 VoiceClonePromptItem을 재구성한다(ref_text 포함, list 전달).
    from voice_studio.infra.profile_repository import ProfileRepository
    from voice_studio.services.profile_service import load_prompt_spec
    prompt = load_prompt_spec(ProfileRepository(job["profile_dir"]), job["profile_uuid"])
    item = qwen.build_prompt_item(prompt) if hasattr(qwen, "build_prompt_item") else None
    emit(status_event("generating"))
    chunks: list[np.ndarray] = []
    total = len(job["segments"])
    for i, text in enumerate(job["segments"]):
        emit(progress_event(i + 1, total))
        wav = qwen.generate(prompt, text, config.REFERENCE_SAMPLE_RATE)
        chunks.append(np.asarray(wav, dtype=np.float32))
    emit(status_event("encoding"))
    pcm = concat_pcm(chunks, config.REFERENCE_SAMPLE_RATE, config.CHUNK_GAP_MS,
                     config.PARAGRAPH_GAP_MS, [bool(f) for f in job["gap_flags"]])
    out = audio.encode_mp3(pcm, job["bitrate_kbps"], job["output_path"])
    emit(result_event(out))

def main(argv: list[str]) -> int:
    if len(argv) < 2:
        emit(error_event("E_JOB_FILE", "job JSON 경로가 필요합니다."))
        return 2
    try:
        job = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
        mode = job.get("mode")
        if mode == "register":
            run_register(job)
        elif mode == "narrate":
            run_narrate(job)
        else:
            emit(error_event("E_JOB_MODE", f"알 수 없는 작업 모드: {mode}"))
            return 2
        return 0
    except VoiceStudioError as exc:
        emit(error_event(exc.code, exc.user_message, exc.detail))
        return 1
    except Exception as exc:
        emit(error_event("E_WORKER_CRASHED", "음성 생성 작업이 비정상 종료되었습니다.",
                         traceback.format_exc(limit=8)))
        return 1
    finally:
        # 프로세스 종료로 CUDA context 소멸(메모리 해제는 empty_cache가 아닌 종료로 보장).
        sys.stdout.flush()

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
