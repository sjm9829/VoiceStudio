"""worker 프로세스 진입점.

메인 프로세스가 완성한 job JSON을 검증 후 실행하고, 진행/결과를 stdout JSONL 이벤트로 출력한다.
Qwen 모델/STT 등 무거운 의존성은 이 프로세스에서만 로드하며, 실제 어댑터만 사용한다.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from voice_studio.core.errors import VoiceStudioError
from voice_studio.workers.job_schema import JobSchemaError, parse_job
from voice_studio.workers.protocol import emit, error_event, result_event, status_event, progress_event


def _make_services(profile_dir: str):
    """worker 전용 서비스(실제 어댑터만 사용)."""
    from voice_studio.infra.ffmpeg_adapter import FfmpegAdapter
    from voice_studio.infra.profile_repository import ProfileRepository
    from voice_studio.services.audio_service import AudioService
    from voice_studio.services.profile_service import ProfileService
    audio = AudioService(FfmpegAdapter())
    repo = ProfileRepository(Path(profile_dir))
    return repo, audio, ProfileService(repo, audio)


def _release_gpu() -> None:
    """CUDA context/VRAM을 반환한다(가능한 경우에만)."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def run_register(job) -> None:
    emit(status_event("model_loading", job_id=job.job_id))
    repo, audio, service = _make_services(job.profile_dir)
    from voice_studio.infra.qwen_adapter import RealQwenAdapter, default_device
    qwen = RealQwenAdapter(model_path=job.model_path or None, device=default_device())
    emit(status_event("analyzing_reference", job_id=job.job_id))
    pcm = audio.decode_reference_segment(job.source_path, job.start_s, job.end_s)
    spec = qwen.create_prompt(pcm, 24000, job.ref_text.strip())
    emit(status_event("saving_profile", job_id=job.job_id))
    profile = service.register(
        name=job.name, source_path=job.source_path, start_s=job.start_s, end_s=job.end_s,
        ref_text=job.ref_text, consent=True, prompt=spec, waveform=pcm, sample_rate=24000)
    emit(result_event(str(profile.profile_dir), profile_uuid=profile.uuid,
                      name=profile.name, job_id=job.job_id))
    _release_gpu()


def run_narrate(job) -> None:
    emit(status_event("model_loading", job_id=job.job_id))
    repo, audio, service = _make_services(job.profile_dir)
    from voice_studio.infra.qwen_adapter import RealQwenAdapter, default_device
    from voice_studio.services.narration_service import NarrationService
    from voice_studio.domain.generation_job import GenerationJob, JobStatus
    qwen = RealQwenAdapter(model_path=job.model_path or None, device=default_device())
    narration = NarrationService(qwen, audio, repo)
    # 모델을 이 프로세스에서 1회 로드한 뒤, 전달받은 segments를 순차 생성한다.
    gen_job = GenerationJob(
        job_id=job.job_id, profile_uuid=job.profile_uuid, script="\n".join(job.segments),
        status=JobStatus.PENDING, segments=list(job.segments), gap_flags=list(job.gap_flags),
        failed_chunks=[], output_path=None)
    prompt = service.load_prompt_spec(job.profile_uuid)
    emit(status_event("generating", job_id=job.job_id, total=len(job.segments)))
    out = narration.generate(
        gen_job, prompt, output_path=job.output_path, bitrate_kbps=job.bitrate_kbps,
        on_progress=lambda kind, i, total: emit(progress_event(i, total, kind=kind, job_id=job.job_id)))
    emit(result_event(out, job_id=job.job_id))
    _release_gpu()


def main(argv: list[str]) -> int:
    """argv: ['worker', job.json]. 계약 위반/실행 실패는 오류 이벤트 후 0이 아닌 값으로 종료."""
    if len(argv) != 2 or argv[0] != "worker":
        emit(error_event("E_WORKER_ARGS", "worker 실행 인자가 올바르지 않습니다.", detail=str(argv)))
        return 2
    try:
        data = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
        job = parse_job(data)
    except JobSchemaError as e:
        emit(error_event(e.code, e.user_message, detail=e.detail))
        return 2
    except Exception as e:
        emit(error_event("E_JOB_INVALID", "작업 파일을 읽을 수 없습니다.", detail=str(e)))
        return 2
    try:
        if job.mode == "register":
            run_register(job)
        else:
            run_narrate(job)
        return 0
    except VoiceStudioError as e:
        emit(error_event(e.code, str(e)))
        return 3
    except Exception as e:
        emit(error_event("E_WORKER_CRASH", "작업 처리 중 오류가 발생했습니다.",
                         detail=f"{type(e).__name__}: {e}"))
        return 3


if __name__ == "__main__":
    # 직접 실행: python worker_main.py job.json → main(["worker", job.json])
    args = ["worker"] + sys.argv[1:] if len(sys.argv) == 2 else sys.argv[1:]
    sys.exit(main(args))
