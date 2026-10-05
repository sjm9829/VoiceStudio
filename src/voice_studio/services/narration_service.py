"""나레이션 생성 서비스.

메인 프로세스에서는 이 서비스가 대본 분할/작업 조립만 담당하고,
실제 생성은 worker 프로세스(voice_studio.workers.worker_main)가 수행한다.
worker는 모델을 한 번 로드한 뒤 chunk를 순차 생성한다.
"""

from __future__ import annotations
import uuid
from typing import Iterable
import numpy as np
from ..core import config
from ..core.errors import ProfileError, WorkerError
from ..domain.generation_job import GenerationJob, JobStatus
from ..infra.profile_repository import ProfileRepository
from ..infra.qwen_adapter import VoiceClonePromptSpec, QwenAdapter
from ..services.text_segmenter import segment_with_flags
from ..services.audio_service import AudioService, concat_pcm

class NarrationService:
    """worker 프로세스 내부에서 실제 생성을 수행하는 쪽(QwenAdapter 주입)."""

    def __init__(self, qwen: QwenAdapter, audio: AudioService, profiles: ProfileRepository):
        self.qwen = qwen
        self.audio = audio
        self.profiles = profiles

    def build_job(self, profile_uuid: str, script: str) -> GenerationJob:
        segments, flags = segment_with_flags(script)
        if not segments:
            raise ProfileError("대본을 입력해 주세요.")
        return GenerationJob(
            job_id=str(uuid.uuid4()), profile_uuid=profile_uuid, script=script,
            status=JobStatus.PENDING, segments=segments,
            failed_chunks=[], output_path=None, error=None,
        )

    def generate(self, job: GenerationJob, prompt: VoiceClonePromptSpec, *,
                 cancel_check=None, on_progress=None, output_path: str, bitrate_kbps: int,
                 gap_ms: int = config.CHUNK_GAP_MS, paragraph_gap_ms: int = config.PARAGRAPH_GAP_MS) -> str:
        """chunk를 순차 생성해 하나의 MP3로 완성한다. 동일 VoiceClonePromptItem 사용."""
        if not prompt.ref_text.strip():
            raise ProfileError("프로필의 참조 대사가 비어 있습니다.")
        chunks: list[np.ndarray] = []
        total = len(job.segments)
        for i, text in enumerate(job.segments):
            if cancel_check is not None and cancel_check():
                job.status = JobStatus.CANCELLED
                raise WorkerError("취소됨") if False else _Cancelled()
            if on_progress is not None:
                on_progress("generating", i + 1, total)
            try:
                chunks.append(self.qwen.generate(prompt, text, config.REFERENCE_SAMPLE_RATE))
            except Exception as exc:
                job.failed_chunks.append(i)
                raise WorkerError(f"{i + 1}번째 구간 생성 실패: {exc}") from exc
        if on_progress is not None:
            on_progress("encoding", total, total)
        pcm = concat_pcm(chunks, config.REFERENCE_SAMPLE_RATE, gap_ms, paragraph_gap_ms, job_gap_flags(job))
        out = self.audio.encode_mp3(pcm, bitrate_kbps, output_path)
        if on_progress is not None:
            on_progress("done", total, total)
        job.status = JobStatus.COMPLETED
        job.output_path = out
        return out

class _Cancelled(Exception):
    pass

def job_gap_flags(job: GenerationJob) -> list[bool]:
    """GenerationJob에 문단 경계 플래그가 없으면 모두 False(기본 무음 간격)."""
    stored = getattr(job, "gap_flags", None)
    if stored is not None and len(stored) == len(job.segments):
        return list(stored)
    return [False] * len(job.segments)
