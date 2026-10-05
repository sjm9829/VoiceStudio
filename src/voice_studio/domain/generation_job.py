"""나레이션 생성 작업 도메인."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

class WorkerPhase(str, Enum):
    """worker 이벤트 단계. UI에는 쉬운 한글로 번역되어 표시된다."""
    MODEL_LOADING = "model_loading"
    GENERATING = "generating"
    ENCODING = "encoding"
    DONE = "done"

PHASE_KO = {
    WorkerPhase.MODEL_LOADING.value: "모델을 준비하는 중…",
    WorkerPhase.GENERATING.value: "구간 {i}/{n} 만드는 중…",
    WorkerPhase.ENCODING.value: "MP3를 만드는 중…",
    WorkerPhase.DONE.value: "완료",
}

@dataclass
class GenerationJob:
    job_id: str
    profile_uuid: str
    script: str
    status: JobStatus = JobStatus.PENDING
    segments: list[str] = field(default_factory=list)
    gap_flags: list[bool] = field(default_factory=list)  # 문단 경계 플래그(첫 구간 제외)
    failed_chunks: list[int] = field(default_factory=list)
    output_path: str | None = None
    error: dict[str, Any] | None = None

    def phase_text(self, phase: WorkerPhase, index: int = 0, total: int = 0) -> str:
        template = PHASE_KO[phase.value]
        return template.format(i=index, n=total) if "{i}" in template else template
