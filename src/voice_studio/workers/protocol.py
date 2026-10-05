"""worker job/event 스키마. 메인 프로세스와 worker 프로세스 사이의 계약."""

from __future__ import annotations
from typing import Any

# job 입력(JSON 파일): 메인 프로세스가 작성하고 worker가 읽는다.
JOB_SCHEMA = {
    "job_id": str,          # 임시 폴더/트레이싱용 uuid
    "profile_uuid": str,
    "mode": str,            # "register" | "narrate"
    "segments": list,       # narrate: 분할된 대본 목록
    "gap_flags": list,      # narrate: 문단 경계 플래그
    "bitrate_kbps": int,
    "output_path": str,
    "profile_dir": str,     # metadata.json/prompt.safetensors/reference.flac 위치
}

# 등록 모드 추가 필드
REGISTER_EXTRA = {
    "reference_flac": str,  # 선택 구간 FLAC 경로(worker가 읽음)
    "ref_text": str,
}

# worker가 stdout으로 출력하는 JSONL 이벤트
def event(kind: str, **detail: Any) -> dict[str, Any]:
    """kind: status | progress | error | result"""
    ev = {"kind": kind}
    ev.update(detail)
    return ev

def status_event(phase: str, index: int = 0, total: int = 0) -> dict[str, Any]:
    return event("status", phase=phase, index=index, total=total)

def progress_event(index: int, total: int) -> dict[str, Any]:
    return event("progress", index=index, total=total)

def error_event(code: str, message: str, detail: str = "") -> dict[str, Any]:
    return event("error", code=code, message=message, detail=detail)

def result_event(output_path: str) -> dict[str, Any]:
    return event("result", output_path=output_path)
