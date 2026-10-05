"""worker job/event 스키마. 메인 프로세스와 worker 프로세스 사이의 계약.

job 입력(JSON 파일)의 실제 파서/검증은 workers/job_schema.py가 담당한다.
이 모듈은 worker → 메인 프로세스로 흐르는 이벤트 형식을 정의한다.
"""

from __future__ import annotations
import json
import sys
from typing import Any

def event(kind: str, **detail: Any) -> dict[str, Any]:
    """kind: status | progress | error | result"""
    ev = {"kind": kind}
    ev.update(detail)
    return ev

def status_event(phase: str, index: int = 0, total: int = 0, job_id: str = "") -> dict[str, Any]:
    return event("status", phase=phase, index=index, total=total, job_id=job_id)

def progress_event(index: int, total: int, kind: str = "generating", job_id: str = "") -> dict[str, Any]:
    return event("progress", index=index, total=total, phase=kind, job_id=job_id)

def error_event(code: str, message: str, detail: str = "") -> dict[str, Any]:
    return event("error", code=code, message=message, detail=detail)

def result_event(output_path: str, job_id: str = "", **extra: Any) -> dict[str, Any]:
    ev = event("result", output_path=output_path, job_id=job_id)
    ev.update({k: v for k, v in extra.items() if v is not None})
    return ev

def emit(ev: dict[str, Any]) -> None:
    """stdout JSONL 1줄 출력. line-buffer flush로 QProcess readyRead가 즉시 받도록 한다."""
    sys.stdout.write(json.dumps(ev, ensure_ascii=False) + "\n")
    sys.stdout.flush()
