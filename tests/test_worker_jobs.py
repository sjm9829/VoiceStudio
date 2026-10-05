"""worker job 스키마/실행 계약 통합 테스트(GPU 불필요)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from voice_studio.workers.job_schema import (JOB_SCHEMA_VERSION, JobSchemaError, parse_job,
                                             build_narrate_payload, build_register_payload)
from voice_studio.workers.launcher import is_frozen, worker_command
from voice_studio.workers.linebuffer import JsonlBuffer


def _narrate_payload(**over):
    payload = build_narrate_payload(
        job_id="job-1", profile_uuid="p-uuid", segments=["안녕하세요.", "두 번째 구간입니다."],
        gap_flags=[False, True], profile_dir="/tmp/profiles", output_path="/tmp/out.mp3",
        bitrate_kbps=192, model_path="/tmp/models/snapshot")
    payload.update(over)
    return payload


def test_narrate_payload_roundtrip():
    job = parse_job(_narrate_payload())
    assert job.mode == "narrate" and job.schema_version == JOB_SCHEMA_VERSION
    assert job.segments == ["안녕하세요.", "두 번째 구간입니다."]
    assert job.gap_flags == [False, True]
    assert job.bitrate_kbps == 192


def test_register_payload_roundtrip():
    payload = build_register_payload(
        job_id="job-2", name="내 목소리", source_path="/tmp/ref.mp3",
        start_s=1.0, end_s=16.0, ref_text="안녕하세요.", profile_dir="/tmp/profiles",
        model_path="/tmp/models/snapshot")
    job = parse_job(payload)
    assert job.mode == "register" and job.name == "내 목소리"
    assert job.ref_text == "안녕하세요."


def test_missing_field_is_schema_error_with_code():
    data = _narrate_payload()
    del data["output_path"]
    with pytest.raises(JobSchemaError) as e:
        parse_job(data)
    assert e.value.code == "E_JOB_MISSING_FIELD"


def test_wrong_type_is_schema_error():
    data = _narrate_payload(segments="하나의 문자열")
    with pytest.raises(JobSchemaError) as e:
        parse_job(data)
    assert e.value.code == "E_JOB_BAD_VALUE"


def test_unknown_mode_and_version():
    with pytest.raises(JobSchemaError) as e:
        parse_job(_narrate_payload(mode="??"))
    assert e.value.code == "E_JOB_MODE"
    with pytest.raises(JobSchemaError) as e:
        parse_job(_narrate_payload(schema_version=99))
    assert e.value.code == "E_JOB_VERSION"


def test_dev_worker_command_uses_same_entrypoint():
    assert not is_frozen()
    program, args = worker_command("/tmp/job.json")
    assert program == sys.executable
    assert args[:3] == ["-m", "voice_studio.main", "--worker"]


def test_frozen_worker_command(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    program, args = worker_command("/tmp/job.json")
    assert program == sys.executable  # VoiceStudio.exe 자체
    assert args == ["--worker", "/tmp/job.json"]


def test_jsonl_buffer_handles_partial_chunks():
    buf = JsonlBuffer()
    ev = json.dumps({"kind": "progress", "index": 1, "total": 3}, ensure_ascii=False)
    cut = len(ev) // 2
    assert buf.feed(ev[:cut].encode()) == []           # 반쪽은 대기
    out = buf.feed(ev[cut:].encode() + b"\n" + b'{"kind":"result"}\n')
    assert len(out) == 2 and out[0]["kind"] == "progress" and out[1]["kind"] == "result"


def test_jsonl_buffer_drops_garbage_line():
    buf = JsonlBuffer()
    out = buf.feed(b"not json\n{\"ok\": true}\n")
    assert len(out) == 1 and out[0]["ok"] is True
