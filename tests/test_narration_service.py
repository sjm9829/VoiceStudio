import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pytest
from voice_studio.services.narration_service import NarrationService, job_gap_flags
from voice_studio.services.audio_service import AudioService
from voice_studio.infra.ffmpeg_adapter import FakeFfmpegAdapter
from voice_studio.infra.profile_repository import ProfileRepository
from voice_studio.infra.qwen_adapter import FakeQwenAdapter
from voice_studio.core.errors import WorkerError
from voice_studio.domain.generation_job import JobStatus

def make_service(tmp_path):
    qwen = FakeQwenAdapter()
    audio = AudioService(FakeFfmpegAdapter())
    return NarrationService(qwen, audio, ProfileRepository(tmp_path / "profiles")), qwen

def test_build_job_segments(tmp_path):
    svc, _ = make_service(tmp_path)
    job = svc.build_job("u", "첫 문장입니다. 둘째 문장입니다.\n\n새 문단입니다.")
    assert job.status == JobStatus.PENDING
    assert len(job.segments) >= 2
    assert job.job_id

def test_empty_script_rejected(tmp_path):
    svc, _ = make_service(tmp_path)
    try:
        svc.build_job("u", "   ")
        assert False
    except Exception:
        pass

def test_generate_produces_single_mp3(tmp_path):
    svc, qwen = make_service(tmp_path)
    job = svc.build_job("u", "안녕하세요. 테스트 나레이션입니다.\n\n새 문단의 문장입니다.")
    prompt = FakeQwenAdapter().create_prompt(np.ones(24000, np.float32), 24000, "참조 대사")
    out = tmp_path / "result.mp3"
    events = []
    path = svc.generate(job, prompt, on_progress=lambda *a: events.append(a),
                        output_path=str(out), bitrate_kbps=192)
    assert path == str(out) and out.is_file()
    assert job.status == JobStatus.COMPLETED
    assert qwen.calls  # 모든 chunk에 대해 generate 호출
    kinds = [e[0] for e in events]
    assert "generating" in kinds and "encoding" in kinds and "done" in kinds

def test_generate_reports_failed_chunk(tmp_path):
    svc, qwen = make_service(tmp_path)
    job = svc.build_job("u", "문장 하나입니다.")
    class Boom(FakeQwenAdapter):
        def generate(self, prompt, text, sample_rate):
            raise RuntimeError("boom")
    svc.qwen = Boom()
    prompt = svc.qwen.create_prompt(np.ones(24000, np.float32), 24000, "참조")
    out = tmp_path / "x.mp3"
    with pytest.raises(WorkerError):
        svc.generate(job, prompt, output_path=str(out), bitrate_kbps=192)
    assert job.failed_chunks == [0]

def test_cancel_check(tmp_path):
    svc, _ = make_service(tmp_path)
    job = svc.build_job("u", "문장 하나입니다. 문장 둘입니다.")
    prompt = FakeQwenAdapter().create_prompt(np.ones(24000, np.float32), 24000, "참조")
    try:
        svc.generate(job, prompt, cancel_check=lambda: True,
                     output_path=str(tmp_path / "c.mp3"), bitrate_kbps=192)
        assert False
    except Exception:
        pass
    assert job.status == JobStatus.CANCELLED

def test_job_gap_flags_default(tmp_path):
    svc, _ = make_service(tmp_path)
    job = svc.build_job("u", "문장입니다. 또 문장입니다.")
    assert job_gap_flags(job) == [False] * len(job.segments)
