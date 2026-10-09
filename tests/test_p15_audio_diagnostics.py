"""P15 음성 진단/처리 경로 회귀 테스트.

원칙: 테스트가 녹색이어도 실제 발음 누락 판정은 Windows RTX GPU 실기에서
진단 WAV(01~04) 청취·파형 비교로 확정해야 한다. 이 파일은 다음 계약만
자동화한다: 진단 옵트인, 무손실 원본 WAV, FFmpeg 리샘플 이전, 가드 계약,
메모리 상한, 기본 경로 무파일.
"""

from __future__ import annotations
import ast
import inspect
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from voice_studio.infra.ffmpeg_adapter import FakeFfmpegAdapter, RealFfmpegAdapter
from voice_studio.services.audio_diagnostics import DiagnosticsCapture, read_wav_f32
from voice_studio.services.audio_service import AudioService, _pad_guard, _resample_linear
from voice_studio.services.narration_service import NarrationService
from voice_studio.infra.profile_repository import ProfileRepository
from voice_studio.domain.generation_job import GenerationJob, JobStatus
from voice_studio.infra.qwen_adapter import FakeQwenAdapter


def _service(tmp_path):
    return AudioService(FakeFfmpegAdapter(duration=30.0))


def test_raw_wav_is_exact_lossless_copy(tmp_path):
    """원본 24kHz PCM이 리샘플·정규화·양자화 없이 float32 WAV로 그대로 저장된다."""
    rng = np.random.default_rng(7)
    pcm = rng.uniform(-1, 1, 3000).astype(np.float32)
    cap = DiagnosticsCapture(_service(tmp_path), str(tmp_path / "diag"), "job1")
    cap.save_qwen_raw(pcm)
    saved, rate = read_wav_f32(tmp_path / "diag" / "job1" / "01_qwen_raw_24k.wav")
    assert rate == 24000
    np.testing.assert_array_equal(saved, pcm)  # 비트 단위 동일(무손실)


def test_diagnostics_off_creates_no_files(tmp_path):
    """diagnostics_dir=None(기본)이면 어떤 진단 파일도 생성되지 않는다."""
    svc = _service(tmp_path)
    cap = DiagnosticsCapture(svc, None, "job2")
    assert cap.enabled is False
    cap.save_qwen_raw(np.zeros(10, np.float32))
    cap.save_pre_encode(np.zeros(10, np.float32))
    cap.save_final(str(tmp_path / "x.mp3"))
    cap.save_mp3_decoded(str(tmp_path / "x.mp3"))
    assert not (tmp_path / "diag").exists()


def test_diagnostics_writes_only_under_job_subdir(tmp_path):
    """진단 파일은 diagnostics_dir/<job_id>/ 아래에만 생기고 사용자 파일을 덮지 않는다."""
    guard_file = tmp_path / "keep.txt"
    guard_file.write_text("user data")
    svc = _service(tmp_path)
    cap = DiagnosticsCapture(svc, str(tmp_path), "job3")
    cap.save_qwen_raw(np.zeros(10, np.float32))
    assert guard_file.read_text() == "user data"
    assert (tmp_path / "job3" / "01_qwen_raw_24k.wav").is_file()


def test_encode_mp3_passes_24k_in_and_48k_out(tmp_path):
    """어댑터 계약: 24kHz 입력 + out_sample_rate=48000(리샘플은 FFmpeg가 담당)."""
    fake = FakeFfmpegAdapter(duration=30.0)
    svc = AudioService(fake)
    svc.encode_mp3(np.zeros(24000, np.float32), 192, str(tmp_path / "o.mp3"))
    assert fake.mp3_out_rates[-1] == 48000
    _, _, size, sr = fake.mp3_encoded[-1]
    assert sr == 24000 and size == 24000 + 480 + 480


def test_realfmpeg_encode_mp3_signature_keeps_out_sample_rate():
    """production 어댑터가 out_sample_rate kwarg를 받는 계약 유지(기본 None)."""
    params = inspect.signature(RealFfmpegAdapter.encode_mp3).parameters
    assert "out_sample_rate" in params
    assert params["out_sample_rate"].default is None


def test_resample_linear_matches_previous_contract():
    """블록 처리로 바꾼 뒤에도 길이·정수 격자·DC 보존 계약이 유지된다."""
    x = np.sin(np.linspace(0, 2 * np.pi, 240)).astype(np.float32)
    y = _resample_linear(x, 24000, 48000)
    assert len(y) == 480
    np.testing.assert_allclose(y[::2], x, atol=1e-6)
    np.testing.assert_allclose(_resample_linear(np.full(100, 0.75, np.float32), 24000, 48000),
                               0.75, atol=1e-6)


def test_resample_linear_memory_is_block_bounded():
    """장문 입력에서 중간 배열이 전체 출력 비례 폭증하지 않는다(tracemalloc 상한).

    입력 20초 @24kHz(480k 샘플) → 출력 960k 샘플(≈3.8MB). 블록 처리 시
    추가 중간 할당은 블록 크기(2^18) 수준이므로 전체 피크가 출력 크기의
    약 2배 이하로 머문다(구버전 6-컬럼 전체 인덱스/가중치 행렬이면 초과).
    """
    x = np.random.default_rng(1).standard_normal(24000 * 20).astype(np.float32)
    tracemalloc.start()
    y = _resample_linear(x, 24000, 48000)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert y.size == 960000
    assert peak < y.nbytes * 4, f"peak={peak/1e6:.1f}MB output={y.nbytes/1e6:.1f}MB"


def _narration(tmp_path):
    repo = ProfileRepository(tmp_path / "profiles")
    from voice_studio.infra.qwen_adapter import VoiceClonePromptSpec
    prompt = VoiceClonePromptSpec(ref_code={"fake": True}, ref_spk_embedding=np.full(256, 0.5, np.float32),
                                  ref_text="참조 대사")
    return NarrationService(FakeQwenAdapter(), _service(tmp_path), repo), prompt


def test_generate_default_writes_no_diagnostics(tmp_path):
    svc, prompt = _narration(tmp_path)
    job = GenerationJob(job_id="j1", profile_uuid="p", script="안녕하세요 서정민입니다.",
                        status=JobStatus.PENDING, segments=["안녕하세요 서정민입니다."],
                        failed_chunks=[], output_path=None)
    out = svc.generate(job, prompt, output_path=str(tmp_path / "a.mp3"), bitrate_kbps=192)
    assert Path(out).is_file()
    assert not (tmp_path / "diag").exists()  # 옵트인 전용 계약


def test_generate_optin_writes_diagnostics_files(tmp_path):
    svc, prompt = _narration(tmp_path)
    job = GenerationJob(job_id="j2", profile_uuid="p", script="안녕하세요 서정민입니다.",
                        status=JobStatus.PENDING, segments=["안녕하세요 서정민입니다."],
                        failed_chunks=[], output_path=None)
    out = svc.generate(job, prompt, output_path=str(tmp_path / "b.mp3"), bitrate_kbps=192,
                       diagnostics_dir=str(tmp_path / "diag"))
    assert Path(out).is_file()
    d = tmp_path / "diag" / "j2"
    raw, rate = read_wav_f32(d / "01_qwen_raw_24k.wav")
    assert rate == 24000 and raw.size > 0
    pre, rate2 = read_wav_f32(d / "02_pre_encode.wav")
    assert rate2 == 24000 and pre.size == raw.size + 480 + 480  # 가드 포함(24kHz)
    assert (d / "03_final.mp3").read_bytes() == Path(out).read_bytes()


def test_worker_reads_diagnostics_env():
    """worker가 VOICE_STUDIO_DIAGNOSTICS_DIR을 diagnostics_dir로 전달하는 계약(AST 확인)."""
    src = Path("src/voice_studio/workers/worker_main.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_narrate")
    text = ast.get_source_segment(src, fn)
    assert 'os.environ.get("VOICE_STUDIO_DIAGNOSTICS_DIR")' in text
    assert "diagnostics_dir=diagnostics_dir" in text


def test_generate_forwards_tts_language_optin(tmp_path):
    """tts_language=None(기본)과 명시 "Korean"이 qwen.generate에 전달되는 계약."""
    svc, prompt = _narration(tmp_path)
    job = GenerationJob(job_id="j3", profile_uuid="p", script="테스트", status=JobStatus.PENDING,
                        segments=["테스트"], failed_chunks=[], output_path=None)
    svc.generate(job, prompt, output_path=str(tmp_path / "c.mp3"), bitrate_kbps=192)
    assert svc.qwen.calls  # 기본 동작 무변경
    job2 = GenerationJob(job_id="j4", profile_uuid="p", script="테스트", status=JobStatus.PENDING,
                         segments=["테스트"], failed_chunks=[], output_path=None)
    svc.generate(job2, prompt, output_path=str(tmp_path / "d.mp3"), bitrate_kbps=192,
                 tts_language="Korean")
    assert len(svc.qwen.calls) == 2


def test_real_qwen_adapter_forwards_language_official_arg(monkeypatch):
    """RealQwenAdapter.generate가 qwen-tts 0.1.1 공식 language 인자를 전달한다(None 기본).

    dev env에 qwen_tts의 전체 의존성(librosa 등)이 없으므로 공식 export와 동일한
    최소 VoiceClonePromptItem 스텁을 qwen_tts 모듈에 주입해 import 경계만 검증한다.
    """
    import sys, types, dataclasses
    stub = types.ModuleType("qwen_tts")

    @dataclasses.dataclass
    class VoiceClonePromptItem:
        ref_code: object = None
        ref_spk_embedding: object = None
        x_vector_only_mode: bool = False
        icl_mode: bool = True
        ref_text: object = None

    stub.VoiceClonePromptItem = VoiceClonePromptItem
    monkeypatch.setitem(sys.modules, "qwen_tts", stub)
    from voice_studio.infra.qwen_adapter import RealQwenAdapter, VoiceClonePromptSpec
    adapter = RealQwenAdapter.__new__(RealQwenAdapter)
    seen = {}

    class _StubModel:
        def generate_voice_clone(self, text, language=None, voice_clone_prompt=None):
            seen.update(text=text, language=language, prompt=voice_clone_prompt)
            return [np.zeros(240, np.float32)], 24000

    adapter._model = _StubModel()
    prompt = VoiceClonePromptSpec(ref_code=np.zeros(4, np.float32),
                                  ref_spk_embedding=np.zeros(8, np.float32), ref_text="t")
    wav = adapter.generate(prompt, "안녕", 24000)
    assert wav.dtype == np.float32 and seen["language"] is None  # 기본값 Auto 유지
    adapter.generate(prompt, "안녕", 24000, language="Korean")
    assert seen["language"] == "Korean"


def test_worker_env_tts_language_wiring():
    """worker가 VOICE_STUDIO_TTS_LANGUAGE를 narration.generate에 전달하는 계약(AST)."""
    src = Path("src/voice_studio/workers/worker_main.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_narrate")
    text = ast.get_source_segment(src, fn)
    assert 'os.environ.get("VOICE_STUDIO_TTS_LANGUAGE")' in text
    assert "tts_language=tts_language" in text
