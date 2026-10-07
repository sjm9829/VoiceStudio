"""실제 Qwen/GPU opt-in 테스트(P13 §1 재작성).

`pytest -m gpu`로만 의미가 있다. 개발 환경(torch 미설치/GPU 없음)에서는 SKIP이
정상이며, SKIP을 성공으로 위장하지 않는다(P13 §34). Windows RTX 실기에서:
  uv run pytest -q tests/test_gpu_real.py -m gpu
기준: 실제 모델 로드 → create_prompt(VoiceClonePromptSpec) → 프로필 저장 →
어댑터 폐기 → 새 모델 인스턴스/프로필 재로드 → 한국어 generate →
np.float32 1-D / 비어있지 않음 / finite / non-silent.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

gpu = pytest.mark.gpu


def _torch():
    """torch를 지연 import한다(개발 환경에 torch가 없어도 collection은 성공한다)."""
    import torch
    return torch

REF_TEXT = "안녕하세요. 참조 음성 테스트 대사입니다."
GEN_TEXT = "안녕하세요. 실제 GPU 생성 검증 문장입니다."


def _cuda_or_skip():
    """torch 미설치/CUDA 미보유 모두 SKIP이지만, GPU가 있는 P13 실행에서는 절대 skip하지 않는다."""
    try:
        t = _torch()
    except ImportError:
        pytest.skip("torch 미설치 환경")
    if not t.cuda.is_available():
        pytest.skip("CUDA GPU가 없는 환경")


def _reference_pcm(seconds=6.0, rate=24000):
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _model_path():
    from voice_studio.services.model_manager import ModelManager
    return ModelManager().download()


def _adapter():
    from voice_studio.infra.qwen_adapter import RealQwenAdapter, production_device
    return RealQwenAdapter(model_path=_model_path(), device=production_device())


def _validation_ffmpeg_adapter():
    """third_party/bin binary를 명시적으로 쓰는 adapter(PATH 비의존, P13 §7)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "runtime_validation_helpers",
        Path(__file__).resolve().parents[1] / "scripts" / "runtime_validation_helpers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.make_validation_adapter()


def _assert_sane_pcm(pcm):
    assert isinstance(pcm, np.ndarray), type(pcm)
    assert pcm.ndim == 1
    assert pcm.dtype == np.float32
    assert pcm.size > 0
    assert np.isfinite(pcm).all()
    # non-silent: 실제 파형이 존재해야 한다(전부 0이면 실패)
    assert float(np.abs(pcm).max()) > 1e-4


@gpu
def test_a_real_qwen_load_and_prompt_spec():
    """A. 실제 Qwen 로드 + create_prompt → VoiceClonePromptSpec 필드 확인."""
    _cuda_or_skip()
    from voice_studio.infra.qwen_adapter import VoiceClonePromptSpec
    adapter = _adapter()
    spec = adapter.create_prompt(_reference_pcm(), 24000, REF_TEXT)
    assert isinstance(spec, VoiceClonePromptSpec)
    assert spec.ref_code is not None
    assert spec.ref_spk_embedding is not None
    assert spec.ref_text == REF_TEXT


@gpu
def test_b_korean_generate_direct(tmp_path):
    """B. 새로 만든 spec으로 한국어 직접 생성: float32 1-D / finite / non-silent."""
    _cuda_or_skip()
    adapter = _adapter()
    spec = adapter.create_prompt(_reference_pcm(), 24000, REF_TEXT)
    pcm = adapter.generate(spec, GEN_TEXT, 24000)
    _assert_sane_pcm(pcm)
    del adapter
    import gc
    gc.collect(); _torch().cuda.empty_cache()


@gpu
def test_c_profile_roundtrip_in_new_model_instance(tmp_path):
    """C. 프로필 저장 → 어댑터 폐기 → 새 모델 인스턴스에서 load_prompt_spec → 생성."""
    _cuda_or_skip()
    from voice_studio.infra.profile_repository import ProfileRepository
    from voice_studio.services.profile_service import ProfileService
    from voice_studio.services.audio_service import AudioService
    profiles_root = tmp_path / "profiles"
    adapter = _adapter()
    spec = adapter.create_prompt(_reference_pcm(), 24000, REF_TEXT)
    repo = ProfileRepository(profiles_root)
    service = ProfileService(repo, AudioService(_validation_ffmpeg_adapter()), qwen=adapter)
    pcm = _reference_pcm()
    wav_path = tmp_path / "ref.wav"
    import wave
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000)
        wf.writeframes((pcm * 32767).astype(np.int16).tobytes())
    profile = service.register(
        name="GPU roundtrip", source_path=str(wav_path), start_s=0.0, end_s=6.0,
        ref_text=REF_TEXT, consent=True, prompt=spec, waveform=pcm, sample_rate=24000)
    assert profile.ref_code_kind == "tensor"
    uuid = profile.uuid

    # 어댑터 폐기(객체 참조 해제 + 캐시 반환)
    del adapter, service, spec
    import gc
    gc.collect()
    _torch().cuda.empty_cache(); _torch().cuda.synchronize()

    # 새 모델 인스턴스 + 저장된 프로필 재로드 → 생성
    adapter2 = _adapter()
    service2 = ProfileService(repo, AudioService(_validation_ffmpeg_adapter()), qwen=adapter2)
    spec2 = service2.load_prompt_spec(uuid)
    assert spec2.ref_code is not None and spec2.ref_spk_embedding is not None
    pcm2 = adapter2.generate(spec2, GEN_TEXT, 24000)
    _assert_sane_pcm(pcm2)


@gpu
def test_d_profile_roundtrip_in_new_process(tmp_path):
    """D. 저장된 프로필을 완전히 새 프로세스에서: load → RealQwenAdapter → generate → PCM sanity."""
    _cuda_or_skip()
    import wave
    from voice_studio.infra.profile_repository import ProfileRepository
    from voice_studio.services.profile_service import ProfileService
    from voice_studio.services.audio_service import AudioService
    from voice_studio.infra.qwen_adapter import RealQwenAdapter, production_device
    profiles_root = tmp_path / "profiles"
    pcm = _reference_pcm()
    wav_path = tmp_path / "ref.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000)
        wf.writeframes((pcm * 32767).astype(np.int16).tobytes())
    adapter = RealQwenAdapter(model_path=_model_path(), device=production_device())
    spec = adapter.create_prompt(pcm, 24000, REF_TEXT)
    service = ProfileService(ProfileRepository(profiles_root), AudioService(_validation_ffmpeg_adapter()), qwen=adapter)
    profile = service.register(
        name="GPU proc", source_path=str(wav_path), start_s=0.0, end_s=6.0,
        ref_text=REF_TEXT, consent=True, prompt=spec, waveform=pcm, sample_rate=24000)
    uuid = profile.uuid
    del adapter, service, spec
    import gc
    gc.collect(); _torch().cuda.empty_cache()

    root = Path(__file__).resolve().parents[1]
    # 새 프로세스에서: 프로필 load → RealQwenAdapter 로드 → generate → PCM sanity까지 실제 수행.
    code = (
        "import sys; sys.path.insert(0, r'{src}');\n"
        "import numpy as np;\n"
        
        "import importlib.util as ilu;\n"
        "vs = ilu.spec_from_file_location('rvh', r'{rvh}'); m = ilu.module_from_spec(vs); vs.loader.exec_module(m);\n"
        "from voice_studio.infra.profile_repository import ProfileRepository;\n"
        "from voice_studio.infra.qwen_adapter import RealQwenAdapter, production_device;\n"
        "from voice_studio.services.audio_service import AudioService;\n"
        "from voice_studio.services.profile_service import ProfileService;\n"
        "repo = ProfileRepository(r'{root}');\n"
        "service = ProfileService(repo, AudioService(_validation_ffmpeg_adapter()));\n"
        "spec = service.load_prompt_spec('{uuid}');\n"
        "assert spec.ref_code is not None and spec.ref_spk_embedding is not None;\n"
        "assert spec.ref_text;\n"
        "adapter = RealQwenAdapter(model_path=r'{model}', device=production_device());\n"
        "pcm = adapter.generate(spec, {gen!r}, 24000);\n"
        "assert isinstance(pcm, np.ndarray) and pcm.ndim == 1 and pcm.dtype == np.float32;\n"
        "assert pcm.size > 0 and np.isfinite(pcm).all();\n"
        "assert float(np.abs(pcm).max()) > 1e-4;\n"
        "print('GPU_ROUNDTRIP_OK')\n"
    ).format(src=str(root / "src"), rvh=str(root / "scripts" / "runtime_validation_helpers.py"),
             root=str(profiles_root), uuid=uuid, model=_model_path(), gen=GEN_TEXT)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
    assert "GPU_ROUNDTRIP_OK" in r.stdout, r.stderr


@gpu
def test_e_gpu_memory_peak_evidence(tmp_path):
    """E. 생성 후 torch CUDA peak 메모리 > 0(실제 GPU 사용 증거)."""
    _cuda_or_skip()
    from voice_studio.infra.qwen_adapter import RealQwenAdapter, production_device
    adapter = RealQwenAdapter(model_path=_model_path(), device=production_device())
    spec = adapter.create_prompt(_reference_pcm(), 24000, REF_TEXT)
    pcm = adapter.generate(spec, "짧은 문장입니다.", 24000)
    _assert_sane_pcm(pcm)
    import gc
    gc.collect()
    t = _torch()
    t.cuda.empty_cache(); t.cuda.synchronize()
    assert t.cuda.max_memory_allocated() > 0
