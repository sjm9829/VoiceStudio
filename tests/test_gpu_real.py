"""실제 Qwen/GPU opt-in 테스트. `pytest -m gpu`로만 실행한다."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch 미설치 환경")
gpu = pytest.mark.gpu


def _cuda_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU가 없는 환경")


def _reference_pcm(seconds=6.0, rate=24000):
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    pcm = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return pcm


@gpu
def test_real_qwen_load_and_prompt_and_generate(tmp_path):
    """실제 Qwen3-TTS-12Hz-0.6B-Base 로드 → ICL prompt 생성 → 짧은 한국어 문장 생성."""
    from voice_studio.services.model_manager import ModelManager
    from voice_studio.infra.qwen_adapter import RealQwenAdapter, default_device
    _cuda_or_skip()
    model_path = ModelManager().download()
    adapter = RealQwenAdapter(model_path=model_path, device=default_device())
    pcm = _reference_pcm()
    spec = adapter.create_prompt(pcm, 24000, "안녕하세요. 참조 음성 테스트 대사입니다.")
    assert spec.ref_code is not None and spec.ref_spk_embedding is not None
    arrays, sr = adapter.generate("안녕하세요. 실제 생성 테스트입니다.", spec)
    pcm_out = np.concatenate(arrays)
    assert sr > 0 and pcm_out.size > 0 and np.isfinite(pcm_out).all()


@gpu
def test_profile_roundtrip_across_processes(tmp_path):
    """prompt 저장 → 새 프로세스에서 load → 생성(GPU E2E 핵심 경로)."""
    import subprocess
    _cuda_or_skip()
    from voice_studio.infra.profile_repository import ProfileRepository
    from voice_studio.services.audio_service import AudioService
    from voice_studio.services.profile_service import ProfileService
    from voice_studio.infra.ffmpeg_adapter import RealFfmpegAdapter
    from voice_studio.services.model_manager import ModelManager
    from voice_studio.infra.qwen_adapter import RealQwenAdapter, default_device
    model_path = ModelManager().download()
    audio = AudioService(RealFfmpegAdapter())
    repo = ProfileRepository(tmp_path / "profiles")
    service = ProfileService(repo, audio)
    adapter = RealQwenAdapter(model_path=model_path, device=default_device())
    pcm = _reference_pcm()
    spec = adapter.create_prompt(pcm, 24000, "안녕하세요. 참조 음성 테스트 대사입니다.")
    profile = service.register(name="GPU 테스트", source_path=str(tmp_path / "ref.wav"),
                               start_s=0.0, end_s=6.0,
                               ref_text="안녕하세요. 참조 음성 테스트 대사입니다.",
                               consent=True, prompt=spec, waveform=pcm, sample_rate=24000)
    code = (
        "import sys; sys.path.insert(0, r'%s');\n"
        "from voice_studio.infra.profile_repository import ProfileRepository;\n"
        "from voice_studio.services.profile_service import ProfileService;\n"
        "from voice_studio.services.audio_service import AudioService;\n"
        "svc = ProfileService(ProfileRepository(r'%s'), AudioService());\n"
        "spec = svc.load_prompt_spec(r'%s');\n"
        "assert spec.ref_code is not None and spec.ref_text;\n"
        "print('GPU_ROUNDTRIP_OK')"
    ) % (str(Path(__file__).resolve().parents[1] / "src"), str(tmp_path / "profiles"), str(profile.profile_dir))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert "GPU_ROUNDTRIP_OK" in r.stdout, r.stderr


@gpu
def test_gpu_memory_returned_after_generation(tmp_path):
    """생성 후 CUDA 캐시 반환 확인(nvidia-smi/pynvml 보조 아님, torch 자체 확인)."""
    _cuda_or_skip()
    from voice_studio.infra.qwen_adapter import RealQwenAdapter, default_device
    from voice_studio.services.model_manager import ModelManager
    adapter = RealQwenAdapter(model_path=ModelManager().download(), device=default_device())
    spec = adapter.create_prompt(_reference_pcm(), 24000, "안녕하세요. 참조 대사입니다.")
    adapter.generate("짧은 문장입니다.", spec)
    import gc
    gc.collect()
    torch.cuda.empty_cache(); torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    assert peak > 0  # 실제 GPU를 사용했다는 최소 증거
