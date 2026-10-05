"""실제 FFmpeg 계약 테스트(선택적). ffmpeg가 없는 환경에서는 skip."""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

ffmpeg = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(ffmpeg is None, reason="실제 ffmpeg가 없는 환경")


def _service(tmp_path):
    from voice_studio.services.audio_service import AudioService
    return AudioService()


@pytest.fixture
def tones(tmp_path):
    """ffmpeg로 3초짜리 440Hz 테스트 톤 파일(mp3/m4a/wav/flac)을 생성한다."""
    from voice_studio.infra.ffmpeg_adapter import RealFfmpegAdapter
    out = {}
    for ext in ("mp3", "m4a", "wav", "flac"):
        path = tmp_path / f"tone_korean test.{ext}"  # 한글 아닌 공백 포함 경로
        subprocess_ok = RealFfmpegAdapter()._run(
            ["-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
             "-ar", "24000", "-ac", "1", str(path)])
        assert subprocess_ok.returncode == 0
        out[ext] = str(path)
    return out


def test_mp3_m4a_wav_flac_decode_to_24k_mono(tones):
    from voice_studio.services.audio_service import AudioService
    svc = AudioService()
    for path in tones.values():
        pcm = svc.decode_reference_segment(path, 0.0, 2.0)
        assert pcm.dtype == np.float32 and pcm.ndim == 1 and pcm.size >= 24000


def test_reference_flac_roundtrip(tones):
    from voice_studio.services.audio_service import AudioService
    svc = AudioService()
    out = str(Path(tones["mp3"]).with_suffix(".ref.flac"))
    svc.save_reference_flac(tones["mp3"], 0.0, 2.0, out)
    pcm = svc.decode_reference_segment(out, 0.0, 1.0)
    assert pcm.dtype == np.float32 and pcm.size >= 12000


def test_pcm_to_mp3_all_bitrates(tmp_path):
    from voice_studio.services.audio_service import AudioService
    from voice_studio.infra.ffmpeg_adapter import RealFfmpegAdapter
    svc = AudioService()
    pcm = np.zeros(24000, dtype=np.float32)
    for kbps in (128, 192, 256):
        out = str(tmp_path / f"out_{kbps}.mp3")
        RealFfmpegAdapter().encode_mp3(pcm, 48000, kbps, out)
        assert Path(out).stat().st_size > 0


def test_korean_path_roundtrip(tmp_path):
    from voice_studio.services.audio_service import AudioService
    from voice_studio.infra.ffmpeg_adapter import RealFfmpegAdapter
    svc = AudioService()
    korean_dir = tmp_path / "한글 폴더"
    korean_dir.mkdir()
    wav = korean_dir / "참조 음성.wav"
    RealFfmpegAdapter()._run(["-y", "-f", "lavfi", "-i", "sine=frequency=300:duration=2",
                              "-ar", "24000", "-ac", "1", str(wav)])
    pcm = svc.decode_reference_segment(str(wav), 0.0, 1.5)
    assert pcm.size > 0
