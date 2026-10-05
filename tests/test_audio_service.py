import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pytest
from voice_studio.services.audio_service import AudioService, concat_pcm, _resample_linear
from voice_studio.core.errors import UnsupportedAudioError
from voice_studio.infra.ffmpeg_adapter import FakeFfmpegAdapter

def service():
    return AudioService(FakeFfmpegAdapter(duration=30.0))

def test_probe_supported():
    info = service().probe("a.mp3")
    assert info["duration"] == 30.0

def test_probe_unsupported_suffix():
    with pytest.raises(UnsupportedAudioError):
        service().probe("a.ogg")

def test_decode_reference_segment_24k_mono_f32():
    pcm = service().decode_reference_segment("a.mp3", 1.0, 16.0)
    assert pcm.dtype == np.float32
    assert abs(len(pcm) / 24000 - 15.0) < 0.01

def test_too_short_selection_rejected():
    with pytest.raises(UnsupportedAudioError):
        service().decode_reference_segment("a.mp3", 1.0, 2.0)

def test_waveform_buckets_fixed():
    peaks = service().waveform("a.mp3", buckets=400)
    assert len(peaks) == 400
    assert all(0.0 <= p <= 1.0 for p in peaks)

def test_encode_mp3_output_contract(tmp_path):
    pcm = np.zeros(24000, dtype=np.float32)
    out = service().encode_mp3(pcm, 192, str(tmp_path / "out.mp3"))
    assert out.endswith(".mp3") and (tmp_path / "out.mp3").is_file()

def test_invalid_bitrate_rejected(tmp_path):
    with pytest.raises(ValueError):
        service().encode_mp3(np.zeros(10, np.float32), 320, str(tmp_path / "x.mp3"))

def test_concat_with_gaps():
    sr = 24000
    a = np.ones(sr, np.float32); b = np.full(sr, 0.5, np.float32)
    pcm = concat_pcm([a, b], sr, gap_ms=180, paragraph_gap_ms=320, gap_flags=[False, True])
    # 1초 + 320ms 무음 + 1초
    assert abs(len(pcm) / sr - (2.0 + 0.32)) < 0.01

def test_resample_24k_to_48k():
    x = np.sin(np.linspace(0, 2 * np.pi, 240)).astype(np.float32)
    y = _resample_linear(x, 24000, 48000)
    assert len(y) == 2 * len(x)
