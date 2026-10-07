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


# ---- P14 regression: 무음 가드 + Lanczos-3 리샘플 ----

def test_encode_mp3_guard_padding(tmp_path):
    """encode_mp3는 48kHz로 업샘플 후 앞뒤 20ms(960샘플) 무음 가드를 붙인다."""
    fake = FakeFfmpegAdapter(duration=30.0)
    svc = AudioService(fake)
    pcm = np.zeros(24000, dtype=np.float32)  # 24kHz 1초
    svc.encode_mp3(pcm, 192, str(tmp_path / "g.mp3"))
    bitrate, out, size, sr = fake.mp3_encoded[-1]
    assert bitrate == 192 and out.endswith("g.mp3")
    assert sr == 48000
    # 1s @48k = 48000 + head 20ms(960) + tail 20ms(960)
    assert size == 48000 + 960 + 960


def test_encode_mp3_empty_pcm_stays_empty(tmp_path):
    fake = FakeFfmpegAdapter(duration=30.0)
    svc = AudioService(fake)
    svc.encode_mp3(np.zeros(0, np.float32), 128, str(tmp_path / "e.mp3"))
    _, _, size, sr = fake.mp3_encoded[-1]
    assert size == 0 and sr == 48000


def test_lanczos_preserves_integer_grid_samples():
    """2:1 업샘플에서 짝수 인덱스는 원본 샘플과 정확히 일치해야 한다(polyphase 계약)."""
    x = np.sin(np.linspace(0, 2 * np.pi, 240)).astype(np.float32)
    y = _resample_linear(x, 24000, 48000)
    assert len(y) == 480
    np.testing.assert_allclose(y[::2], x, atol=1e-6)


def test_lanczos_preserves_dc():
    x = np.full(100, 0.75, dtype=np.float32)
    y = _resample_linear(x, 24000, 48000)
    np.testing.assert_allclose(y, 0.75, atol=1e-6)


def test_lanczos_half_step_is_not_linear_average():
    """Lanczos-3의 절반 위치 값은 2-tap 선형(0.5)과 달라야 한다(차수 향상 확인).

    나이퀴스트 직전 2-샘플 주기 패턴에서 선형은 0.5, Lanczos-3은 6-tap
    가중치 때문에 0.364 부근으로 나온다.
    """
    x = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32)
    y = _resample_linear(x, 24000, 48000)
    assert len(y) == 16
    assert abs(float(y[1]) - 0.5) > 1e-2
    assert abs(float(y[1]) - 0.364) < 0.01
