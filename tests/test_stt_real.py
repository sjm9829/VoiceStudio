"""STT 실제 런타임 opt-in 테스트(P13 §12-13).

`pytest -m stt`로 실행. faster-whisper가 설치된 환경에서 실제 small/int8 CPU
모델을 로드하고(필요 시 다운로드) transcribe 경로를 실행한다. 무음/사인파
입력은 인식 결과가 비어도 정상이며, 이 테스트의 목적은 예외 없이 로드→추론이
완료되는 것(설치본에서 실패하는 경로의 실기 재현)이다.

실행: uv run pytest -q tests/test_stt_real.py -m stt
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("faster_whisper", reason="faster-whisper 미설치 환경")
stt = pytest.mark.stt


def _pcm(seconds=2.0, rate=24000):
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


@stt
def test_real_faster_whisper_load_and_transcribe():
    """실제 WhisperModel(small/int8, CPU) 로드 → 한국어 transcribe 실행.

    설치본에서 자동 받아쓰기가 실패하는 지점은 (1) 모델 다운로드(console=False
    환경의 hub 진행바)와 (2) transcribe 경로다. 이 테스트는 두 경로 모두를
    CPU에서 실제로 실행해 예외가 없음을 확인한다.
    """
    from voice_studio.services.transcription_service import FasterWhisperTranscriber
    transcriber = FasterWhisperTranscriber()
    assert not transcriber.is_ready()
    transcriber.ensure_loaded()
    assert transcriber.is_ready()
    text = transcriber.transcribe(_pcm(), 24000)
    assert isinstance(text, str)  # 사인파 입력: 결과가 비어도 예외만 없으면 충분


@stt
def test_real_faster_whisper_resample_paths():
    """24k → 16k 리샘플 경로가 실제 transcribe 입력 계약을 만족한다(float32 1-D 16k)."""
    from voice_studio.services.transcription_service import _resample_to_16k
    pcm = _pcm(1.0)
    out = _resample_to_16k(pcm, 24000)
    assert out.dtype == np.float32 and out.ndim == 1
    assert abs(out.size - 16000) <= 2
