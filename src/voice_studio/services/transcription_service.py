"""자동 받아쓰기 서비스. faster-whisper small/int8, CPU 전용, 지연 로드."""

from __future__ import annotations
from typing import Protocol
import numpy as np
from ..core import config

class Transcriber(Protocol):
    """음성→텍스트 인터페이스. 구현을 교체 가능하게 분리한다."""

    model_id: str

    def transcribe(self, pcm: np.ndarray, sample_rate: int) -> str: ...

    def is_ready(self) -> bool: ...

    def ensure_loaded(self, progress_cb=None) -> None: ...

class FasterWhisperTranscriber:
    """CPU 기반 faster-whisper 구현. 첫 사용 시 모델을 내려받는다(설치 크기 억제)."""

    def __init__(self, model_id: str = config.STT_MODEL_ID, compute_type: str = config.STT_COMPUTE_TYPE):
        self.model_id = model_id
        self.compute_type = compute_type
        self._model = None

    def is_ready(self) -> bool:
        return self._model is not None

    def ensure_loaded(self, progress_cb=None) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel  # 지연 import: 메인 프로세스는 GPU/CUDA와 무관
        self._model = WhisperModel(self.model_id, device="cpu", compute_type=self.compute_type)

    def transcribe(self, pcm: np.ndarray, sample_rate: int) -> str:
        if pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32)
        self.ensure_loaded()
        segments, _info = self._model.transcribe(pcm, language="ko", beam_size=1)
        return " ".join(seg.text.strip() for seg in segments).strip()

class FakeTranscriber:
    """테스트용 가짜 받아쓰기. 미리 정한 텍스트를 반환한다."""

    def __init__(self, text: str = "가짜 받아쓰기 결과입니다."):
        self.model_id = "fake"
        self._text = text

    def is_ready(self) -> bool:
        return True

    def ensure_loaded(self, progress_cb=None) -> None:
        pass

    def transcribe(self, pcm: np.ndarray, sample_rate: int) -> str:
        return self._text
