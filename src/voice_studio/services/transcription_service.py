"""자동 받아쓰기 서비스. faster-whisper small/int8, CPU 전용, 지연 로드."""

from __future__ import annotations
from typing import Protocol
import logging, os
import numpy as np

_log = logging.getLogger(__name__)
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
        """모델 로드/다운로드. 설치본(console=False)에서의 실패를 로그로 진단한다(P13 §12).

        - faster-whisper 모델 다운로드는 huggingface_hub를 거친다. PyInstaller
          console=False(windowed) 설치본에서 sys.stdout/sys.stderr가 None이면
          hub의 tqdm 진행바가 'NoneType' object has no attribute 'write'로
          실패한다. 모델 매니저(P13 hotfix)와 동일하게 진행바를 비활성화한다.
        - 실패 시 정확한 exception을 traceback째 파일 로그에 기록한 뒤 그대로
          재raise한다. UI에는 friendly message만 나가고 진단 근거는 로그에 남는다.
        """
        if self._model is not None:
            return
        try:
            from huggingface_hub.utils import disable_progress_bars
            disable_progress_bars()
        except Exception:  # pragma: no cover - 구버전 hub fallback
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        _log.info("STT 모델 로드 시작: model=%s compute=%s", self.model_id, self.compute_type)
        try:
            from faster_whisper import WhisperModel  # 지연 import: 메인 프로세스는 GPU/CUDA와 무관
            self._model = WhisperModel(self.model_id, device="cpu", compute_type=self.compute_type)
        except Exception:
            _log.exception("STT 모델 로드 실패: model=%s compute=%s", self.model_id, self.compute_type)
            raise
        _log.info("STT 모델 로드 완료: model=%s", self.model_id)

    def transcribe(self, pcm: np.ndarray, sample_rate: int) -> str:
        """선택 구간 PCM을 받아 텍스트로 변환한다.

        faster-whisper는 ndarray 입력 시 float32 mono 16kHz를 기대하므로
        어떤 입력 샘플레이트가 들어와도 16kHz로 변환한 뒤 전달한다.
        """
        if pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32)
        pcm = _resample_to_16k(pcm, sample_rate)
        self.ensure_loaded()  # 첫 사용 시 모델 다운로드(오프라인이면 오류가 올라간다)
        try:
            segments, _info = self._model.transcribe(pcm, language="ko", beam_size=1)
        except Exception:
            _log.exception("STT transcribe 실패: model=%s sample_rate_in=%s pcm=%s",
                           self.model_id, sample_rate, pcm.shape)
            raise
        return " ".join(seg.text.strip() for seg in segments).strip()

def _resample_to_16k(pcm: np.ndarray, sample_rate: int) -> np.ndarray:
    target = 16000
    if int(sample_rate) == target:
        return np.ascontiguousarray(pcm, dtype=np.float32)
    duration = pcm.size / float(sample_rate)
    n_out = max(1, int(duration * target))
    src_t = np.arange(pcm.size, dtype=np.float64) / sample_rate
    dst_t = np.arange(n_out, dtype=np.float64) / target
    return np.interp(dst_t, src_t, pcm.astype(np.float64)).astype(np.float32)

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
