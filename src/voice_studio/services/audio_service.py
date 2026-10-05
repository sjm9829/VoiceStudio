"""오디오 서비스. FFmpeg 어댑터를 UI/도메인 뒤로 숨긴다."""

from __future__ import annotations
from typing import Protocol
import numpy as np
from ..core import config
from ..core.errors import UnsupportedAudioError, FfmpegNotFoundError
from ..infra.ffmpeg_adapter import FfmpegAdapter, RealFfmpegAdapter, FakeFfmpegAdapter, SUPPORTED_SUFFIXES

class AudioService:
    def __init__(self, adapter: FfmpegAdapter | None = None):
        self.adapter = adapter

    def _adapter(self) -> FfmpegAdapter:
        if self.adapter is None:
            try:
                self.adapter = RealFfmpegAdapter()
            except FfmpegNotFoundError:
                raise
        return self.adapter

    @staticmethod
    def is_supported(path: str) -> bool:
        return path.lower().endswith(SUPPORTED_SUFFIXES)

    def probe(self, path: str) -> dict:
        if not self.is_supported(path):
            raise UnsupportedAudioError(f"지원하지 않는 형식: {path}")
        return self._adapter().probe(path)

    def waveform(self, path: str, buckets: int = 800) -> list[float]:
        """전체 파일의 peak envelope. 긴 파일에서도 버킷 수가 고정되어 UI가 멈추지 않는다."""
        return self._adapter().waveform(path, buckets)

    def decode_reference_segment(self, path: str, start_s: float, end_s: float) -> np.ndarray:
        """선택 구간을 24kHz mono float32로 디코딩해 worker에 제공한다."""
        if end_s - start_s < config.REFERENCE_MIN_SECONDS:
            raise UnsupportedAudioError(
                f"선택 구간이 너무 짧습니다. {config.REFERENCE_MIN_SECONDS:.0f}초 이상 선택해 주세요.")
        return self._adapter().decode_segment(path, start_s, end_s, config.REFERENCE_SAMPLE_RATE)

    def save_reference_flac(self, path: str, start_s: float, end_s: float, out_flac: str) -> str:
        """선택 구간을 24kHz mono lossless FLAC으로 보관한다(원본 전체 복사 없음)."""
        return self._adapter().decode_segment_to_flac(path, start_s, end_s, out_flac)

    def encode_mp3(self, pcm: np.ndarray, bitrate_kbps: int, out_path: str) -> str:
        """Qwen 24kHz 출력을 CPU에서 이어붙인 뒤 48kHz mono MP3로 한 번만 인코딩한다."""
        if bitrate_kbps not in config.MP3_BITRATE_CHOICES:
            raise ValueError(f"지원하지 않는 음질: {bitrate_kbps}")
        up = _resample_linear(pcm, config.REFERENCE_SAMPLE_RATE, config.OUTPUT_SAMPLE_RATE)
        return self._adapter().encode_mp3(up, config.OUTPUT_SAMPLE_RATE, bitrate_kbps, out_path)

def _resample_linear(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or pcm.size == 0:
        return pcm.astype(np.float32)
    duration = pcm.size / src_rate
    n_out = int(duration * dst_rate)
    src_t = np.arange(pcm.size, dtype=np.float64) / src_rate
    dst_t = np.arange(n_out, dtype=np.float64) / dst_rate
    return np.interp(dst_t, src_t, pcm.astype(np.float64)).astype(np.float32)

def concat_pcm(chunks: list[np.ndarray], sample_rate: int, gap_ms: int, paragraph_gap_ms: int, gap_flags: list[bool]) -> np.ndarray:
    """chunk PCM을 무음 간격과 함께 CPU에서 이어붙인다. gap_flags[i]=True면 문단 경계 간격."""
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    parts: list[np.ndarray] = [chunks[0]]
    for i, chunk in enumerate(chunks[1:], start=1):
        ms = paragraph_gap_ms if gap_flags[i] else gap_ms
        parts.append(np.zeros(int(sample_rate * ms / 1000), dtype=np.float32))
        parts.append(chunk)
    return np.concatenate(parts).astype(np.float32)
