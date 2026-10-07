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
        if end_s - start_s < config.REFERENCE_APP_MIN_SECONDS:
            raise UnsupportedAudioError(
                f"선택 구간이 너무 짧습니다. {config.REFERENCE_APP_MIN_SECONDS:.0f}초 이상 선택해 주세요.")
        return self._adapter().decode_segment(path, start_s, end_s, config.REFERENCE_SAMPLE_RATE)

    def decode_preview_segment(self, path: str, start_s: float, end_s: float) -> np.ndarray:
        """미리 듣기용 선택 구간 디코딩(24kHz mono float32). 최소 길이 제한은 적용하지 않는다."""
        return self._adapter().decode_segment(path, start_s, end_s, config.REFERENCE_SAMPLE_RATE)

    def encode_wav(self, pcm: np.ndarray, out_path: str) -> str:
        """미리 듣기용 임시 WAV 인코딩(24kHz mono)."""
        return self._adapter().encode_wav(pcm, config.REFERENCE_SAMPLE_RATE, out_path)

    def save_reference_flac(self, path: str, start_s: float, end_s: float, out_flac: str) -> str:
        """선택 구간을 24kHz mono lossless FLAC으로 보관한다(원본 전체 복사 없음)."""
        return self._adapter().decode_segment_to_flac(path, start_s, end_s, out_flac)

    def encode_mp3(self, pcm: np.ndarray, bitrate_kbps: int, out_path: str) -> str:
        """Qwen 24kHz 출력을 CPU에서 이어붙인 뒤 48kHz mono MP3로 한 번만 인코딩한다.

        P14: 인코딩 직전 앞뒤에 20ms 무음 가드를 붙인다. LAME(libmp3lame)의
        head encoder delay(48kHz에서 576샘플 ≈ 12ms)와 마지막 frame padding이
        가드 무음으로 흡수되어, 첫 음절의 attack과 마지막 음절의 release가
        잘리지 않고 남는다.
        """
        if bitrate_kbps not in config.MP3_BITRATE_CHOICES:
            raise ValueError(f"지원하지 않는 음질: {bitrate_kbps}")
        up = _resample_linear(pcm, config.REFERENCE_SAMPLE_RATE, config.OUTPUT_SAMPLE_RATE)
        up = _pad_guard(up, config.OUTPUT_SAMPLE_RATE)
        return self._adapter().encode_mp3(up, config.OUTPUT_SAMPLE_RATE, bitrate_kbps, out_path)

# 무음 가드 길이(ms). LAME head encoder delay 576샘플(≈12ms @48kHz)보다 긴 20ms.
GUARD_MS = 20


def _pad_guard(pcm: np.ndarray, sample_rate: int) -> np.ndarray:
    """인코딩 대상 PCM 앞뒤에 GUARD_MS만큼 무음을 붙인다(P14: 앞/뒤 잘림 완화)."""
    if pcm.size == 0:
        return pcm
    n = int(sample_rate * GUARD_MS / 1000)
    return np.concatenate([np.zeros(n, dtype=np.float32), pcm,
                           np.zeros(n, dtype=np.float32)]).astype(np.float32)


def _lanczos3(a: np.ndarray) -> np.ndarray:
    """Lanczos-3 windowed-sinc 커널(정규화 sinc: sin(pi a)/(pi a)). |a|>=3이면 0."""
    a = np.abs(a)
    w = np.zeros(a.shape, dtype=np.float64)
    mask = (a > 0) & (a < 3)
    an = a[mask]
    w[mask] = (np.sin(np.pi * an) / (np.pi * an)) * (np.sin(np.pi * an / 3) / (np.pi * an / 3))
    w[a == 0] = 1.0
    return w


def _resample_linear(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Lanczos-3 창 sinc 리샘플(P14: 2-tap 선형 → 6-tap polyphase).

    이름은 하위 호환(테스트 import)을 위해 유지한다. 출력 길이
    n_out = int(size/src_rate*dst_rate)는 기존 np.interp 버전과 동일한 계약이다.
    무음(0)과 DC(상수) 입력은 가중치 합 1 정규화로 정확히 보존된다.
    """
    if src_rate == dst_rate or pcm.size == 0:
        return pcm.astype(np.float32)
    duration = pcm.size / src_rate
    n_out = int(duration * dst_rate)
    x = pcm.astype(np.float64)
    p = np.arange(n_out, dtype=np.float64) * (src_rate / dst_rate)
    base = np.floor(p).astype(np.int64)
    n_in = x.size
    cols = []
    weights = []
    for off in range(-2, 4):  # 6-tap window: base-2 .. base+3
        raw = base + off
        w = _lanczos3(p - raw.astype(np.float64))
        cols.append(np.clip(raw, 0, n_in - 1))
        weights.append(w)
    idx = np.stack(cols, axis=1)
    wmat = np.stack(weights, axis=1)
    total = wmat.sum(axis=1, keepdims=True)
    total[total == 0] = 1.0
    wmat = wmat / total
    out = (x[idx] * wmat).sum(axis=1)
    return out.astype(np.float32)

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
