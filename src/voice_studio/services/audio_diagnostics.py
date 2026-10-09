"""진단용 오디오 캡처(P15).

명시적으로 진단 디렉터리를 전달한 경우에만 동작한다. 기본(None)에서는
어떤 파일도 생성하지 않고, 사용자 프로필이나 결과 파일을 덮어쓰지 않는다.

원본 PCM은 IEEE float32 WAV(포맷 태그 3)로 무손실 저장한다. s16로 저장하면
양자화가 섞여 "원본 vs MP3" 비교가 오염되므로, 리샘플링·정규화·양자화 없이
입력 float32 샘플을 그대로 기록한다.
"""

from __future__ import annotations
import shutil
import struct
from pathlib import Path
import numpy as np


def _write_wav_f32(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    """float32 mono IEEE-float WAV 헤더를 직접 써서 무손실 저장한다."""
    data = np.ascontiguousarray(pcm, dtype=np.float32).tobytes()
    byte_rate = sample_rate * 4
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 3, 1, sample_rate, byte_rate, 4, 32)
    header += b"data" + struct.pack("<I", len(data))
    with open(path, "wb") as fh:
        fh.write(header + data)


def read_wav_f32(path: str | Path) -> tuple[np.ndarray, int]:
    """float32 WAV(포맷 태그 3)를 직접 파싱해 (pcm, sample_rate)로 반환한다(회귀 검증용).

    표준 wave 모듈은 IEEE float 포맷을 읽지 못하므로 헤더를 직접 해석한다.
    """
    data = Path(path).read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    pos, fmt_tag, rate = 12, None, None
    raw = None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = int.from_bytes(data[pos + 4:pos + 8], "little")
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt_tag, ch, rate = int.from_bytes(body[0:2], "little"), int.from_bytes(body[2:4], "little"), int.from_bytes(body[4:8], "little")
            assert fmt_tag == 3 and ch == 1
        elif cid == b"data":
            raw = body
        pos += 8 + size + (size % 2)
    assert raw is not None
    return np.frombuffer(raw, dtype="<f4").copy(), int(rate)


class DiagnosticsCapture:
    """하나의 narration job에 대한 진단 파일 기록기. diagnostics_dir=None이면 no-op."""

    def __init__(self, audio, diagnostics_dir: str | None, job_id: str):
        self.audio = audio
        self.dir = Path(diagnostics_dir) / job_id if diagnostics_dir else None
        self.enabled = self.dir is not None
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)  # 사용자 파일 경로 아래 임의 덮어쓰기 방지: job_id 하위 폴더만 사용

    def save_qwen_raw(self, pcm: np.ndarray) -> None:
        if self.enabled:
            _write_wav_f32(self.dir / "01_qwen_raw_24k.wav", pcm, 24000)

    def save_pre_encode(self, pcm: np.ndarray) -> None:
        if self.enabled:
            _write_wav_f32(self.dir / "02_pre_encode.wav", pcm, 24000)

    def save_final(self, mp3_path: str) -> None:
        if self.enabled:
            shutil.copyfile(mp3_path, self.dir / "03_final.mp3")

    def save_mp3_decoded(self, mp3_path: str) -> None:
        """최종 MP3를 48kHz WAV로 재디코딩해 저장한다(ffmpeg 있는 실기에서만 유효)."""
        if not self.enabled:
            return
        info = self.audio.probe(mp3_path)
        pcm = self.audio._adapter().decode_segment(mp3_path, 0.0, float(info["duration"]), 48000)
        _write_wav_f32(self.dir / "04_mp3_decoded.wav", pcm, 48000)
