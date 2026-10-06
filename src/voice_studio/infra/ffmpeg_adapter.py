"""FFmpeg/ffprobe 어댑터. 모든 ffmpeg 호출을 이 모듈 뒤로 은닉한다.

AudioService는 이 인터페이스(FfmpegAdapter 프로토콜)만 의존하므로 테스트에서 FakeFfmpegAdapter로 치환한다.
"""

from __future__ import annotations
import json, shutil, subprocess
from pathlib import Path
from typing import Protocol
import numpy as np
from ..core.errors import FfmpegNotFoundError, UnsupportedAudioError

SUPPORTED_SUFFIXES = (".mp3", ".m4a", ".wav", ".flac")

class FfmpegAdapter(Protocol):
    def probe(self, path: str) -> dict: ...
    def decode_segment(self, path: str, start_s: float, end_s: float, sample_rate: int) -> np.ndarray: ...
    def waveform(self, path: str, buckets: int) -> list[float]: ...
    def encode_mp3(self, pcm: np.ndarray, sample_rate: int, bitrate_kbps: int, out_path: str) -> str: ...
    def decode_segment_to_flac(self, path: str, start_s: float, end_s: float, out_flac: str) -> str: ...
    def encode_wav(self, pcm: np.ndarray, sample_rate: int, out_path: str) -> str: ...

def bundled_bin_dirs() -> "list[Path]":
    r"""앱 설치 디렉터리의 bin 후보 폴더들(P12.2-05).

    PyInstaller onedir(6.x)에서 datas는 <설치 폴더>\_internal 아래로 들어가므로
    VoiceStudio.exe 옆 bin/과 _internal/bin/ 둘 다 확인한다.
    """
    import sys
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        return [exe_dir / "bin", exe_dir / "_internal" / "bin"]
    return []

def bundled_bin_dir() -> "Path | None":
    """개발 환경에서는 None. frozen에서는 첫 후보(exe 옆 bin)를 돌려준다(하위 호환)."""
    dirs = bundled_bin_dirs()
    return dirs[0] if dirs else None

def _resolve_binary(name: str, override: str | None) -> str | None:
    """탐색 우선순위: 1) 앱 설치 디렉터리에 포함된 바이너리, 2) 시스템 PATH."""
    if override:
        return override
    for bundled in bundled_bin_dirs():
        candidate = bundled / f"{name}.exe"
        if candidate.is_file():
            return str(candidate)
    return shutil.which(name)

class RealFfmpegAdapter:
    """실제 ffmpeg/ffprobe 바이너리를 사용하는 어댑터.

    최종 사용자는 FFmpeg를 별도 설치하지 않아도 된다: 설치 프로그램이 앱 디렉터리의
    bin/ 아래에 ffmpeg.exe/ffprobe.exe를 포함하고, 이 어댑터가 그것을 먼저 찾는다(P12.1-11).
    """

    def __init__(self, ffmpeg: str | None = None, ffprobe: str | None = None):
        self.ffmpeg = _resolve_binary("ffmpeg", ffmpeg)
        self.ffprobe = _resolve_binary("ffprobe", ffprobe)
        if not self.ffmpeg or not self.ffprobe:
            raise FfmpegNotFoundError(f"ffmpeg={self.ffmpeg} ffprobe={self.ffprobe}")

    def available(self) -> bool:
        """ffmpeg/ffprobe를 찾았는지 여부(진단용, raise 대신 bool, P12.2-22)."""
        return bool(self.ffmpeg) and bool(self.ffprobe)

    def probe_version(self) -> str:
        """ffmpeg 버전 문자열(진단용). 실패 시 FfmpegNotFoundError."""
        r = self._run([self.ffmpeg, "-version"])
        if r.returncode != 0:
            raise FfmpegNotFoundError("ffmpeg -version 실패")
        return (r.stdout.splitlines() or [""])[0]

    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(args, capture_output=True, text=True, timeout=300)

    def probe(self, path: str) -> dict:
        if not path.lower().endswith(SUPPORTED_SUFFIXES):
            raise UnsupportedAudioError(f"unsupported suffix: {path}")
        r = self._run([self.ffprobe, "-v", "error", "-print_format", "json",
                       "-show_format", "-show_streams", path])
        if r.returncode != 0:
            raise UnsupportedAudioError(r.stderr.strip()[:300])
        data = json.loads(r.stdout)
        fmt = data.get("format", {})
        audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
        if audio is None:
            raise UnsupportedAudioError("no audio stream")
        return {
            "duration": float(fmt.get("duration") or audio.get("duration") or 0.0),
            "format_name": fmt.get("format_name", ""),
            "sample_rate": int(audio.get("sample_rate", 0)),
            "channels": int(audio.get("channels", 0)),
            "codec": audio.get("codec_name", ""),
        }

    @staticmethod
    def _validate_segment_range(start_s: float, end_s: float) -> float:
        """절대 구간(start/end)을 FFmpeg용 duration으로 변환하고 유효성을 검증한다(P13).

        호출자(UI/domain)는 절대 시각을 유지하고, FFmpeg에는 -ss(시작) + -t(길이)로만 전달한다.
        -to는 절대 종료 시각 해석이 입력 파일마다 달라 Windows FFmpeg에서
        "-to value smaller than -ss" 오류를 낼 수 있으므로 사용하지 않는다.
        """
        start_s = float(start_s)
        duration_s = float(end_s) - start_s
        if start_s < 0 or duration_s <= 0:
            raise UnsupportedAudioError(
                f"잘못된 선택 구간: start={start_s:.3f}s end={float(end_s):.3f}s")
        return duration_s

    def decode_segment(self, path: str, start_s: float, end_s: float, sample_rate: int) -> np.ndarray:
        duration_s = self._validate_segment_range(start_s, end_s)
        args = [self.ffmpeg, "-v", "error", "-ss", f"{start_s:.3f}", "-i", path,
                "-t", f"{duration_s:.3f}",
                "-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "-"]
        r = subprocess.run(args, capture_output=True, timeout=300)
        if r.returncode != 0:
            raise UnsupportedAudioError(r.stderr.decode(errors="replace")[:300])
        return np.frombuffer(r.stdout, dtype=np.float32)

    def decode_segment_to_flac(self, path: str, start_s: float, end_s: float, out_flac: str) -> str:
        duration_s = self._validate_segment_range(start_s, end_s)
        args = [self.ffmpeg, "-y", "-v", "error", "-ss", f"{start_s:.3f}", "-i", path,
                "-t", f"{duration_s:.3f}",
                "-ac", "1", "-ar", "24000", "-sample_fmt", "s32", out_flac]
        r = self._run(args)
        if r.returncode != 0:
            raise UnsupportedAudioError(r.stderr.strip()[:300])
        return out_flac

    def waveform(self, path: str, buckets: int) -> list[float]:
        """전체 파일을 저해상도 mono 8kHz f32로 디코딩해 peak envelope 버킷으로 축소."""
        args = [self.ffmpeg, "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", "8000", "-"]
        r = subprocess.run(args, capture_output=True, timeout=600)
        if r.returncode != 0:
            raise UnsupportedAudioError(r.stderr.decode(errors="replace")[:300])
        data = np.frombuffer(r.stdout, dtype=np.float32)
        if data.size == 0:
            return [0.0] * buckets
        idx = np.linspace(0, data.size, buckets + 1).astype(int)
        peaks = [float(np.max(np.abs(data[idx[i]:idx[i+1]]))) if idx[i+1] > idx[i] else 0.0
                 for i in range(buckets)]
        return peaks

    def encode_wav(self, pcm: np.ndarray, sample_rate: int, out_path: str) -> str:
        """미리 듣기용 임시 WAV(24kHz mono) 인코딩."""
        raw = np.clip(pcm, -1.0, 1.0).astype(np.float32).tobytes()
        args = [self.ffmpeg, "-y", "-v", "error", "-f", "f32le", "-ar", str(sample_rate),
                "-ac", "1", "-i", "-", "-codec:a", "pcm_s16le", out_path]
        r = subprocess.run(args, input=raw, capture_output=True, timeout=300)
        if r.returncode != 0:
            raise UnsupportedAudioError(r.stderr.decode(errors="replace")[:300])
        return out_path

    def encode_mp3(self, pcm: np.ndarray, sample_rate: int, bitrate_kbps: int, out_path: str) -> str:
        raw = np.clip(pcm, -1.0, 1.0).astype(np.float32).tobytes()
        args = [self.ffmpeg, "-y", "-v", "error", "-f", "f32le", "-ar", str(sample_rate),
                "-ac", "1", "-i", "-", "-codec:a", "libmp3lame", "-b:a", f"{bitrate_kbps}k",
                "-ac", "1", out_path]
        r = subprocess.run(args, input=raw, capture_output=True, timeout=600)
        if r.returncode != 0:
            raise UnsupportedAudioError(r.stderr.decode(errors="replace")[:300])
        return out_path

class FakeFfmpegAdapter:
    """ffmpeg 미설치 환경(개발/테스트)용 가짜 어댑터. 사인파를 합성한다."""

    def __init__(self, duration: float = 10.0, sample_rate: int = 24000):
        self.duration = duration
        self.sample_rate = sample_rate
        self.mp3_encoded: list[tuple[int, str]] = []

    def probe(self, path: str) -> dict:
        if not path.lower().endswith(SUPPORTED_SUFFIXES):
            raise UnsupportedAudioError(f"unsupported suffix: {path}")
        return {"duration": self.duration, "format_name": "fake", "sample_rate": self.sample_rate,
                "channels": 1, "codec": "pcm_f32le"}

    def decode_segment(self, path: str, start_s: float, end_s: float, sample_rate: int) -> np.ndarray:
        if end_s <= start_s or start_s < 0 or end_s > self.duration:
            raise UnsupportedAudioError(f"bad range {start_s}-{end_s}")
        t = np.arange(int((end_s - start_s) * sample_rate), dtype=np.float32) / sample_rate
        return 0.25 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)

    def decode_segment_to_flac(self, path: str, start_s: float, end_s: float, out_flac: str) -> str:
        pcm = self.decode_segment(path, start_s, end_s, 24000)
        np.save(out_flac + ".npy", pcm)
        with open(out_flac, "wb") as fh:
            fh.write(b"FAKEFLAC" + len(pcm).to_bytes(8, "little"))
        return out_flac

    def waveform(self, path: str, buckets: int) -> list[float]:
        n = max(1, int(self.duration * 8))
        data = 0.25 * np.sin(2 * np.pi * 220.0 * np.arange(n) / 8).astype(np.float32)
        idx = np.linspace(0, data.size, buckets + 1).astype(int)
        return [float(np.max(np.abs(data[idx[i]:idx[i+1]]))) if idx[i+1] > idx[i] else 0.0
                for i in range(buckets)]

    def encode_wav(self, pcm: np.ndarray, sample_rate: int, out_path: str) -> str:
        with open(out_path, "wb") as fh:
            fh.write(b"FAKEWAV" + len(pcm).to_bytes(8, "little"))
        return out_path

    def encode_mp3(self, pcm: np.ndarray, sample_rate: int, bitrate_kbps: int, out_path: str) -> str:
        self.mp3_encoded.append((bitrate_kbps, out_path))
        with open(out_path, "wb") as fh:
            fh.write(b"FAKEMP3" + len(pcm).to_bytes(8, "little"))
        return out_path
