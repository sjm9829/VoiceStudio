"""P13 검증 전용 공통 helper(production runtime 오염 금지).

source/build validation은 시스템 PATH의 임의 FFmpeg에 의존하지 않고
repository의 third_party/bin binary를 명시적으로 사용한다(P13 §4, §6, §7).
production frozen RealFfmpegAdapter 정책(bundled → PATH)은 변경하지 않는다.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_BIN_DIR = ROOT / "third_party" / "bin"


class ValidationFfmpegMissing(RuntimeError):
    """third_party/bin의 FFmpeg pair가 준비되지 않았다."""


def validation_ffmpeg_paths(bin_dir: Path = REPO_BIN_DIR) -> tuple[str, str]:
    """repository third_party/bin의 ffmpeg.exe/ffprobe.exe 경로를 돌려준다.

    없으면 prepare_ffmpeg.py 실행 안내와 함께 실패한다. PATH를 절대 fallback으로
    쓰지 않으므로 PATH에 ffmpeg가 전혀 없어도 third_party/bin만 있으면 동작한다.
    """
    ffmpeg = bin_dir / "ffmpeg.exe"
    ffprobe = bin_dir / "ffprobe.exe"
    missing = [str(p) for p in (ffmpeg, ffprobe) if not p.is_file()]
    if missing:
        raise ValidationFfmpegMissing(
            "FFmpeg validation binaries missing: "
            + ", ".join(missing)
            + ". Run: python scripts/prepare_ffmpeg.py")
    return str(ffmpeg), str(ffprobe)


def make_validation_adapter(bin_dir: Path = REPO_BIN_DIR):
    """third_party/bin binary를 명시적으로 쓰는 RealFfmpegAdapter를 만든다."""
    from voice_studio.infra.ffmpeg_adapter import RealFfmpegAdapter

    ffmpeg, ffprobe = validation_ffmpeg_paths(bin_dir)
    return RealFfmpegAdapter(ffmpeg=ffmpeg, ffprobe=ffprobe)
