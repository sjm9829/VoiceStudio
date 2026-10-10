"""보이스 스튜디오 진입점.

일반 실행: QApplication/MainWindow
worker 실행: `--worker job.json` → QApplication 없이 worker_main.main 실행 후 종료.
PyInstaller frozen에서 sys.executable이 VoiceStudio.exe이므로 같은 진입점을 공유한다.
"""

from __future__ import annotations
import sys
from pathlib import Path


def _run_worker(argv: list[str]) -> int:
    from voice_studio.workers.worker_main import main as worker_main
    return worker_main(argv)


def run_smoke_test(argv: list[str] | None = None) -> int:
    """frozen 빌드 셀프 스모크(P12.2-21, P12.3-17, GPU/빌드 역할 분리).

    기본 --smoke-test는 패키징 무결성 검사이므로 GPU/CUDA는 optional(SKIP)이다.
    --require-gpu를 함께 주면 P13 strict 모드로 CUDA/GPU/dtype 정책 실패가 FAIL이다.
    모델 weight 로드나 GPU 생성은 수행하지 않는다.
    """
    checks: list[tuple[str, bool, str]] = []
    require_gpu = "--require-gpu" in (argv if argv is not None else sys.argv)

    def _check(name: str):
        def deco(fn):
            try:
                fn()
                checks.append((name, True, ""))
            except Exception as e:
                checks.append((name, False, f"{type(e).__name__}: {e}"))
        return deco

    @_check("PySide6")
    def _pyside():
        import PySide6  # noqa: F401

    @_check("worker dispatch")
    def _worker_dispatch():
        from voice_studio.workers.launcher import worker_command  # noqa: F401

    @_check("diagnostics dispatch")
    def _diag_dispatch():
        from voice_studio.workers.launcher import diagnostics_command  # noqa: F401

    # P12.3 Final Hotfix: _resolve_binary는 모듈 함수이지 instance method가 아니므로
    # 어댑터 인스턴스에 그 이름을 호출하면 frozen에서 AttributeError로 실패했다.
    # 어댑터 인스턴스의 ffmpeg/ffprobe 속성으로 검사하고, 실제 binary 실행 가능 여부까지 확인한다.
    @_check("FFmpeg")
    def _ffmpeg():
        from voice_studio.infra.ffmpeg_adapter import RealFfmpegAdapter
        adapter = RealFfmpegAdapter()
        assert adapter.ffmpeg
        adapter.probe_version()

    @_check("FFprobe")
    def _ffprobe():
        from voice_studio.infra.ffmpeg_adapter import RealFfmpegAdapter
        adapter = RealFfmpegAdapter()
        assert adapter.ffprobe

    @_check("heavy runtime imports")
    def _heavy_check():
        from voice_studio.heavy_smoke import heavy_checks
        heavy_checks(checks, require_gpu=require_gpu)

    @_check("AppContext")
    def _ctx():
        from voice_studio.core.paths import ensure_app_dirs  # noqa: F401
        ensure_app_dirs()
        from voice_studio.core import config  # noqa: F401

    from voice_studio.core.paths import logs_dir
    import sys as _sys
    frozen = bool(getattr(_sys, "frozen", False))
    skipped: list[str] = []
    failed: list[str] = []
    for name, okflag, detail in checks:
        if okflag:
            print(f"{name}: OK")
        elif detail.startswith("SKIP") and not require_gpu:
            # P13 waveform hotfix 후속: heavy_smoke는 GPU/CUDA를 optional SKIP으로 표시한다.
            # frozen 빌드 PC에 GPU가 없어도 기본 smoke는 packaging integrity 검사이므로
            # SKIP이며 실패가 아니다(--require-gpu일 때만 fatal).
            print(f"{name}: SKIP (frozen) {detail}")
            skipped.append(name)
        elif frozen or require_gpu:
            print(f"{name}: FAIL {detail}")
            failed.append(f"{name}: {detail}")
        else:
            # 개발 환경(Repository run)에서는 heavy 의존성(GPU 런타임/ffmpeg.exe)
            # 미설치가 정상이므로 SKIP으로 기록한다. frozen 빌드와 --require-gpu에서는 FAIL이다.
            print(f"{name}: SKIP (dev) {detail}")
            skipped.append(name)
    if failed:
        _finish_smoke(logs_dir(), "SMOKE_FAILED", "; ".join(failed))
        return 1
    _finish_smoke(logs_dir(), "SMOKE_OK", f"skipped={len(skipped)}" if skipped else "")
    return 0


def _finish_smoke(logs_path, status: str, detail: str) -> None:
    """GUI exe는 stdout이 보이지 않으므로 결과를 logs 파일에도 남긴다(P12.2-21)."""
    try:
        logs_path.mkdir(parents=True, exist_ok=True)
        with open(logs_path / "smoke-test.log", "a", encoding="utf-8") as fh:
            fh.write(f"{status} {detail}\n")
    except Exception:
        pass
    print(status, detail)


def run_stt_smoke(argv: list[str]) -> int:
    """frozen/source 공용 STT internal validation CLI(P13 §9-10).

    VoiceStudio.exe --stt-smoke --audio "<path>" --start 0 --end 6
    bundled FFmpeg로 구간을 decode하고 frozen 환경 안의 bundled
    faster-whisper/ctranslate2로 실제 transcribe한다. 성공 exit 0,
    실패 non-zero. 기술 상세는 log에, 성공 마커 STT_SMOKE_OK는 log+stdout에
    남긴다. profile 생성/사용자 audio 복사/Qwen GPU 로드는 하지 않는다.
    """
    def _parse_value(name: str, default: str) -> str:
        if name in argv:
            i = argv.index(name)
            if i + 1 < len(argv):
                return argv[i + 1]
        return default

    audio_path = _parse_value("--audio", "")
    start_s = float(_parse_value("--start", "0"))
    end_s = float(_parse_value("--end", "6"))

    try:
        from voice_studio.core.paths import logs_dir
        (logs_dir() / "stt-smoke.log").unlink(missing_ok=True)  # run별 판정 분리
    except Exception:
        pass

    def _log(line: str) -> None:
        try:
            from voice_studio.core.paths import logs_dir
            d = logs_dir()
            d.mkdir(parents=True, exist_ok=True)
            with open(d / "stt-smoke.log", "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            pass
        try:
            print(line)
        except Exception:
            pass  # console=False frozen에서 stdout이 None일 수 있다

    _log(f"STT_SMOKE_START audio={audio_path!r} start={start_s} end={end_s}")
    try:
        if not audio_path:
            raise ValueError("--audio path is required")
        if not Path(audio_path).is_file():
            raise FileNotFoundError(f"audio file not found: {audio_path}")
        from voice_studio.infra.ffmpeg_adapter import RealFfmpegAdapter
        adapter = RealFfmpegAdapter()  # frozen: bundled _internal/bin 우선
        pcm = adapter.decode_segment(audio_path, start_s, end_s, 24000)
        if pcm.dtype.name != "float32" or pcm.ndim != 1 or pcm.size == 0:
            raise ValueError(f"decoded PCM sanity failed: {pcm.dtype} {pcm.shape}")
        from voice_studio.services.transcription_service import FasterWhisperTranscriber
        text = FasterWhisperTranscriber().transcribe(pcm, 24000)
        _log(f"STT_SMOKE_OK text={text!r}")
        return 0
    except Exception as exc:
        import traceback
        _log("STT_SMOKE_FAILED")
        try:
            from voice_studio.core.paths import logs_dir
            with open(logs_dir() / "stt-smoke.log", "a", encoding="utf-8") as fh:
                fh.write(traceback.format_exc() + "\n")
        except Exception:
            pass
        return 1


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv if argv is None else argv)
    if "--smoke-test" in args:
        return run_smoke_test(args)
    if "--diagnostics" in args:  # P12.3-06: frozen에서는 exe 진입점으로 진단 실행
        from voice_studio.diagnostics import main as diagnostics_main
        return diagnostics_main(args)
    if "--stt-smoke" in args:  # P13: frozen/source 공용 내부 검증 CLI
        return run_stt_smoke(args)
    if "--worker" in args:
        i = args.index("--worker")
        if i + 1 >= len(args):
            from voice_studio.workers.worker_main import main as worker_main
            return worker_main(["worker", ""])
        return _run_worker(["worker", args[i + 1]])
    # 메인 프로세스는 Qwen 모델을 절대 로드하지 않는다.
    from PySide6.QtWidgets import QApplication
    from voice_studio.app_context import create_context
    from voice_studio.ui.main_window import MainWindow
    from voice_studio.core.paths import cleanup_stale_jobs, ensure_app_dirs, cleanup_preview_cache
    ensure_app_dirs()
    cleanup_preview_cache()  # 시작 시 남은 미리듣기 임시 WAV 정리(P12.1-14)
    cleanup_stale_jobs()  # 비정상 종료로 남은 작업 캐시 정리(P12.3-19)
    app = QApplication(args)
    from voice_studio.ui.theme import apply_app_style
    apply_app_style(app)  # P17-B: 공통 밝은 테마
    window = MainWindow(create_context())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
