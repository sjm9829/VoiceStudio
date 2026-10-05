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


def run_smoke_test() -> int:
    """frozen 빌드 셀프 스모크(P12.2-21).

    Qt/worker 인자 파싱/핵심 모듈 import만 확인하고 종료한다. 모델 다운로드나 GPU
    생성은 수행하지 않는다(설치 직후 실패 원인을 1차로 좁히는 용도).
    """
    try:
        import PySide6  # noqa: F401
        from voice_studio.core.paths import ensure_app_dirs, logs_dir
        ensure_app_dirs()
        from voice_studio.workers.job_schema import parse_job  # noqa: F401
        from voice_studio.ui.main_window import MainWindow  # noqa: F401
        from voice_studio.workers.launcher import worker_command  # noqa: F401
        from voice_studio.core import config  # noqa: F401
        _finish_smoke(logs_dir(), "SMOKE_OK", "")
        return 0
    except Exception as e:
        try:
            from voice_studio.core.paths import logs_dir
            _finish_smoke(logs_dir(), "SMOKE_FAILED", f"{type(e).__name__}: {e}")
        except Exception:
            pass
        return 1


def _finish_smoke(logs_path, status: str, detail: str) -> None:
    """GUI exe는 stdout이 보이지 않으므로 결과를 logs 파일에도 남긴다(P12.2-21)."""
    try:
        logs_path.mkdir(parents=True, exist_ok=True)
        with open(logs_path / "smoke-test.log", "a", encoding="utf-8") as fh:
            fh.write(f"{status} {detail}\n")
    except Exception:
        pass
    print(status, detail)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv if argv is None else argv)
    if "--smoke-test" in args:
        return run_smoke_test()
    if "--diagnostics" in args:  # P12.3-06: frozen에서는 exe 진입점으로 진단 실행
        from voice_studio.diagnostics import main as diagnostics_main
        return diagnostics_main(args)
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
    from voice_studio.core.paths import ensure_app_dirs, cleanup_preview_cache
    ensure_app_dirs()
    cleanup_preview_cache()  # 시작 시 남은 미리듣기 임시 WAV 정리(P12.1-14)
    app = QApplication(args)
    window = MainWindow(create_context())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
