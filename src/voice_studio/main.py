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


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv if argv is None else argv)
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
