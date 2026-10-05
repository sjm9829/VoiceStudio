"""보이스 스튜디오 진입점."""

from __future__ import annotations
import sys

def main() -> int:
    from PySide6.QtWidgets import QApplication
    from .ui.main_window import MainWindow
    from .app_context import AppContext
    app = QApplication(sys.argv)
    app.setApplicationName("보이스 스튜디오")
    context = AppContext()
    window = MainWindow(context)
    window.show()
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
