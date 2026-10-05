"""파형 표시 + 시작/끝 드래그 핸들 위젯."""

from __future__ import annotations
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtWidgets import QWidget

class WaveformWidget(QWidget):
    """peak envelope 파형에 시작/끝 선택 핸들을 겹쳐 그린다."""
    selection_changed = Signal(float, float)  # 시작/끝 초

    def __init__(self, parent=None):
        super().__init__(parent)
        self.peaks: list[float] = []
        self.duration = 0.0
        self.start_s = 0.0
        self.end_s = 0.0
        self._drag = None  # "start" | "end" | "move"
        self.setMinimumHeight(120)

    def set_peaks(self, peaks: list[float], duration: float):
        self.peaks = peaks
        self.duration = duration
        self.start_s = 0.0
        self.end_s = duration
        self.update()

    def set_selection(self, start_s: float, end_s: float):
        self.start_s = max(0.0, min(start_s, self.duration))
        self.end_s = max(self.start_s, min(end_s, self.duration))
        self.selection_changed.emit(self.start_s, self.end_s)
        self.update()

    def _x_for(self, t: float) -> float:
        if self.duration <= 0:
            return 0.0
        return (t / self.duration) * self.width()

    def _t_for(self, x: int) -> float:
        if self.width() <= 0:
            return 0.0
        return max(0.0, min(1.0, x / self.width())) * self.duration

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = QRectF(self.rect())
        painter.fillRect(rect, QColor("#111318"))
        if self.peaks and self.duration > 0:
            mid = self.height() / 2
            step = self.width() / max(1, len(self.peaks))
            pen = QPen(QColor("#7fd1c8"), 1)
            painter.setPen(pen)
            for i, p in enumerate(self.peaks):
                h = max(1.0, p * (self.height() - 8) / 2)
                x = i * step
                t = (i / max(1, len(self.peaks) - 1)) * self.duration if len(self.peaks) > 1 else 0
                painter.setPen(QPen(QColor("#4c576a"), 1))
                if self.start_s <= t <= self.end_s:
                    painter.setPen(QPen(QColor("#7fd1c8"), 1))
                painter.drawLine(int(x), int(mid - h), int(x), int(mid + h))
        # 선택 영역 오버레이
        if self.duration > 0:
            x0, x1 = self._x_for(self.start_s), self._x_for(self.end_s)
            painter.setPen(QPen(QColor("#7fd1c8"), 2))
            painter.drawLine(int(x0), 0, int(x0), self.height())
            painter.drawLine(int(x1), 0, int(x1), self.height())

    def mousePressEvent(self, e):
        t = self._t_for(e.position().toPoint().x())
        self._drag = "start" if abs(self._x_for(self.start_s) - e.position().x()) < 8 else (
            "end" if abs(self._x_for(self.end_s) - e.position().x()) < 8 else "move")
        self._press_t = t

    def mouseMoveEvent(self, e):
        if self._drag is None:
            return
        t = self._t_for(e.position().toPoint().x())
        if self._drag == "start":
            self.set_selection(min(t, self.end_s - 0.1), self.end_s)
        elif self._drag == "end":
            self.set_selection(self.start_s, max(t, self.start_s + 0.1))
        else:
            delta = t - self._press_t
            self._press_t = t
            self.set_selection(self.start_s + delta, self.end_s + delta)

    def mouseReleaseEvent(self, e):
        self._drag = None
