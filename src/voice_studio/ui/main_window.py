"""메인 나레이션 화면."""

from __future__ import annotations
from PySide6.QtCore import Qt, QProcess, QTimer
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
                               QPushButton, QPlainTextEdit, QLabel, QFileDialog, QMessageBox,
                               QProgressBar)
from ..core.errors import ProfileError
from ..workers.protocol import status_event, progress_event, error_event, result_event
from .voice_manager_dialog import VoiceManagerDialog
from .settings_dialog import SettingsDialog

PHASE_KO = {
    "model_loading": "모델을 준비하는 중…",
    "encoding": "MP3를 만드는 중…",
    "done": "완료",
}

class MainWindow(QMainWindow):
    def __init__(self, context):
        super().__init__()
        self.context = context
        self.setWindowTitle("보이스 스튜디오")
        self.resize(880, 620)
        self._worker: QProcess | None = None
        self._cancel_requested = False
        self._last_output: str | None = None
        self._build_ui()
        self._refresh_voices()

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        top = QHBoxLayout()
        top.addWidget(QLabel("목소리"))
        self.voice_combo = QComboBox()
        self.voice_combo.setMinimumWidth(220)
        top.addWidget(self.voice_combo)
        manage_btn = QPushButton("목소리 관리")
        manage_btn.clicked.connect(self.open_manager)
        top.addWidget(manage_btn)
        top.addStretch()
        settings_btn = QPushButton("설정")
        settings_btn.clicked.connect(self.open_settings)
        top.addWidget(settings_btn)
        layout.addLayout(top)

        self.script_edit = QPlainTextEdit()
        self.script_edit.setPlaceholderText("음성으로 만들 대본을 입력하세요.")
        layout.addWidget(self.script_edit, 1)

        self.status_label = QLabel("")
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress)

        buttons = QHBoxLayout()
        self.generate_btn = QPushButton("음성 생성")
        self.generate_btn.clicked.connect(self.start_generation)
        self.cancel_btn = QPushButton("취소")
        self.cancel_btn.clicked.connect(self.request_cancel)
        self.cancel_btn.setVisible(False)
        self.save_btn = QPushButton("MP3 저장")
        self.save_btn.clicked.connect(self.save_mp3)
        self.save_btn.setEnabled(False)
        self.play_btn = QPushButton("들어보기")
        self.play_btn.clicked.connect(self.play_preview)
        self.play_btn.setEnabled(False)
        buttons.addWidget(self.generate_btn)
        buttons.addWidget(self.cancel_btn)
        buttons.addWidget(self.play_btn)
        buttons.addWidget(self.save_btn)
        layout.addLayout(buttons)

    # ---- 목소리 목록 ----
    def _refresh_voices(self):
        self.voice_combo.clear()
        profiles = self.context.profile_service.list_profiles()
        for p in profiles:
            self.voice_combo.addItem(p.name, p.uuid)
        if not profiles:
            self.status_label.setText("목소리를 먼저 등록해 주세요.")

    def open_manager(self):
        dlg = VoiceManagerDialog(self.context, self)
        dlg.exec()
        self._refresh_voices()

    def open_settings(self):
        SettingsDialog(self.context, self).exec()

    # ---- 생성 ----
    def start_generation(self):
        uuid = self.voice_combo.currentData()
        if uuid is None:
            QMessageBox.information(self, "보이스 스튜디오", "목소리를 먼저 등록해 주세요.")
            return
        script = self.script_edit.toPlainText().strip()
        if not script:
            QMessageBox.information(self, "보이스 스튜디오", "대본을 입력해 주세요.")
            return
        self._start_worker({"mode": "narrate", "profile_uuid": uuid, "script": script})

    def _start_worker(self, payload: dict):
        import json, uuid as uuidlib
        from ..core.paths import safe_job_cache_dir
        job_id = str(uuidlib.uuid4())
        payload = dict(payload, job_id=job_id)
        job_file = safe_job_cache_dir(job_id) / "job.json"
        job_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        self._worker = QProcess(self)
        self._worker.readyReadStandardOutput.connect(self._on_worker_output)
        self._worker.finished.connect(self._on_worker_finished)
        import sys as _sys
        from pathlib import Path as _P
        worker_py = _P(__file__).resolve().parents[1] / "workers" / "worker_main.py"
        self._worker.start(_sys.executable, [str(worker_py), str(job_file)])
        self.generate_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self._cancel_requested = False
        self._set_status("model_loading", 0, 0)

    def _on_worker_output(self):
        if self._worker is None:
            return
        data = bytes(self._worker.readAllStandardOutput()).decode("utf-8", "replace")
        for line in data.splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = ev.get("kind")
            if kind == "status":
                self._set_status(ev.get("phase"), ev.get("index", 0), ev.get("total", 0))
            elif kind == "progress":
                total = ev.get("total", 1)
                self.progress.setMaximum(total)
                self.progress.setValue(ev.get("index", 0))
                i, n = ev.get("index", 0), ev.get("total", 0)
                self.status_label.setText(f"구간 {i}/{n} 만드는 중…")
            elif kind == "error":
                self.status_label.setText(ev.get("message", "오류가 발생했습니다."))
            elif kind == "result":
                self._last_output = ev.get("output_path")

    def _on_worker_finished(self, code, status):
        self.cancel_btn.setVisible(False)
        self.generate_btn.setEnabled(True)
        if self._cancel_requested:
            self.status_label.setText("작업을 취소했습니다.")
        elif code == 0 and self._last_output:
            self.status_label.setText("완료")
            self.save_btn.setEnabled(True)
            self.play_btn.setEnabled(True)
        elif code != 0:
            self.status_label.setText(self.status_label.text() or "작업이 실패했습니다.")
        self._worker = None

    def _set_status(self, phase: str, index: int, total: int):
        text = PHASE_KO.get(phase, "")
        if phase == "generating" and total:
            text = f"구간 {index}/{total} 만드는 중…"
        self.status_label.setText(text)
        if phase in ("generating", "encoding"):
            self.progress.setVisible(True)
            if total:
                self.progress.setMaximum(total); self.progress.setValue(index)
        else:
            self.progress.setVisible(False)

    def request_cancel(self):
        self._cancel_requested = True
        if self._worker is not None:
            self._worker.terminate()   # worker 종료로 CUDA context 소멸
            QTimer.singleShot(3000, self._worker.kill)

    def play_preview(self):
        if self._last_output:
            from PySide6.QtMultimedia import QSoundEffect  # 실제 재생은 배포 환경에서 검증
            import os as _os
            _os.startfile(self._last_output) if hasattr(_os, "startfile") else None

    def save_mp3(self):
        if not self._last_output:
            return
        out_dir = self.context.settings.get("mp3_output_dir", "")
        suggested = out_dir + "/나레이션.mp3" if out_dir else "나레이션.mp3"
        path, _ = QFileDialog.getSaveFileName(self, "MP3 저장", suggested, "MP3 (*.mp3)")
        if not path:
            return
        import shutil
        shutil.copyfile(self._last_output, path)
        QMessageBox.information(self, "보이스 스튜디오", f"MP3를 저장했습니다:\n{path}")
