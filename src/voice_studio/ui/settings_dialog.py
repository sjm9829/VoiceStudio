"""설정 대화상자."""

from __future__ import annotations
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
                               QLineEdit, QFileDialog, QGroupBox, QMessageBox, QProgressBar)
import json
from pathlib import Path

from ..core import config
from ..services.model_manager import ModelManager

BITRATE_LABELS = {"표준 128": 128, "고음질 192": 192, "최고 256": 256}

class _ModelDownloadThread(QThread):
    """모델 다운로드를 UI 스레드 밖에서 실행한다(P12.2-14). 수 GB 다운로드 중 창이 멈추지 않게 한다."""
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, model_manager, force=False):
        super().__init__()
        self.model_manager = model_manager
        self.force = force

    def run(self):
        try:
            path = self.model_manager.download(force=self.force)
            self.done.emit(str(path))
        except Exception as exc:
            self.failed.emit(str(exc))


class SettingsDialog(QDialog):
    def __init__(self, context, parent=None):
        super().__init__(parent)
        self.context = context
        self.setWindowTitle("설정")
        self.resize(560, 480)
        layout = QVBoxLayout(self)

        save_group = QGroupBox("MP3 저장")
        save_layout = QVBoxLayout(save_group)
        dir_row = QHBoxLayout()
        self.dir_edit = QLineEdit(context.settings.get("mp3_output_dir", ""))
        browse = QPushButton("폴더 선택")
        browse.clicked.connect(self.pick_dir)
        dir_row.addWidget(self.dir_edit, 1); dir_row.addWidget(browse)
        save_layout.addLayout(dir_row)
        quality_row = QHBoxLayout()
        quality_row.addWidget(QLabel("MP3 음질 (Mono)"))
        self.quality = QComboBox()
        for label, kbps in BITRATE_LABELS.items():
            self.quality.addItem(label, kbps)
        self._select_saved_quality()  # 저장된 음질을 그대로 복원(P12.2-16)
        quality_row.addWidget(self.quality)
        save_layout.addLayout(quality_row)
        layout.addWidget(save_group)

        gpu_group = QGroupBox("NVIDIA 그래픽 카드")
        gpu_layout = QVBoxLayout(gpu_group)
        self.gpu_label = QLabel("상태를 확인하는 중…")
        gpu_layout.addWidget(self.gpu_label)
        gpu_layout.addWidget(QLabel("사용 후 그래픽 메모리 해제: 사용(기본). 작업 프로세스를 종료해 메모리를 확실히 해제합니다."))
        check_gpu = QPushButton("그래픽 카드 상태 확인")
        check_gpu.clicked.connect(self.check_gpu)
        gpu_layout.addWidget(check_gpu)
        layout.addWidget(gpu_group)

        model_group = QGroupBox("음성 모델")
        model_layout = QVBoxLayout(model_group)
        self.model_label = QLabel(context.model_manager.status_text())
        model_layout.addWidget(self.model_label)
        dl = QPushButton("모델 받기")
        dl.clicked.connect(self.download_model)
        redl = QPushButton("모델 다시 받기")
        redl.clicked.connect(lambda: self.download_model(force=True))
        self.download_buttons = (dl, redl)
        model_layout.addWidget(dl); model_layout.addWidget(redl)
        self.dl_progress = QProgressBar()
        self.dl_progress.setVisible(False)
        self.dl_progress.setRange(0, 0)  # 불확정 진행(정확한 byte progress 미구현)
        model_layout.addWidget(self.dl_progress)
        layout.addWidget(model_group)

        stt_group = QGroupBox("자동 받아쓰기 모델")
        stt_layout = QVBoxLayout(stt_group)
        stt_layout.addWidget(QLabel("CPU 기반 받아쓰기. 첫 사용 시 모델을 내려받습니다."))
        layout.addWidget(stt_group)

        close = QPushButton("닫기")
        close.clicked.connect(self._save_and_close)
        layout.addWidget(close)

    def pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "저장 폴더 선택", self.dir_edit.text())
        if d:
            self.dir_edit.setText(d)

    def check_gpu(self):
        """설치 후 Self-Diagnosis 요약(P12.2-22).

        무거운 모델 런타임 의존성을 메인 프로세스에서 import하지
        않도록(구조 계약) 자식 인터프리터로 진단을 실행하고 요약만 받아 표시한다.
        세부 기술 정보는 logs/diagnosis.log에만 남긴다.
        """
        import subprocess
        from ..workers.launcher import diagnostics_command
        try:
            program, dargs = diagnostics_command()
            r = subprocess.run([program, *dargs],
                               capture_output=True, text=True, timeout=90,
                               cwd=str(Path(__file__).resolve().parents[2]))
            out = (r.stdout or "").strip().splitlines()
            data = json.loads(out[-1]) if out else {}
            summary = data.get("summary") or {}
        except Exception:
            summary = {"그래픽 카드": "진단을 실행할 수 없습니다.", "음성 모델": "확인 필요",
                       "오디오 구성 요소": "확인 필요", "자동 받아쓰기": "직접 대사 입력으로 사용 가능"}
        order = ("그래픽 카드", "CUDA", "음성 모델", "오디오 구성 요소", "자동 받아쓰기")
        lines = [f"{k}: {summary[k]}" for k in order if k in summary]
        lines += [f"{k}: {v}" for k, v in summary.items() if k not in order]
        self.gpu_label.setText("\n".join(lines))


    def _select_saved_quality(self):
        saved = self.context.settings.get("mp3_bitrate_kbps", 192)
        index = self.quality.findData(int(saved))
        self.quality.setCurrentIndex(index if index >= 0 else 1)

    def download_model(self, force=False):
        """모델 다운로드를 별도 스레드로 실행해 UI freeze를 막는다(P12.2-14)."""
        if getattr(self, "_dl_thread", None) is not None and self._dl_thread.isRunning():
            return  # 중복 클릭 무시
        self.dl_progress.setVisible(True)
        for b in self.download_buttons:
            b.setEnabled(False)
        self._dl_thread = _ModelDownloadThread(self.context.model_manager, force=force)
        self._dl_thread.done.connect(self._on_download_done)
        self._dl_thread.failed.connect(self._on_download_failed)
        self._dl_thread.start()

    def _on_download_done(self, path: str):
        self.dl_progress.setVisible(False)
        for b in self.download_buttons:
            b.setEnabled(True)
        self.model_label.setText(self.context.model_manager.status_text())

    def _on_download_failed(self, message: str):
        self.dl_progress.setVisible(False)
        for b in self.download_buttons:
            b.setEnabled(True)  # 실패 후 UI 복구
        QMessageBox.warning(self, "모델 받기", message[:300])

    def _save_and_close(self):
        out_dir = self.dir_edit.text().strip()
        if out_dir:
            try:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
            except OSError:
                QMessageBox.warning(self, "설정", "저장 폴더를 만들 수 없습니다. 경로를 확인해 주세요.")
                return
            self.dir_edit.setText(out_dir)
        self.context.save_settings({
            **self.context.settings,
            "mp3_output_dir": self.dir_edit.text(),
            "mp3_bitrate_kbps": self.quality.currentData(),
        })
        self.accept()
