"""설정 대화상자."""

from __future__ import annotations
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
                               QLineEdit, QFileDialog, QGroupBox, QMessageBox)
from ..core import config
from ..services.model_manager import ModelManager

BITRATE_LABELS = {"표준 128": 128, "고음질 192": 192, "최고 256": 256}

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
        self.quality.setCurrentIndex(1)
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
        model_layout.addWidget(dl); model_layout.addWidget(redl)
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
        try:
            import subprocess
            r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
            self.gpu_label.setText(r.stdout.strip() or "NVIDIA 그래픽 카드를 찾을 수 없습니다.")
        except FileNotFoundError:
            self.gpu_label.setText("NVIDIA 그래픽 카드를 사용할 수 없습니다. (nvidia-smi 없음)")

    def download_model(self, force=False):
        try:
            path = self.context.model_manager.download(force=force)
            self.model_label.setText(self.context.model_manager.status_text())
        except Exception as exc:
            QMessageBox.warning(self, "모델 받기", str(exc))

    def _save_and_close(self):
        self.context.save_settings({
            **self.context.settings,
            "mp3_output_dir": self.dir_edit.text(),
            "mp3_bitrate_kbps": self.quality.currentData(),
        })
        self.accept()
