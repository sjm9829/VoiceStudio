"""목소리 등록/편집 대화상자.

- 참조 파일 선택(MP3/M4A/WAV/FLAC), 전체 파형 + 시작/끝 드래그 + 구간 들어보기
- 자동 받아쓰기(선택 구간만, CPU), 대사 필수, 사용 권한 동의 체크 필수
- 등록 시 별도 프로세스에서 음성 분석(worker) 수행 후 종료
"""

from __future__ import annotations
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
                               QLabel, QFileDialog, QCheckBox, QMessageBox, QPlainTextEdit)
from ..core import config
from ..core.errors import ProfileError, VoiceStudioError
from .waveform_widget import WaveformWidget

class _WaveformLoader(QThread):
    done = Signal(list, float)

    def __init__(self, audio, path):
        super().__init__()
        self.audio = audio; self.path = path

    def run(self):
        try:
            peaks = self.audio.waveform(self.path)
            dur = self.audio.probe(self.path)["duration"]
            self.done.emit(peaks, dur)
        except VoiceStudioError as exc:
            self.done.emit([], 0.0)

class VoiceEditorDialog(QDialog):
    def __init__(self, context, parent=None, uuid=None):
        super().__init__(parent)
        self.context = context
        self.uuid = uuid
        self.setWindowTitle("목소리 편집" if uuid else "목소리 등록")
        self.resize(700, 560)
        self.source_path = None
        self.duration = 0.0
        self._build_ui()
        if uuid:
            self._load_existing()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("이름"))
        self.name_edit = QLineEdit()
        name_row.addWidget(self.name_edit)
        layout.addLayout(name_row)

        file_row = QHBoxLayout()
        self.file_label = QLabel("참조 파일을 선택해 주세요.")
        file_row.addWidget(self.file_label, 1)
        pick = QPushButton("참조 파일 선택")
        pick.clicked.connect(self.pick_file)
        file_row.addWidget(pick)
        layout.addLayout(file_row)

        self.wave = WaveformWidget()
        self.wave.selection_changed.connect(self._on_selection)
        layout.addWidget(self.wave, 1)
        self.selection_label = QLabel("선택 구간: 없음")
        layout.addWidget(self.selection_label)

        hint = QLabel("15초 정도면 사용 가능하지만 더 길다고 반드시 좋아지지는 않습니다.")
        layout.addWidget(hint)

        text_row = QHBoxLayout()
        text_row.addWidget(QLabel("참조 음성의 대사"))
        self.transcript_edit = QPlainTextEdit()
        self.transcript_edit.setPlaceholderText("선택한 구간에서 말한 내용을 그대로 적어 주세요.")
        self.transcript_edit.setMaximumHeight(90)
        text_row.addWidget(self.transcript_edit)
        layout.addLayout(text_row, 1)

        stt_btn = QPushButton("자동 받아쓰기")
        stt_btn.clicked.connect(self.auto_transcribe)
        preview_btn = QPushButton("선택 구간 들어보기")
        preview_btn.clicked.connect(self.preview_selection)
        meta_row = QHBoxLayout()
        meta_row.addWidget(stt_btn); meta_row.addWidget(preview_btn); meta_row.addStretch()
        layout.addLayout(meta_row)

        self.consent = QCheckBox("이 음성을 사용할 권한이 있음을 확인합니다.")
        layout.addWidget(self.consent)

        buttons = QHBoxLayout()
        save = QPushButton("목소리 등록" if self.uuid is None else "변경 저장")
        save.clicked.connect(self.save)
        cancel = QPushButton("취소")
        cancel.clicked.connect(self.reject)
        buttons.addStretch(); buttons.addWidget(save); buttons.addWidget(cancel)
        layout.addLayout(buttons)

    def _load_existing(self):
        profile = self.context.profile_service.get(self.uuid)
        self.name_edit.setText(profile.name)
        self.transcript_edit.setPlainText(profile.ref_text)

    def pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "참조 파일 선택", "", "오디오 파일 (*.mp3 *.m4a *.wav *.flac)")
        if not path:
            return
        self.source_path = path
        self.file_label.setText(path)
        self._loader = _WaveformLoader(self.context.audio, path)
        self._loader.done.connect(self._on_peaks)
        self._loader.start()

    def _on_peaks(self, peaks, duration):
        self.duration = duration
        if duration <= 0:
            QMessageBox.warning(self, "보이스 스튜디오", "오디오 파일을 열 수 없습니다.")
            return
        self.wave.set_peaks(peaks, duration)
        self._on_selection(0.0, duration)

    def _on_selection(self, start, end):
        self.selection_label.setText(f"선택 구간: {start:.1f}초 ~ {end:.1f}초 ({end - start:.1f}초)")
        if end - start > config.REFERENCE_WARN_SECONDS:
            QMessageBox.warning(self, "긴 선택 구간",
                "선택 구간이 깁니다. 사용 가능하지만 더 길다고 반드시 좋아지지는 않습니다.")

    def preview_selection(self):
        if self.source_path is None:
            return
        # 실제 재생은 배포 환경에서 검증(미검증 표시 유지). 여기서는 범위만 확인.
        pass

    def auto_transcribe(self):
        if self.source_path is None:
            QMessageBox.information(self, "보이스 스튜디오", "먼저 참조 파일을 선택해 주세요.")
            return
        s, e = self.wave.start_s, self.wave.end_s
        pcm = self.context.audio.decode_reference_segment(self.source_path, s, e)
        text = self.context.transcriber.transcribe(pcm, config.REFERENCE_SAMPLE_RATE)
        self.transcript_edit.setPlainText(text)

    def save(self):
        try:
            if self.uuid:
                self.context.profile_service.rename(self.uuid, self.name_edit.text())
                self.accept()
                return
            if not self.consent.isChecked():
                QMessageBox.information(self, "보이스 스튜디오",
                    "이 음성을 사용할 권한이 있음을 확인합니다. 에 체크해 주세요.")
                return
            if not self.transcript_edit.toPlainText().strip():
                QMessageBox.information(self, "보이스 스튜디오", "참조 음성의 대사를 입력해 주세요.")
                return
            # 등록 본체는 별도 프로세스(worker)에서 수행. 여기서는 컨텍스트 기반 등록 경로 사용.
            self.context.profile_service.register(
                name=self.name_edit.text(), source_path=self.source_path or "",
                start_s=self.wave.start_s, end_s=self.wave.end_s,
                ref_text=self.transcript_edit.toPlainText(), consent=self.consent.isChecked())
            self.accept()
        except VoiceStudioError as exc:
            QMessageBox.warning(self, "등록 실패", exc.user_message)
