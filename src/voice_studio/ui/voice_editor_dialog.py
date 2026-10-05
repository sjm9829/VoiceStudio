"""목소리 등록/편집 대화상자.

- 참조 파일 선택(MP3/M4A/WAV/FLAC), 전체 파형 + 시작/끝 드래그 + 선택 구간 들어보기(실제 재생)
- 자동 받아쓰기: 실제 FasterWhisperTranscriber(cpu/int8)를 별도 스레드에서 실행(UI 블록 없음)
- 등록은 UI 프로세스에서 직접 하지 않고 register worker(QProcess)로 위임한다.
  Qwen 모델 로드/ICL 프롬프트 생성/프로필 저장은 전부 worker 프로세스에서 수행한다.
"""

from __future__ import annotations
import json
import os
import shutil
import uuid as uuidlib
from pathlib import Path

from PySide6.QtCore import Qt, QProcess, QTimer, QThread, Signal
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
                               QLabel, QFileDialog, QCheckBox, QMessageBox, QPlainTextEdit,
                               QProgressBar)
from ..core import config
from ..core.errors import (VoiceStudioError, ProfileError, DuplicateNameError,
                           ConsentRequiredError, TranscriptRequiredError, FfmpegNotFoundError)
from ..core.paths import preview_cache_dir
from ..workers.job_schema import build_register_payload
from ..workers.launcher import worker_command
from ..workers.linebuffer import JsonlBuffer
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
        except VoiceStudioError:
            self.done.emit([], 0.0)

class _TranscribeThread(QThread):
    """선택 구간 디코딩 + 받아쓰기를 UI 스레드 밖에서 실행한다."""
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, audio, transcriber, path, start_s, end_s):
        super().__init__()
        self.audio = audio
        self.transcriber = transcriber
        self.path = path
        self.start_s = start_s
        self.end_s = end_s

    def run(self):
        try:
            pcm = self.audio.decode_preview_segment(self.path, self.start_s, self.end_s)
            text = self.transcriber.transcribe(pcm, config.REFERENCE_SAMPLE_RATE)
            self.done.emit(text)
        except Exception as exc:
            self.failed.emit(str(exc))

class _PreviewThread(QThread):
    """선택 구간만 임시 WAV로 만들어 실제로 들어볼 수 있게 준비한다."""
    ready = Signal(str)
    failed = Signal(str)

    def __init__(self, audio, path, start_s, end_s):
        super().__init__()
        self.audio = audio
        self.path = path
        self.start_s = start_s
        self.end_s = end_s
        self.wav_path: str | None = None

    def run(self):
        try:
            pcm = self.audio.decode_preview_segment(self.path, self.start_s, self.end_s)
            # 관리 경로(%LOCALAPPDATA%\VoiceStudio\cache\preview) 사용(P12.1-14).
            d = preview_cache_dir()
            d.mkdir(parents=True, exist_ok=True)
            wav_path = str(d / f"vs_preview_{uuidlib.uuid4().hex}.wav")
            self.audio.encode_wav(pcm, wav_path)
            self.wav_path = wav_path
            self.ready.emit(wav_path)
        except Exception as exc:
            self.failed.emit(str(exc))

    def cleanup(self):
        if self.wav_path:
            try:
                os.unlink(self.wav_path)
            except OSError:
                pass
            self.wav_path = None

class VoiceEditorDialog(QDialog):
    def __init__(self, context, parent=None, uuid=None):
        super().__init__(parent)
        self.context = context
        self.uuid = uuid
        self.setWindowTitle("목소리 편집" if uuid else "목소리 등록")
        self.resize(700, 560)
        self.source_path = None
        self.duration = 0.0
        self._transcribe_thread = None
        self._preview_thread = None
        self._worker: QProcess | None = None
        self._buffer = JsonlBuffer()
        self._job_dir: Path | None = None
        self._result_event: dict | None = None
        self._error_message: str | None = None
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

        hint = QLabel("3초 이상 선택할 수 있습니다. 5~15초를 권장하며, 더 길다고 반드시 좋아지지는 않습니다. 30초 이상은 경고만 표시합니다.")
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

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        self.consent = QCheckBox("이 음성을 사용할 권한이 있음을 확인합니다.")
        layout.addWidget(self.consent)

        buttons = QHBoxLayout()
        self.save_btn = QPushButton("목소리 등록" if self.uuid is None else "변경 저장")
        self.save_btn.clicked.connect(self.save)
        cancel = QPushButton("취소")
        cancel.clicked.connect(self.reject)
        buttons.addStretch(); buttons.addWidget(self.save_btn); buttons.addWidget(cancel)
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
        # 기본 선택을 라벨이 아니라 실제 selection state로 반영(P12.1-04).
        # 긴 파일을 열어도 worker에는 권장 15초 구간만 전달된다.
        self.wave.set_selection(0.0, min(duration, config.REFERENCE_TARGET_SECONDS))

    def _on_selection(self, start, end):
        self.selection_label.setText(f"선택 구간: {start:.1f}초 ~ {end:.1f}초 ({end - start:.1f}초)")
        if end - start >= config.REFERENCE_WARN_SECONDS and not getattr(self, "_warned_long", False):
            self._warned_long = True
            QMessageBox.information(self, "보이스 스튜디오",
                                    "30초 이상의 참조 음성도 사용할 수 있지만 더 길다고 반드시 좋아지지는 않습니다.")

    def preview_selection(self):
        """선택 구간만 임시 WAV로 만들어 실제로 재생하고, 재생 후 임시 파일을 정리한다."""
        if self.source_path is None:
            QMessageBox.information(self, "보이스 스튜디오", "먼저 참조 파일을 선택해 주세요.")
            return
        if self._preview_thread is not None and self._preview_thread.isRunning():
            return
        self._preview_thread = _PreviewThread(self.context.audio, self.source_path,
                                              self.wave.start_s, self.wave.end_s)
        self._preview_thread.ready.connect(self._on_preview_ready)
        self._preview_thread.failed.connect(
            lambda msg: QMessageBox.warning(self, "보이스 스튜디오", f"들어보기를 실행할 수 없습니다.\n{msg}"))
        self._preview_thread.start()

    def _on_preview_ready(self, wav_path: str):
        if hasattr(os, "startfile"):
            os.startfile(wav_path)  # noqa: S606  (Windows 전용: 기본 연결 프로그램으로 WAV 재생)
            QTimer.singleShot(15000, self._preview_thread.cleanup)
            return
        try:
            from PySide6.QtMultimedia import QSoundEffect
            from PySide6.QtCore import QUrl
            self._preview_effect = QSoundEffect(self)
            self._preview_effect.setSource(QUrl.fromLocalFile(wav_path))
            self._preview_effect.play()
            QTimer.singleShot(15000, self._preview_thread.cleanup)
        except Exception:
            self._preview_thread.cleanup()
            QMessageBox.information(self, "보이스 스튜디오", "이 환경에서는 미리 듣기를 지원하지 않습니다.")

    def auto_transcribe(self):
        if self.source_path is None:
            QMessageBox.information(self, "보이스 스튜디오", "먼저 참조 파일을 선택해 주세요.")
            return
        if self._transcribe_thread is not None and self._transcribe_thread.isRunning():
            return
        self.status_label.setText("받아쓰기 중… (처음이라면 모델을 내려받는 중일 수 있습니다)")
        self._transcribe_thread = _TranscribeThread(
            self.context.audio, self.context.transcriber,
            self.source_path, self.wave.start_s, self.wave.end_s)
        self._transcribe_thread.done.connect(self._on_transcribe_done)
        self._transcribe_thread.failed.connect(self._on_transcribe_failed)
        self._transcribe_thread.start()

    def _on_transcribe_done(self, text: str):
        self.status_label.setText("")
        if not text:
            QMessageBox.information(self, "보이스 스튜디오", "말이 인식되지 않았습니다. 대사를 직접 입력해 주세요.")
            return
        self.transcript_edit.setPlainText(text)

    def _on_transcribe_failed(self, message: str):
        self.status_label.setText("")
        QMessageBox.warning(self, "받아쓰기 실패",
                            "받아쓰기를 실행할 수 없습니다. 인터넷 연결과 설정을 확인해 주세요.\n"
                            + message[:200])

    # ---- 등록: worker 위임 ----
    def save(self):
        try:
            if self.uuid:
                self.context.profile_service.rename(self.uuid, self.name_edit.text())
                self.accept()
                return
            self._validate_register_inputs()
            self._start_register_worker()
        except VoiceStudioError as exc:
            # 모델 미다운로드 등 사용자 안내가 필요한 오류(P12.1-06/07)
            QMessageBox.warning(self, "보이스 스튜디오", exc.user_message)
        except OSError as exc:
            QMessageBox.warning(self, "보이스 스튜디오", f"작업 파일을 준비할 수 없습니다: {exc}")


    def _validate_register_inputs(self):
        """register worker 시작 전 UI 입력 검증(P12.1-06)."""
        name = self.name_edit.text().strip()
        if not name:
            raise ProfileError("목소리 이름을 입력해 주세요.", user_message="목소리 이름을 입력해 주세요.")
        existing = [p.name for p in self.context.profile_service.list_profiles() if p.uuid != self.uuid]
        if name in existing:
            raise DuplicateNameError()
        if not self.source_path or not Path(self.source_path).is_file():
            raise ProfileError("참조 파일 없음", user_message="참조 파일을 선택해 주세요.")
        if self.wave.end_s <= self.wave.start_s:
            raise ProfileError("선택 구간 없음", user_message="선택 구간이 없습니다. 파형에서 구간을 선택해 주세요.")
        if self.wave.end_s - self.wave.start_s < config.REFERENCE_APP_MIN_SECONDS:
            raise ProfileError("선택 구간 짧음", user_message=f"선택 구간이 너무 짧습니다. {config.REFERENCE_APP_MIN_SECONDS:.0f}초 이상 선택해 주세요.")
        if not self.transcript_edit.toPlainText().strip():
            raise TranscriptRequiredError()
        if not self.consent.isChecked():
            raise ConsentRequiredError()
        try:
            from ..infra.ffmpeg_adapter import RealFfmpegAdapter
            RealFfmpegAdapter()
        except FfmpegNotFoundError:
            raise FfmpegNotFoundError()

    def _start_register_worker(self):
        from ..core.paths import safe_job_cache_dir
        job = build_register_payload(
            job_id=str(uuidlib.uuid4()), name=self.name_edit.text().strip(),
            source_path=self.source_path or "",
            start_s=self.wave.start_s, end_s=self.wave.end_s,
            ref_text=self.transcript_edit.toPlainText().strip(),
            profile_dir=str(self.context.profile_repository.root),
            model_path=self.context.model_manager.model_path())
        job_dir = safe_job_cache_dir(job["job_id"])
        job_file = job_dir / "job.json"
        job_file.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        self._job_dir = job_dir
        self._buffer = JsonlBuffer()
        self._result_event = None
        self._error_message = None
        program, args = worker_command(str(job_file))
        self._worker = QProcess(self)
        self._worker.readyReadStandardOutput.connect(self._on_worker_output)
        self._worker.readyReadStandardError.connect(self._on_worker_stderr)
        self._worker.finished.connect(self._on_register_finished)
        self._worker.start(program, args)
        self.save_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # 불확정 진행
        self.status_label.setText("목소리를 등록하는 중… (처음이라면 모델을 준비하는 중일 수 있습니다)")

    def _on_worker_output(self):
        if self._worker is None:
            return
        for ev in self._buffer.feed(bytes(self._worker.readAllStandardOutput())):
            kind = ev.get("kind")
            if kind == "error":
                self._error_message = ev.get("message", "오류가 발생했습니다.")
                detail = ev.get("detail") or ""
                if detail:
                    self._error_message = f"{self._error_message}\n{detail}"
            elif kind == "result":
                self._result_event = ev

    def _on_worker_stderr(self):
        # stderr는 진단용. UI에는 보이지 않게 유지한다.
        pass

        def closeEvent(self, event):
        # P12.3-25: 대화상자가 닫히면 worker 슬롯을 반납한다(등록 중 닫기 포함).
        self.context.jobs.release()
        super().closeEvent(event)

    def _on_register_finished(self, code, status):
        if self._worker is not None:
            try:
                for ev in self._buffer.feed(bytes(self._worker.readAllStandardOutput())):
                    if ev.get("kind") == "result":
                        self._result_event = ev
                    elif ev.get("kind") == "error" and self._error_message is None:
                        self._error_message = ev.get("message", "")
            except Exception:
                pass
        self.progress.setVisible(False)
        self.save_btn.setEnabled(True)
        self._cleanup_job()
        self._worker = None
        if code == 0 and self._result_event is not None:
            self.status_label.setText("")
            self.accept()
        else:
            QMessageBox.warning(self, "등록 실패", self._error_message or "등록에 실패했습니다.")

    def _cleanup_job(self):
        try:
            if self._job_dir is not None and self._job_dir.exists():
                shutil.rmtree(self._job_dir, ignore_errors=True)
        except OSError:
            pass
        self._job_dir = None
