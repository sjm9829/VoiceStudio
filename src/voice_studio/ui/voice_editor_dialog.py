"""목소리 등록/편집 대화상자.

- 참조 파일 선택(MP3/M4A/WAV/FLAC), 전체 파형 + 시작/끝 드래그 + 선택 구간 들어보기(실제 재생)
- 자동 받아쓰기: 실제 FasterWhisperTranscriber(cpu/int8)를 별도 스레드에서 실행(UI 블록 없음)
- 등록은 UI 프로세스에서 직접 하지 않고 register worker(QProcess)로 위임한다.
  Qwen 모델 로드/ICL 프롬프트 생성/프로필 저장은 전부 worker 프로세스에서 수행한다.
"""

from __future__ import annotations
import datetime
import json
import logging
import math
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
                           ConsentRequiredError, TranscriptRequiredError, FfmpegNotFoundError,
                           UnsupportedAudioError)
from ..core.paths import logs_dir, preview_cache_dir
from ..workers.job_schema import build_register_payload
from ..workers.launcher import worker_command
from ..workers.linebuffer import JsonlBuffer
from .waveform_widget import WaveformWidget

class _WaveformLoader(QThread):
    """probe → duration 검증 → waveform을 순서대로 실행하는 로더(P13 waveform hotfix).

    계약: 성공 시 done(peaks, duration) 1회, 실패 시 failed(예외) 1회.
    실패를 done([], 0.0)으로 삼키지 않고, VoiceStudioError와 unexpected 예외 모두
    failed로 전달해 UI가 실패/성공을 구분할 수 있게 한다. UI 스레드에서 절대 emit 이외
    작업을 하지 않는다.
    """
    done = Signal(list, float)
    failed = Signal(object)

    def __init__(self, audio, path):
        super().__init__()
        self.audio = audio; self.path = path

    def run(self):
        log = logging.getLogger(__name__)
        try:
            log.info("파형 로딩: probe 시작 path=%r", self.path)
            info = self.audio.probe(self.path)
            duration = info["duration"]
            if not (isinstance(duration, (int, float)) and math.isfinite(float(duration))
                    and float(duration) > 0):
                raise UnsupportedAudioError(
                    f"재생 시간을 확인할 수 없습니다: {duration!r}")
            log.info("파형 로딩: probe 완료 duration=%.3fs, waveform 시작", float(duration))
            peaks = self.audio.waveform(self.path)
            log.info("파형 로딩: waveform 완료 buckets=%d", len(peaks))
            self.done.emit(peaks, float(duration))
        except Exception as exc:
            log.warning("파형 로딩 실패: path=%r %s: %s", self.path,
                        type(exc).__name__, exc, exc_info=True)
            self.failed.emit(exc)

class _TranscribeThread(QThread):
    """선택 구간 디코딩 + 받아쓰기를 UI 스레드 밖에서 실행한다."""
    done = Signal(str)
    failed = Signal(object)  # 실패 원인 예외 객체(카테고리별 안내 분리, P13 hotfix)

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
            self.failed.emit(exc)

def transcribe_failure_message(exc: Exception) -> str:
    """받아쓰기 실패 원인에 따라 사용자 안내를 분리한다(P13 hotfix).

    오디오 decode 실패와 모델 준비(다운로드/네트워크) 실패를 다른 안내로 구분하고
    traceback이나 기술 상세는 UI에 노출하지 않는다.
    """
    from ..core.errors import UnsupportedAudioError, OfflineError
    if isinstance(exc, UnsupportedAudioError):
        return ("선택한 음성 구간을 읽을 수 없습니다.\n"
                "다른 구간을 선택하거나 오디오 파일을 확인해 주세요.")
    if isinstance(exc, OfflineError):
        return ("자동 받아쓰기 모델을 준비할 수 없습니다.\n"
                "인터넷 연결을 확인한 뒤 다시 시도해 주세요.")
    text = str(exc)
    lowered = text.lower()
    if any(k in lowered for k in ("network", "connection", "timeout", "download", "offline")):
        return ("자동 받아쓰기 모델을 준비할 수 없습니다.\n"
                "인터넷 연결을 확인한 뒤 다시 시도해 주세요.")
    return "자동 받아쓰기를 실행할 수 없습니다."


class _PreviewThread(QThread):
    """선택 구간만 임시 WAV로 만들어 실제로 들어볼 수 있게 준비한다."""
    ready = Signal(str)
    failed = Signal(object)  # 실패 원인 예외 객체(_TranscribeThread와 동일 계약)

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
            self.failed.emit(exc)

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
        self._analyzing = False
        self._transcribe_thread = None
        self._preview_thread = None
        self._worker: QProcess | None = None
        self._buffer = JsonlBuffer()
        self._job_dir: Path | None = None
        # P13 §7: worker stderr 로그 식별용 메타데이터.
        self._worker_job_id = ""
        self._worker_mode = "register"
        # P12.3 Final Hotfix: 자신이 실제로 acquire한 slot만 release하기 위한 ownership 상태.
        self._job_slot_acquired = False
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

    def _reset_waveform_state(self):
        """파형/선택/기간 state를 완전히 초기화한다(P13 waveform hotfix).

        파일 로드 실패 시 이전 파일의 파형·선택·기간을 재사용하지 않는다.
        """
        self.duration = 0.0
        self.wave.clear()
        self.selection_label.setText("선택 구간: 없음")
        self._warned_long = False

    def _audio_ready(self) -> bool:
        """분석 중/분석 실패 상태에서 preview/받아쓰기/등록을 차단한다(P13 waveform hotfix)."""
        if self._analyzing:
            QMessageBox.information(self, "보이스 스튜디오",
                                    "오디오 파일을 분석하는 중입니다. 완료 후 다시 시도해 주세요.")
            return False
        if self.duration <= 0:
            QMessageBox.information(self, "보이스 스튜디오", "먼저 참조 파일을 선택해 주세요.")
            return False
        return True

    def pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "참조 파일 선택", "", "오디오 파일 (*.mp3 *.m4a *.wav *.flac)")
        if not path:
            return
        # 새 파일 분석 전 이전 파일 state를 먼저 완전히 초기화한다.
        self._reset_waveform_state()
        self.source_path = path
        self.file_label.setText(path)
        self._analyzing = True
        self.status_label.setText("오디오 파일을 분석하는 중…")
        self._loader = _WaveformLoader(self.context.audio, path)
        self._loader.done.connect(self._on_peaks)
        self._loader.failed.connect(self._on_load_failed)
        self._loader.start()

    def _on_peaks(self, peaks, duration):
        self._analyzing = False
        self.status_label.setText("")
        self.duration = duration
        if not (isinstance(duration, (int, float)) and math.isfinite(float(duration))
                and float(duration) > 0):
            # 로더 계약상 발생하지 않지만, 방어적으로 실패 흐름과 동일하게 처리한다.
            self._reset_waveform_state()
            QMessageBox.warning(self, "보이스 스튜디오",
                                "오디오 파일의 재생 시간을 확인할 수 없습니다.\n"
                                "다른 파일을 선택하거나 오디오 파일을 확인해 주세요.")
            return
        self.wave.set_peaks(peaks, duration)
        # 기본 선택을 라벨이 아니라 실제 selection state로 반영(P12.1-04).
        # 긴 파일을 열어도 worker에는 권장 15초 구간만 전달된다.
        self.wave.set_selection(0.0, min(duration, config.REFERENCE_TARGET_SECONDS))

    def _on_load_failed(self, exc: object):
        """파형 로딩 실패: 이전 state 재사용 금지 + 사용자 안내(P13 waveform hotfix).

        기술 상세(stderr/traceback)는 UI에 노출하지 않고 로그로만 남긴다.
        """
        self._analyzing = False
        self.status_label.setText("")
        self._reset_waveform_state()
        logging.getLogger(__name__).warning(
            "참조 파일 분석 실패: path=%r exc=%r", self.source_path, exc)
        QMessageBox.warning(self, "보이스 스튜디오",
                            "오디오 파일의 재생 시간을 확인할 수 없습니다.\n"
                            "다른 파일을 선택하거나 오디오 파일을 확인해 주세요.")

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
        if not self._audio_ready():
            return
        if self._preview_thread is not None and self._preview_thread.isRunning():
            return
        self._preview_thread = _PreviewThread(self.context.audio, self.source_path,
                                              self.wave.start_s, self.wave.end_s)
        self._preview_thread.ready.connect(self._on_preview_ready)
        self._preview_thread.failed.connect(self._on_preview_failed)
        self._preview_thread.start()

    def _on_preview_failed(self, exc_or_msg):
        """들어보기 실패: 짧은 사용자 안내만 노출하고 FFmpeg 상세는 로그로 남긴다(P13 hotfix)."""
        import logging
        detail = str(exc_or_msg)
        logging.getLogger(__name__).warning("미리 듣기 실패: %s", detail)
        QMessageBox.warning(
            self, "보이스 스튜디오",
            "선택한 음성 구간을 재생할 수 없습니다.\n오디오 파일과 선택 구간을 확인해 주세요.")

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
        if not self._audio_ready():
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

    def _on_transcribe_failed(self, exc: object):
        """실패 원인별 안내 분리(P13 hotfix). 기술 상세는 로그로만 남긴다."""
        import logging
        self.status_label.setText("")
        logging.getLogger(__name__).warning("받아쓰기 실패: %s", exc, exc_info=isinstance(exc, BaseException))
        QMessageBox.warning(self, "받아쓰기 실패", transcribe_failure_message(exc))

    # ---- 등록: worker 위임 ----
    def save(self):
        try:
            if self.uuid:
                self.context.profile_service.rename(self.uuid, self.name_edit.text())
                self.accept()
                return
            if self._analyzing:
                QMessageBox.information(self, "보이스 스튜디오",
                                        "오디오 파일을 분석하는 중입니다. 완료 후 다시 시도해 주세요.")
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

    def _release_job_slot(self):
        """자신이 acquire한 GPU worker slot만 반납한다(P12.3 Final Hotfix)."""
        if self._job_slot_acquired:
            self._job_slot_acquired = False
            jobs = getattr(self.context, "jobs", None)
            if jobs is not None:
                jobs.release()

    def _start_register_worker(self):
        from ..core.paths import safe_job_cache_dir
        # 동시 worker 1개 제한(P12.3-25): narrate 등 다른 worker 실행 중이면 시작하지 않는다.
        if not self.context.jobs.try_acquire():
            QMessageBox.warning(
                self,
                "보이스 스튜디오",
                "다른 작업이 실행 중입니다. 완료 후 다시 시도해 주세요.",
            )
            return
        # 여기서부터는 어떤 예외가 나도 slot을 반납해야 한다(P12.3 Final Hotfix 3-2).
        self._job_slot_acquired = True
        try:
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
            self._worker_job_id = job["job_id"]
            self._buffer = JsonlBuffer()
            self._result_event = None
            self._error_message = None
            program, args = worker_command(str(job_file))
            self._worker = QProcess(self)
            self._worker.readyReadStandardOutput.connect(self._on_worker_output)
            self._worker.readyReadStandardError.connect(self._on_worker_stderr)
            self._worker.finished.connect(self._on_register_finished)
            self._worker.start(program, args)
        except Exception:
            self._release_job_slot()
            self._worker = None
            raise
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
                # P13 §8: UI에는 message/user_message만 노출한다. 기술 detail
                # (예: TypeError traceback)는 로그에만 남긴다.
                self._error_message = ev.get("message", "오류가 발생했습니다.")
                detail = ev.get("detail") or ""
                if detail:
                    logging.getLogger(__name__).warning(
                        "worker error detail: job=%s mode=%s detail=%s",
                        ev.get("job_id", self._worker_job_id), ev.get("mode", "?"), detail)
            elif kind == "result":
                self._result_event = ev

    def _on_worker_stderr(self):
        r"""worker stderr를 진단 로그 파일로 보존한다(P13 §7).

        UI에는 표시하지 않고, %LOCALAPPDATA%\VoiceStudio\logs\worker-stderr.log에
        job_id/mode/timestamp와 함께 append한다. stdout JSONL protocol과 섞지 않는다.
        """
        if self._worker is None:
            return
        text = bytes(self._worker.readAllStandardError()).decode("utf-8", "replace")
        if not text:
            return
        self._append_worker_stderr_log(text)

    def _append_worker_stderr_log(self, text: str) -> None:
        logging.getLogger(__name__).warning("worker stderr: %s", text)
        try:
            logs_dir().mkdir(parents=True, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with (logs_dir() / "worker-stderr.log").open("a", encoding="utf-8") as fp:
                fp.write(f"\n---- {stamp} job={self._worker_job_id} mode={self._worker_mode} ----\n")
                fp.write(text)
                if not text.endswith("\n"):
                    fp.write("\n")
        except OSError:
            logging.getLogger(__name__).warning("worker-stderr.log 기록 실패", exc_info=True)

    def closeEvent(self, event):
        """QThread lifecycle(P13 §15) + P12.3 Final Hotfix slot 계약.

        파형 분석/받아쓰기/미리듣기 스레드가 실행 중이면 close를 차단한다.
        running 중인 QThread가 dialog보다 먼저 파괴되면
        "QThread: Destroyed while thread is still running" crash가 나므로,
        가장 단순하고 안전한 정책인 close 차단을 사용한다.
        등록 worker(QProcess) 실행 중에는 기존대로 terminate 요청 후 진행한다.
        """
        for name in ("_loader", "_transcribe_thread", "_preview_thread"):
            thread = getattr(self, name, None)
            if thread is not None and thread.isRunning():
                QMessageBox.information(
                    self, "보이스 스튜디오",
                    "오디오 작업이 진행 중입니다. 완료 후 창을 닫아 주세요.")
                event.ignore()
                return
        # P12.3 Final Hotfix: 무조건 release하지 않는다.
        # 등록 worker 실행 중이 아니면 coordinator를 건드리지 않고, 실행 중이면
        # worker 종료를 요청한다. slot 반납은 _on_register_finished에서 수행하므로
        # 다른 화면이 점유한 narrate slot을 여기서 풀 수 없다.
        if self._worker is not None:
            terminate = getattr(self._worker, "terminate", None)
            if callable(terminate):
                terminate()
                QTimer.singleShot(3000, self._worker.kill)  # terminate 실패 시 강제 종료
        super().closeEvent(event)

    def _on_register_finished(self, code, status):
        try:
            if self._worker is not None:
                try:
                    for ev in self._buffer.feed(bytes(self._worker.readAllStandardOutput())):
                        if ev.get("kind") == "result":
                            self._result_event = ev
                        elif ev.get("kind") == "error" and self._error_message is None:
                            self._error_message = ev.get("message", "")
                except Exception:
                    pass
            # P12.3 Final Hotfix: UI 후처리 중 예외가 나도 slot 반납은 보장한다(성공/실패 공통).
            self._release_job_slot()
            self.progress.setVisible(False)
            self.save_btn.setEnabled(True)
            self._cleanup_job()
            if hasattr(self._worker, "deleteLater"):
                self._worker.deleteLater()  # P13 §22: QProcess 객체 정리
            self._worker = None
            if code == 0 and self._result_event is not None:
                self.status_label.setText("")
                self.accept()
            else:
                QMessageBox.warning(self, "등록 실패", self._error_message or "등록에 실패했습니다.")
        finally:
            self._release_job_slot()

    def _cleanup_job(self):
        try:
            if self._job_dir is not None and self._job_dir.exists():
                shutil.rmtree(self._job_dir, ignore_errors=True)
        except OSError:
            pass
        self._job_dir = None
