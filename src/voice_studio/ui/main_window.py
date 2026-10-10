"""메인 나레이션 화면.

worker 실행 계약:
- worker 프로세스는 개발 환경에서 `python -m voice_studio.main --worker job.json`,
  frozen 환경에서 `VoiceStudio.exe --worker job.json`으로 실행된다(workers/launcher).
- stdout JSONL은 청크 단위로 잘려 올 수 있으므로 per-process receive buffer(JsonlBuffer)로 조립한다.
- stderr는 별도로 수집해 진단 로그로 남긴다.
"""

from __future__ import annotations
import datetime
import logging
import os
import shutil
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QProcess, QTimer, QUrl, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
                               QPushButton, QPlainTextEdit, QLabel, QFileDialog, QMessageBox,
                               QProgressBar, QFrame, QSlider)
from .theme import (BG, CARD, BORDER, TEXT, MUTED, PRIMARY, PRIMARY_HOVER, DANGER,
                    SUCCESS, SELECT_BG, make_card, step_label, hint_label, primary_button)
from ..core.errors import VoiceStudioError, ProfileError, ModelNotDownloadedError
from ..core.paths import logs_dir
from ..workers.job_schema import build_narrate_payload
from ..workers.launcher import apply_no_window, worker_command
from ..core.paths import safe_job_cache_dir
from ..workers.linebuffer import JsonlBuffer
from .voice_manager_dialog import VoiceManagerDialog
from .settings_dialog import SettingsDialog

PHASE_KO = {
    "model_loading": "모델을 준비하는 중…",
    "analyzing_reference": "목소리 특징을 분석하는 중…",
    "saving_profile": "목소리를 저장하는 중…",
    "generating": "구간 {i}/{n} 만드는 중…",
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
        self._buffer = JsonlBuffer()
        self._stderr_chunks: list[str] = []
        self._job_dir: Path | None = None
        # P12.3 Final Hotfix: 자신이 실제로 acquire한 slot만 release하기 위한 ownership 상태.
        self._job_slot_acquired = False
        self._cancel_requested = False
        self._last_output: str | None = None
        self._result_received = False
        self._build_ui()
        self._refresh_voices()

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(10)

        # ---- 헤더 ----
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("보이스 스튜디오")
        title.setObjectName("headerTitle")
        subtitle = QLabel("내 목소리로 만드는 나레이션")
        subtitle.setObjectName("headerSubtitle")
        title_box.addWidget(title); title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()
        settings_btn = QPushButton("설정")
        settings_btn.clicked.connect(self.open_settings)
        header.addWidget(settings_btn)
        outer.addLayout(header)

        # ---- 1. 목소리 선택 ----
        voice_card = make_card()
        voice_card_layout = voice_card.layout()
        voice_card_layout.addWidget(step_label("1. 목소리 선택"))
        voice_row = QHBoxLayout()
        self.voice_combo = QComboBox()
        self.voice_combo.setMinimumWidth(260)
        self.voice_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        voice_row.addWidget(self.voice_combo, 1)
        manage_btn = QPushButton("목소리 관리")
        manage_btn.clicked.connect(self.open_manager)
        voice_row.addWidget(manage_btn)
        voice_card_layout.addLayout(voice_row)
        outer.addWidget(voice_card)

        # ---- 2. 대본 편집 ----
        script_card = make_card()
        script_layout = script_card.layout()
        script_header = QHBoxLayout()
        script_header.addWidget(step_label("2. 대본"))
        script_header.addStretch()
        self.char_count_label = QLabel("0자")
        self.char_count_label.setObjectName("hintLabel")
        script_header.addWidget(self.char_count_label)
        script_layout.addLayout(script_header)
        self.script_edit = QPlainTextEdit()
        self.script_edit.setPlaceholderText(
            "음성으로 만들 대본을 입력하세요.\n\n"
            "문단을 나누면 자연스럽게 쉬어 가며 읽습니다. Ctrl+Enter로 바로 만들 수 있습니다.")
        self.script_edit.textChanged.connect(self._update_char_count)
        script_layout.addWidget(self.script_edit, 1)
        outer.addWidget(script_card, 3)

        # 생성 동작 버튼 행
        action_row = QHBoxLayout()
        self.generate_btn = primary_button("음성 생성")
        self.generate_btn.clicked.connect(self.start_generation)
        self.cancel_btn = QPushButton("취소")
        self.cancel_btn.setObjectName("danger")
        self.cancel_btn.clicked.connect(self.request_cancel)
        self.cancel_btn.setVisible(False)
        action_row.addStretch()
        action_row.addWidget(self.cancel_btn)
        action_row.addWidget(self.generate_btn)
        outer.addLayout(action_row)

        # ---- 3. 생성 결과 ----
        result_card = make_card()
        result_layout = result_card.layout()
        result_header = QHBoxLayout()
        result_header.addWidget(step_label("3. 생성 결과"))
        result_header.addStretch()
        self.status_label = QLabel("")
        self.status_label.setObjectName("statusLabel")
        result_header.addWidget(self.status_label)
        result_layout.addLayout(result_header)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        result_layout.addWidget(self.progress)
        self.empty_state_label = hint_label(
            "아직 만든 나레이션이 없습니다. 대본을 입력하고 음성 생성을 눌러 주세요.")
        result_layout.addWidget(self.empty_state_label)

        # 내장 오디오 플레이어(P17-B): 재생/일시정지, 탐색, 시각, 볼륨
        player_row = QHBoxLayout()
        self.play_btn = QPushButton("재생")
        self.play_btn.clicked.connect(self._toggle_playback)
        self.play_btn.setEnabled(False)
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setEnabled(False)
        self.seek_slider.sliderMoved.connect(self._on_seek_slider_moved)
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setObjectName("mutedLabel")
        volume_label = QLabel("소리 크기")
        volume_label.setObjectName("hintLabel")
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.setFixedWidth(110)
        self.volume_slider.valueChanged.connect(self._on_volume_changed)
        player_row.addWidget(self.play_btn)
        player_row.addWidget(self.seek_slider, 1)
        player_row.addWidget(self.time_label)
        player_row.addWidget(volume_label)
        player_row.addWidget(self.volume_slider)
        result_layout.addLayout(player_row)

        result_buttons = QHBoxLayout()
        self.save_btn = QPushButton("MP3 저장")
        self.save_btn.clicked.connect(self.save_mp3)
        self.save_btn.setEnabled(False)
        self.open_folder_btn = QPushButton("저장 폴더 열기")
        self.open_folder_btn.clicked.connect(self.open_output_location)
        self.open_folder_btn.setEnabled(False)
        result_buttons.addWidget(self.save_btn)
        result_buttons.addWidget(self.open_folder_btn)
        result_buttons.addStretch()
        result_layout.addLayout(result_buttons)
        outer.addWidget(result_card, 2)

        self._setup_player()
        # P14 UX: 대본 입력 중 Enter로 생성 시작(탭 이동 없이 바로 실행).
        shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)
        shortcut.setObjectName("scriptGenerateShortcut")
        shortcut.activated.connect(self.start_generation)

    # ---- 문자 수 / 빈 상태 ----
    def _update_char_count(self):
        self.char_count_label.setText(f"{len(self.script_edit.toPlainText())}자")

    def _set_empty_state_visible(self, visible: bool):
        self.empty_state_label.setVisible(visible)

    # ---- 내장 오디오 플레이어 ----
    def _setup_player(self):
        """QMediaPlayer 기반 앱 내부 재생(P17-B).

        QtMultimedia를 사용할 수 없는 환경에서는 play_preview가 기존
        기본 연결 프로그램 경로(os.startfile)로 자연히 떨어진다.
        """
        self._player = None
        self._audio_output = None
        try:
            from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
        except Exception:
            return
        self._player = QMediaPlayer(self)
        self._audio_output = QAudioOutput(self)
        self._audio_output.setVolume(0.8)
        self._player.setAudioOutput(self._audio_output)
        self._player.positionChanged.connect(self._on_player_position)
        self._player.durationChanged.connect(self._on_player_duration)
        self._player.playbackStateChanged.connect(self._on_playback_state)
        self._player.mediaStatusChanged.connect(self._on_media_status)
        self._player.sourceChanged.connect(lambda: self._on_player_source_changed())

    def _player_ready(self) -> bool:
        return self._player is not None

    def _load_result_for_playback(self, path: str):
        """완성된 결과를 플레이어에 적재하고 UI 상태를 완료로 맞춘다."""
        self._set_empty_state_visible(False)
        if self._player_ready():
            from PySide6.QtCore import QUrl
            self._player.setSource(QUrl.fromLocalFile(str(path)))
            self.seek_slider.setEnabled(True)
            self.play_btn.setEnabled(True)
        else:
            self.play_btn.setEnabled(True)

    def _toggle_playback(self):
        if not self._player_ready():
            self.play_preview()
            return
        from PySide6.QtMultimedia import QMediaPlayer
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        elif self._player.source().isValid():
            self._player.play()

    def _on_player_position(self, position: int):
        if not self.seek_slider.isSliderDown():
            self.seek_slider.setValue(position)
        self._update_time_label(position)

    def _on_player_duration(self, duration: int):
        self.seek_slider.setRange(0, max(0, duration))
        self._update_time_label(self._player.position() if self._player_ready() else 0)

    @staticmethod
    def _format_ms(ms: int) -> str:
        seconds = int(ms // 1000)
        return f"{seconds // 60}:{seconds % 60:02d}"

    def _update_time_label(self, position: int):
        duration = self._player.duration() if self._player_ready() else 0
        self.time_label.setText(f"{self._format_ms(position)} / {self._format_ms(duration)}")

    def _on_playback_state(self, state):
        from PySide6.QtMultimedia import QMediaPlayer
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.play_btn.setText("일시정지" if playing else "재생")

    def _on_media_status(self, status):
        from PySide6.QtMultimedia import QMediaPlayer
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.seek_slider.setValue(0)
            self.play_btn.setText("재생")

    def _on_player_source_changed(self):
        self._on_player_duration(self._player.duration() if self._player_ready() else 0)

    def _on_seek_slider_moved(self, value: int):
        if self._player_ready():
            self._player.setPosition(value)

    def _on_volume_changed(self, value: int):
        if self._audio_output is not None:
            self._audio_output.setVolume(value / 100.0)
    # ---- 목소리 목록 ----
    def _refresh_voices(self):
        # P14 UX: 관리 대화상자 닫기 후에도 선택한 목소리를 유지한다.
        previous_uuid = self.voice_combo.currentData()
        self.voice_combo.clear()
        profiles = self.context.profile_service.list_profiles()
        for p in profiles:
            self.voice_combo.addItem(p.name, p.uuid)
        if previous_uuid is not None:
            index = self.voice_combo.findData(previous_uuid)
            if index >= 0:
                self.voice_combo.setCurrentIndex(index)
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
        profile_uuid = self.voice_combo.currentData()
        if profile_uuid is None:
            QMessageBox.information(self, "보이스 스튜디오", "목소리를 먼저 등록해 주세요.")
            return
        script = self.script_edit.toPlainText().strip()
        if not script:
            QMessageBox.information(self, "보이스 스튜디오", "대본을 입력해 주세요.")
            return
        try:
            if not self._start_worker(self._build_job(profile_uuid, script)):
                return  # 다른 worker 실행 중(P12.3-25)
        except VoiceStudioError as e:
            # 모델 미다운로드/프로필 오류 등 생성 시작 실패를 UI에서 안내(P12.1-07).
            QMessageBox.warning(self, "보이스 스튜디오", str(e))
        except OSError as e:
            QMessageBox.warning(self, "보이스 스튜디오", f"작업 파일을 준비할 수 없습니다: {e}")

    def _build_job(self, profile_uuid: str, script: str) -> dict:
        """worker가 요구하는 나레이션 job payload를 완성한다.

        대본 분할과 문단 경계 플래그는 이 프로세스에서 계산해 worker에 그대로 전달하고,
        모델 경로는 이미 받아진 로컬 스냅샷 경로로 고정한다(오프라인 동작 보장).
        """
        from ..services.text_segmenter import segment_with_flags
        segments, gap_flags = segment_with_flags(script)
        if not segments:
            raise ProfileError("대본을 입력해 주세요.")
        from ..core.paths import default_mp3_dir
        import uuid as uuidlib
        output_dir = Path(self.context.settings.get("mp3_output_dir", "") or default_mp3_dir())
        # 기본/사용자 지정 폴더 모두 생성 시작 전에 확보한다(P12.2-03). 한글 경로 포함.
        output_dir.mkdir(parents=True, exist_ok=True)
        # worker 결과는 작업 캐시(jobs/<job_id>/result.mp3)에 두고, MP3 저장 시
        # 사용자 위치로 복사한다(P12.2-04). 실패/취소 파일이 Music 폴더에 남지 않는다.
        job_id = str(uuidlib.uuid4())
        output_path = str(safe_job_cache_dir(job_id) / "result.mp3")
        return build_narrate_payload(
            job_id=job_id, profile_uuid=profile_uuid, segments=segments, gap_flags=gap_flags,
            profile_dir=str(self.context.profile_repository.root), output_path=output_path,
            bitrate_kbps=int(self.context.settings.get("mp3_bitrate_kbps", 192)),
            model_path=self.context.model_manager.model_path())

    def _release_job_slot(self):
        """자신이 acquire한 GPU worker slot만 반납한다(P12.3 Final Hotfix)."""
        if self._job_slot_acquired:
            self._job_slot_acquired = False
            self.context.jobs.release()

    def _start_worker(self, payload: dict) -> bool:
        import json
        # 동시 worker 1개 제한(P12.3-25): register 등 다른 worker 실행 중이면 시작하지 않는다.
        if not self.context.jobs.try_acquire():
            QMessageBox.warning(self, "보이스 스튜디오", "다른 작업이 실행 중입니다. 완료 후 다시 시도해 주세요.")
            return False
        # acquire 이후 예외가 나도 slot을 반납한다(P12.3 Final Hotfix 3-2).
        self._job_slot_acquired = True
        try:
            # 연속 생성 시 이전 성공 결과 캐시를 먼저 정리한다(P12.3-18).
            self._cleanup_job()
            self._last_output = None
            job_id = payload["job_id"]
            job_dir = safe_job_cache_dir(job_id)
            job_file = job_dir / "job.json"
            job_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            self._job_dir = job_dir
            self._buffer = JsonlBuffer()
            self._stderr_chunks = []
            self._current_job_id = payload["job_id"]
            self._result_received = False
            program, args = worker_command(str(job_file))
            self._worker = QProcess(self)
            apply_no_window(self._worker)
            self._worker.readyReadStandardOutput.connect(self._on_worker_output)
            self._worker.readyReadStandardError.connect(self._on_worker_stderr)
            self._worker.finished.connect(self._on_worker_finished)
            self._worker.start(program, args)
        except Exception:
            self._release_job_slot()
            self._worker = None
            raise
        self.generate_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self._cancel_requested = False
        self._set_status("model_loading", 0, 0)
        return True

    def _on_worker_output(self):
        if self._worker is None:
            return
        data = bytes(self._worker.readAllStandardOutput())
        for ev in self._buffer.feed(data):
            self._handle_event(ev)

    def _on_worker_stderr(self):
        if self._worker is None:
            return
        data = bytes(self._worker.readAllStandardError()).decode("utf-8", "replace")
        self._stderr_chunks.append(data)
        # P13 §7: stderr를 메모리에만 쌓아두고 버리지 않고 즉시 진단 로그로 flush.
        self._persist_worker_stderr(data)

    def _persist_worker_stderr(self, text: str) -> None:
        r"""worker stderr를 %LOCALAPPDATA%\VoiceStudio\logs\worker-stderr.log에 append한다."""
        if not text:
            return
        logging.getLogger(__name__).warning("narrate worker stderr: %s", text)
        try:
            logs_dir().mkdir(parents=True, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            job = getattr(self, "_current_job_id", "") or ""
            with (logs_dir() / "worker-stderr.log").open("a", encoding="utf-8") as fp:
                fp.write(f"\n---- {stamp} job={job} mode=narrate ----\n{text}")
                if not text.endswith("\n"):
                    fp.write("\n")
        except OSError:
            logging.getLogger(__name__).warning("worker-stderr.log 기록 실패", exc_info=True)

    def _handle_event(self, ev: dict):
        kind = ev.get("kind")
        if kind == "status":
            self._set_status(ev.get("phase"), ev.get("index", 0), ev.get("total", 0))
        elif kind == "progress":
            i, n = ev.get("index", 0), ev.get("total", 0)
            self._set_status("generating", i, n)
        elif kind == "error":
            detail = ev.get("detail") or ""
            text = ev.get("message", "오류가 발생했습니다.")
            self.status_label.setText(text)
            if detail:
                self._stderr_chunks.append(f"[error:{ev.get('code')}] {detail}\n")
        elif kind == "result":
            self._result_received = True
            self._last_output = ev.get("output_path")

    def _on_worker_finished(self, code, status):
        # QProcess 시그널 이후 남은 출력 flush 보장을 위해 버퍼를 비운다.
        if self._worker is not None:
            try:
                for ev in self._buffer.feed(bytes(self._worker.readAllStandardOutput())):
                    self._handle_event(ev)
                err = bytes(self._worker.readAllStandardError()).decode("utf-8", "replace")
                if err:
                    self._stderr_chunks.append(err)
                    self._persist_worker_stderr(err)
            except Exception:
                pass
        self.cancel_btn.setVisible(False)
        self.generate_btn.setEnabled(True)
        self._release_job_slot()  # P12.3 Final Hotfix: 자신이 acquire한 slot만 반납
        if self._cancel_requested:
            self.status_label.setText("작업을 취소했습니다.")
            self._cleanup_job()
        elif code == 0 and self._result_received and self._last_output:
            self._set_status("done", 0, 0)
            self.save_btn.setEnabled(True)
            self.open_folder_btn.setEnabled(True)
            self._load_result_for_playback(self._last_output)  # P17-B: 내장 플레이어 적재
            # 성공: 캐시 result.mp3를 보존(MP3 저장/들어보기용). 폴더 삭제는
            # MP3 저장 성공 시 또는 새 생성 시작/앱 종료 시 수행한다(P12.3-18).
        else:
            self.status_label.setText(self.status_label.text() or "작업이 실패했습니다.")
            self._cleanup_job()
        if hasattr(self._worker, "deleteLater"):
            self._worker.deleteLater()  # P13 §22: QProcess 객체 정리
        self._worker = None

    def _cleanup_job(self, keep_output: bool = False):
        """작업 종료 후 임시 파일 정리.

        keep_output=True(성공)일 때는 jobs/<job_id>/result.mp3를 보존한다(P12.2-04):
        MP3 저장/들어보기가 이 파일을 사용한다. 실패/취소 시 폴더 전체를 삭제한다.
        """
        if keep_output and self._job_dir is not None:
            self._job_dir = None
            return
        try:
            if self._job_dir is not None and self._job_dir.exists():
                shutil.rmtree(self._job_dir, ignore_errors=True)
        except OSError:
            pass
        self._job_dir = None
        self._last_output = None if not keep_output else self._last_output

    def _set_status(self, phase: str, index: int, total: int):
        text = PHASE_KO.get(phase, "")
        if "{i}" in text:
            text = text.format(i=index, n=total)
        self.status_label.setText(text)
        if phase in ("generating", "encoding") and total:
            self.progress.setVisible(True)
            self.progress.setMaximum(total)
            self.progress.setValue(index)
        elif phase == "encoding":
            self.progress.setVisible(True)
        else:
            self.progress.setVisible(False)

    def request_cancel(self):
        self._cancel_requested = True
        if self._worker is not None:
            terminate = getattr(self._worker, "terminate", None)
            if callable(terminate):
                terminate()
                QTimer.singleShot(3000, self._worker.kill)  # terminate 실패 시 강제 종료

    def play_preview(self):
        """완성된 MP3를 재생한다. Windows: 기본 연결 프로그램, 그 외: QtMultimedia."""
        if not self._last_output or not Path(self._last_output).exists():
            QMessageBox.information(self, "보이스 스튜디오", "먼저 음성을 생성해 주세요.")
            return
        if hasattr(os, "startfile"):
            os.startfile(self._last_output)  # noqa: S606  (Windows 전용 재생 경로)
            return
        try:
            from PySide6.QtMultimedia import QSoundEffect
            from PySide6.QtCore import QUrl
            effect = QSoundEffect(self)
            effect.setSource(QUrl.fromLocalFile(self._last_output))
            effect.play()
        except Exception:
            QMessageBox.information(self, "보이스 스튜디오",
                                    f"이 환경에서는 재생을 지원하지 않습니다.\n{self._last_output}")

    def open_output_location(self):
        """마지막 결과(또는 설정된 출력 폴더)를 파일 관리자로 연다(P17-B)."""
        import subprocess
        targets = []
        if self._last_output and Path(self._last_output).exists():
            targets.append(str(Path(self._last_output).parent))
        out_dir = self.context.settings.get("mp3_output_dir", "")
        if out_dir:
            targets.append(out_dir)
        if not targets:
            QMessageBox.information(self, "보이스 스튜디오", "아직 열 폴더가 없습니다.")
            return
        target = targets[0]
        try:
            if hasattr(os, "startfile"):
                os.startfile(target)  # noqa: S606  (Windows 전용)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", target])
            else:
                subprocess.Popen(["xdg-open", target])
        except OSError as e:
            QMessageBox.warning(self, "보이스 스튜디오", f"폴더를 열지 못했습니다: {e}")

    def save_mp3(self):
        if not self._last_output:
            return
        out_dir = self.context.settings.get("mp3_output_dir", "")
        suggested = out_dir + "/나레이션.mp3" if out_dir else "나레이션.mp3"
        path, _ = QFileDialog.getSaveFileName(self, "MP3 저장", suggested, "MP3 (*.mp3)")
        if not path:
            return
        try:
            shutil.copyfile(self._last_output, path)
        except OSError as e:
            QMessageBox.warning(self, "보이스 스튜디오", f"MP3를 저장하지 못했습니다: {e}")
            return
        # P12.3-18: 캐시 job 폴더를 삭제하고 _last_output을 사용자 파일로 교체.
        self._cleanup_job()
        self._last_output = path
        # P17-B: 저장 후에도 내장 플레이어가 새 파일을 가리키도록 갱신.
        if self._player_ready():
            from PySide6.QtCore import QUrl
            self._player.setSource(QUrl.fromLocalFile(path))
        QMessageBox.information(self, "보이스 스튜디오", f"MP3를 저장했습니다:\n{path}")

    def closeEvent(self, event):
        """정상 종료 시 미저장 생성 결과 캐시를 정리한다(P12.3 Final Hotfix).

        무조건 release하지 않는다. 자신이 acquire한 slot만 반납하며, narrate
        worker 실행 중 종료 시에는 worker 종료를 요청하고 slot 반납은
        _on_worker_finished에서 수행한다.
        """
        if self._worker is not None:
            terminate = getattr(self._worker, "terminate", None)
            if callable(terminate):
                terminate()
                QTimer.singleShot(3000, self._worker.kill)  # terminate 실패 시 강제 종료
        self._cleanup_job()
        super().closeEvent(event)
