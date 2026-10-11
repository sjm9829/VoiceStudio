"""메인 나레이션 화면.

worker 실행 계약:
- worker 프로세스는 개발 환경에서 `python -m voice_studio.main --worker job.json`,
  frozen 환경에서 `VoiceStudio.exe --worker job.json`으로 실행된다(workers/launcher).
- stdout JSONL은 청크 단위로 잘려 올 수 있으므로 per-process receive buffer(JsonlBuffer)로 조립한다.
- stderr는 별도로 수집해 진단 로그로 남긴다.
"""

from __future__ import annotations
import datetime
import threading
import logging
import os
import shutil
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QProcess, QTimer, QThread, QUrl, Signal, QObject
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
                               QPushButton, QPlainTextEdit, QLabel, QFileDialog, QMessageBox,
                               QProgressBar, QFrame, QSlider)
from .theme import (BG, CARD, BORDER, TEXT, MUTED, PRIMARY, PRIMARY_HOVER, DANGER,
                    SUCCESS, SELECT_BG, make_card, step_label, hint_label, primary_button)
from ..core.errors import VoiceStudioError, ProfileError, ModelNotDownloadedError
from ..core import config
from ..core.paths import default_mp3_dir
from ..core.paths import logs_dir
from ..workers.job_schema import build_narrate_payload

_logger = logging.getLogger(__name__)
from ..workers.launcher import apply_no_window, worker_command
from ..core.paths import safe_job_cache_dir
from ..workers.linebuffer import JsonlBuffer
from .voice_manager_dialog import VoiceManagerDialog

_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL",
                     *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}

PHASE_KO = {
    "model_loading": "모델을 준비하는 중…",
    "analyzing_reference": "목소리 특징을 분석하는 중…",
    "saving_profile": "목소리를 저장하는 중…",
    "generating": "구간 {i}/{n} 만드는 중…",
    "encoding": "MP3를 만드는 중…",
    "done": "완료",
}

_DETACHED_PREPARE: set = set()  # 닫힌 창의 준비 스레드가 끝날 때까지 참조 유지(GC 방지)

class ModelVerifyWorker(QObject):
    """P18-G2: 이미 준비된 모델/엔진의 무결성 검증만 백그라운드로 수행.

    UI 스레드에서 수 GB SHA-256을 계산하지 않도록 note_verified 서명을
    백그라운드에서 확정한다. 다운로드는 시도하지 않는다.
    """

    progress = Signal(dict)
    finished = Signal(bool, str)

    def __init__(self, model_manager):
        super().__init__()
        self._mm = model_manager

    def run(self):
        try:
            # 검증: is_downloaded()는 내부적으로 해시 재검증을 수행한다.
            chk = getattr(self._mm, "is_downloaded", None)
            if chk is not None and not chk():
                # 서명 불일치/손상: 재다운로드 없이 실패로 보고한다(사용자 재준비 필요).
                self.finished.emit(False, "음성 모델 검증에 실패했습니다. 설정에서 다시 받아 주세요.")
                return
            note = getattr(self._mm, "note_verified", None)  # P18-G2: 서명 확정
            if note is not None:
                note()
            eng = getattr(self._mm, "is_engine_ready", None)
            if eng is not None and not eng():
                self.finished.emit(False, "음성 엔진이 준비되지 않았습니다.")
                return
            self.finished.emit(True, "")
        except Exception as e:  # noqa: BLE001 - UI 표시용 문자열 변환
            self.finished.emit(False, str(e))


class ModelPrepareWorker(QObject):
    """P18-5: 백그라운드에서 모델+엔진 준비. progress(dict), finished(bool,str) 시그널."""

    progress = Signal(dict)
    finished = Signal(bool, str)

    def __init__(self, model_manager):
        super().__init__()
        self._mm = model_manager

    def run(self):
        try:
            if not self._mm.is_downloaded():
                self._mm.download(progress_cb=self.progress.emit)
            if not self._mm.is_engine_ready():
                self._mm.download_engine(progress_cb=self.progress.emit)
            note = getattr(self._mm, "note_verified", None)  # P18-G: 검증 서명 기록(UI 재사용)
            if note is not None:
                note()
            self.finished.emit(True, "")
        except Exception as e:  # noqa: BLE001 - UI 표시용 문자열 변환
            self.finished.emit(False, str(e))

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
        self._dl_thread: QThread | None = None
        self._dl_worker = None
        self._dl_callbacks: list = []
        self._retry_btn = None
        self._gen_voice_uuid: str | None = None
        self._gen_voice_name: str | None = None
        self._gen_finished_at: datetime.datetime | None = None
        self._build_ui()
        self._refresh_voices()
        self._refresh_save_location_label()
        # P18-G2: 준비 여부와 무관하게 백그라운드 무결성 검증을 시작한다.
        # quickly_ready() 통과 시에도 최초 생성 시 UI 스레드 SHA-256 멈춤을 막기 위해
        # 백그라운드에서 note_verified 서명을 미리 확정해 둔다(다운로드 완료면 생략).
        self._startup_model_check(verify_only=True)

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

        # P18-F4: 저장 위치 표시 + 폴더 선택(설정 메뉴 없이 메인 화면에서 변경)
        saveloc_row = QHBoxLayout()
        saveloc_title = QLabel("저장 위치")
        saveloc_title.setObjectName("hintLabel")
        self.save_location_label = QLabel("")
        self.save_location_label.setObjectName("mutedLabel")
        self.save_location_label.setWordWrap(True)
        choose_dir_btn = QPushButton("폴더 선택")
        choose_dir_btn.clicked.connect(self.choose_output_dir)
        saveloc_row.addWidget(saveloc_title)
        saveloc_row.addWidget(self.save_location_label, 1)
        saveloc_row.addWidget(choose_dir_btn)
        result_layout.addLayout(saveloc_row)

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

    def _save_dir(self) -> Path:
        return Path(self.context.settings.get("mp3_output_dir", "") or default_mp3_dir())

    def _refresh_save_location_label(self):
        self.save_location_label.setText(str(self._save_dir()))

    def choose_output_dir(self):
        start = str(self._save_dir())
        chosen = QFileDialog.getExistingDirectory(self, "저장 폴더 선택", start)
        if not chosen:
            return
        try:
            Path(chosen).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.warning(self, "보이스 스튜디오", f"저장 폴더를 만들 수 없습니다: {e}")
            return
        self.context.save_settings({**self.context.settings, "mp3_output_dir": chosen})
        self._refresh_save_location_label()

    def _startup_model_check(self, verify_only: bool = False):
        """P18-F2: 앱 시작 시 백그라운드로 모델/엔진 확인·자동 준비. UI는 먼저 표시된다.

        P18-G2: verify_only=True이면 이미 준비된 모델의 무결성 검증만 백그라운드로
        수행해 note_verified 서명을 확정한다(다운로드 재시도 없음).
        """
        if verify_only and self._model_ready():
            QTimer.singleShot(0, lambda: self._ensure_model_async(
                self._on_startup_model_ready, verify_only=True))
            return
        QTimer.singleShot(0, lambda: self._ensure_model_async(self._on_startup_model_ready))

    def _on_startup_model_ready(self, ok: bool):
        if ok and self.isVisible():
            self.status_label.setText("음성 생성 준비 완료")

    def open_manager(self):
        dlg = VoiceManagerDialog(self.context, self)
        dlg.exec()
        self._refresh_voices()

        # ---- 생성 ----
    def _remember_generation_voice(self, profile_uuid: str):
        """P18-F5: 생성 시작 시점의 프로필 uuid/이름을 저장해 저장 시 이름 고정."""
        self._gen_voice_uuid = profile_uuid
        idx = self.voice_combo.findData(profile_uuid)
        self._gen_voice_name = self.voice_combo.itemText(idx) if idx >= 0 else None

    def start_generation(self):
        profile_uuid = self.voice_combo.currentData()
        if profile_uuid is None:
            QMessageBox.information(self, "보이스 스튜디오", "목소리를 먼저 등록해 주세요.")
            return
        self._remember_generation_voice(profile_uuid)
        script = self.script_edit.toPlainText().strip()
        if not script:
            QMessageBox.information(self, "보이스 스튜디오", "대본을 입력해 주세요.")
            return
        if not self._model_ready() or not self._verified_signature_valid():
            # P18-G2: 준비 안 됐거나 검증 서명이 없으면 백그라운드 검증/준비에 합류하고,
            # 완료 시 입력된 대본/목소리 그대로 단 한 번 생성을 이어서 수행한다.
            self._ensure_model_async(
                lambda ok: self._continue_generation(profile_uuid, script) if ok else None,
                verify_only=self._model_ready())
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
            bitrate_kbps=int(config.MP3_BITRATE_KBPS),  # P18-7 고정
            model_path=self.context.model_manager.model_path(),
            tts_backend="gguf",  # P18-1 단일 백엔드
        )

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
            self._gen_finished_at = datetime.datetime.now()  # P18-F5
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
        out_dir = self.context.settings.get("mp3_output_dir", "")  # P18-F4: 실제 저장 위치 우선
        if out_dir:
            targets.append(out_dir)
        if self._last_output and Path(self._last_output).exists():
            targets.append(str(Path(self._last_output).parent))
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
        """P18-8: 대화상자 없이 목소리명_YYYYMMDD_HHMMSS.mp3로 자동 저장."""
        if not self._last_output:
            return
        path = self._auto_save_path()
        if path is None:
            QMessageBox.warning(self, "보이스 스튜디오", "저장 폴더를 만들 수 없습니다.")
            return
        tmp_path = Path(str(path) + ".part~")  # P18-G: 같은 dir 임시파일 원자 게시
        try:
            with open(tmp_path, "wb") as _t:
                pass
            shutil.copyfile(self._last_output, tmp_path)
            with open(tmp_path, "rb+") as _t:
                _t.flush()
                os.fsync(_t.fileno())
            os.replace(tmp_path, path)  # 예약된 빈 final을 완성본으로 원자 교체
        except OSError as e:
            # P18-F4/G: 실패 시 임시/예약 파일만 정리. 기존 사용자 MP3는 훼손 없음.
            for stray in (tmp_path, Path(path)):
                try:
                    stray.unlink(missing_ok=True)
                except OSError:
                    pass
            QMessageBox.warning(self, "보이스 스튜디오", f"MP3를 저장하지 못했습니다: {e}")
            return
        # P12.3-18: 캐시 job 폴더를 삭제하고 _last_output을 사용자 파일로 교체.
        self._cleanup_job()
        self._last_output = path
        # P17-B: 저장 후에도 내장 플레이어가 새 파일을 가리키도록 갱신.
        if self._player_ready():
            from PySide6.QtCore import QUrl
            self._player.setSource(QUrl.fromLocalFile(str(path)))
        QMessageBox.information(self, "보이스 스튜디오", f"MP3를 저장했습니다:\n{path}")


    def _auto_save_path(self):
        """P18-F5: 목소리명_YYYYMMDD_HHMMSS.mp3. 생성 시작 시점 이름 고정, 금지문자/예약명 처리, 중복 시 _2/_3, 덮어쓰기 없음."""
        import re as _re
        name = self._gen_voice_name or "나레이션"
        safe = _re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name).strip().rstrip(".") or "나레이션"
        if safe.upper() in _WINDOWS_RESERVED:  # Windows 예약 파일명
            safe = f"_{safe}"
        out_dir = self._save_dir()
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        stamp = (self._gen_finished_at or datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")

        def _exclusive(path: Path) -> Path | None:
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)  # 덮어쓰기 없는 예약
            except FileExistsError:
                return None
            except OSError as exc:
                # P18-G: O_EXCL 실패 시 기존 path 반환 폴백 금지 - 기존 파일 덮어쓰기 위험.
                _logger.warning("MP3 예약 실패 %s: %s", path, exc)
                return None
            os.close(fd)
            return path

        cand = _exclusive(out_dir / f"{safe}_{stamp}.mp3")
        n = 2
        while cand is None and n <= 99:
            cand = _exclusive(out_dir / f"{safe}_{stamp}_{n}.mp3")
            n += 1
        return cand

    def _verified_signature_valid(self) -> bool:
        """P18-G2: 백그라운드 검증 서명이 유효한지(UI 스레드 해시 생략 가능) 확인.

        서명이 있으면 저비용 비교만 수행한다. 서명 부재 시 False: verify-only
        워커가 백그라운드에서 note_verified()를 확정하게 한다.
        """
        mm = self.context.model_manager
        csig = getattr(mm, "_content_signature", None)
        if csig is None:
            # 시그니처 메커니즘이 없는 관리자(테스트 스텁 등): 기존 동작 유지.
            return True
        sig = getattr(mm, "_verified_sig", None)
        if sig is None:
            return False
        try:
            return csig() == sig
        except OSError:
            return False

    def _model_ready(self) -> bool:
        """P18-5: 모델+엔진 준비 여부."""
        mm = self.context.model_manager
        # P18-F2: UI 스레드에서 수 GB SHA-256 방지 - 저비용 확인 우선.
        quick = getattr(mm, "quickly_ready", None)
        if quick is not None:
            return bool(quick())
        ready = getattr(mm, "is_downloaded", None)
        if ready is not None and not ready():
            return False
        eng_ok = getattr(mm, "is_engine_ready", None)
        if eng_ok is not None and not eng_ok():
            return False
        return True

    def _ensure_model_async(self, done, verify_only: bool = False):
        """P18-5: 모델/엔진 미준비 시 자동 다운로드(무결성 검증 포함). 실패 시 다시 시도 버튼.

        P18-G2: verify_only=True이면 이미 준비된 모델의 검증만 수행하는 경량 워커를 쓴다.
        다운로드가 전혀 필요 없으므로 진행률 대신 검증 상만 표시한다.
        """
        if self._dl_thread is not None:  # P18-F2: 단일 준비 작업 공유(중복 다운로드 금지)
            self._dl_callbacks.append(done)
            return
        self.generate_btn.setEnabled(False)
        self.status_label.setText(
            "음성 모델을 확인하는 중..." if verify_only else "음성 모델을 준비하는 중...")
        mm = self.context.model_manager
        if not verify_only:
            try:  # P18-F2: 다운로더 모듈을 UI 스레드에서 미리 적재(백그라운드 import 경합 방지)
                from huggingface_hub import HfApi, hf_hub_download  # noqa: F401
            except Exception:  # noqa: BLE001 - 오프라인 환경 허용, worker에서 재시도
                pass
        worker = ModelVerifyWorker(mm) if verify_only else ModelPrepareWorker(mm)
        worker.progress.connect(self._on_model_progress)
        worker.finished.connect(self._on_model_finished)
        self._dl_thread = threading.Thread(
            target=worker.run, name="voice-prepare", daemon=True)
        self._dl_worker = worker
        self._dl_done = done
        self._dl_callbacks = [done]
        self._dl_thread.start()

    def _on_model_progress(self, payload: dict):
        stage = payload.get("stage", "")
        done, total = payload.get("done", 0), payload.get("total", 0)
        label = {"download": "음성 모델", "engine": "음성 엔진", "verify": "음성 모델 검증"}.get(stage, stage)
        if stage == "verify":
            # P18-F4: 검증 단계는 결정적(busy) 표시 - 가짜 퍼센트 금지.
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)
            self.status_label.setText(f"{label} 중...")
            return
        self.progress.setVisible(True)
        if total > 0:
            self.progress.setRange(0, 100)
            pct = min(int(done * 100 / total), 100)
            mb = done / 1e6
            tmb = total / 1e6
            self.status_label.setText(f"{label} 받는 중... {pct}% ({mb:.0f}/{tmb:.0f}MB)")
            if stage == "download":
                self.progress.setValue(pct)
            else:
                self.progress.setValue(pct)
        else:
            self.progress.setRange(0, 0)  # 총량 미지: busy 표시(임의 % 금지)
            self.status_label.setText(f"{label} 받는 중...")

    def _on_model_finished(self, ok: bool, err: str):
        if self._dl_thread is None:  # closeEvent 등으로 이미 정리된 경우 무시
            return
        th = self._dl_thread
        self._dl_thread = None
        if th is not None and th is not threading.current_thread() and th.is_alive():
            th.join(timeout=2.0)
        self._dl_worker = None
        self.progress.setVisible(False)
        self.progress.setValue(0)
        self.generate_btn.setEnabled(True)
        if ok:
            self._clear_model_retry()
            self.status_label.setText("음성 생성 준비 완료")
            callbacks, self._dl_callbacks = self._dl_callbacks, []
            self._dl_done = None
            for cb in callbacks:
                if cb:
                    cb(True)
        else:
            self._dl_callbacks = []  # P18-F2: 재시도 버튼으로 재요청(대기 콜백 폐기)
            self.status_label.setText(f"모델 준비 실패: {err}")
            self._show_model_retry()

    def _show_model_retry(self):
        """P18-5: 실패 시 다시 시도 버튼(상태 라벨 옆)."""
        if getattr(self, "_retry_btn", None) is not None:
            return
        btn = QPushButton("다시 시도")
        btn.clicked.connect(self._retry_model_prepare)
        header = self.status_label.parentWidget().layout()
        header.insertWidget(header.indexOf(self.status_label) + 1, btn)
        self._retry_btn = btn

    def _clear_model_retry(self):
        btn = getattr(self, "_retry_btn", None)
        if btn is not None:
            btn.setParent(None)
            btn.deleteLater()
            self._retry_btn = None

    def _retry_model_prepare(self):
        self._clear_model_retry()
        self._ensure_model_async(lambda ok: self._on_startup_model_ready(ok))

    def _continue_generation(self, profile_uuid: str, script: str):
        if not self.isVisible():
            return
        self._remember_generation_voice(profile_uuid)
        try:
            if not self._start_worker(self._build_job(profile_uuid, script)):
                return
        except VoiceStudioError as e:
            QMessageBox.warning(self, "보이스 스튜디오", str(e))
        except OSError as e:
            QMessageBox.warning(self, "보이스 스튜디오", f"작업 파일을 준비할 수 없습니다: {e}")

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
        # P18-F2: 실행 중 모델 준비 스레드 정리(파괴 중 running thread 방지).
        th = getattr(self, "_dl_thread", None)
        if th is not None and th.is_alive():
            worker = getattr(self, "_dl_worker", None)
            for sig_name in ("progress", "finished"):
                sig = getattr(worker, sig_name, None) if worker is not None else None
                if sig is not None:
                    try:
                        sig.disconnect()
                    except (RuntimeError, TypeError):
                        pass
            th.join(timeout=2.0)
        self._dl_thread = None
        self._dl_worker = None
        self._cleanup_job()
        super().closeEvent(event)