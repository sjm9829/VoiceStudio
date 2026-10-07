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
from pathlib import Path

from PySide6.QtCore import Qt, QProcess, QTimer
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
                               QPushButton, QPlainTextEdit, QLabel, QFileDialog, QMessageBox,
                               QProgressBar)
from ..core.errors import VoiceStudioError, ProfileError, ModelNotDownloadedError
from ..core.paths import logs_dir
from ..workers.job_schema import build_narrate_payload
from ..workers.launcher import worker_command
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
            self.play_btn.setEnabled(True)
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
