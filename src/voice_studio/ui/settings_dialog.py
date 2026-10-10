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
        import logging
        from ..core.errors import VoiceStudioError
        try:
            path = self.model_manager.download(force=self.force)
            # GGUF 백엔드 선택 시 llama.cpp 엔진도 함께 준비한다(P17-H2).
            # 기존 정상 엔진이 있으면 재다운로드하지 않는다.
            engine_ready = getattr(self.model_manager, "is_engine_ready", None)
            if engine_ready is not None and not engine_ready():
                download_engine = getattr(self.model_manager, "download_engine", None)
                if download_engine is not None:
                    download_engine()
            self.done.emit(str(path))
        except VoiceStudioError as exc:
            # 앱 오류는 검증된 사용자 안내만 노출하고 기술 상세는 로그로 남긴다(P13 hotfix).
            logging.getLogger(__name__).warning("모델 다운로드 실패: %s", exc.detail, exc_info=True)
            self.failed.emit(exc.user_message)
        except Exception as exc:
            # 내부 예외('NoneType' object has no attribute 'write' 등)를 그대로 노출하지 않는다.
            logging.getLogger(__name__).warning("모델 다운로드 실패: %s", exc, exc_info=True)
            self.failed.emit("음성 모델을 받지 못했습니다. 인터넷 연결을 확인한 뒤 다시 시도해 주세요.")


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
        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("음성 엔진"))
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("기본 (공식 Qwen3-TTS 0.6B)", "official")
        self.backend_combo.addItem("고정 GGUF 실험판 (1.7B Q8_0, llama.cpp)", "gguf")
        saved_backend = str(context.settings.get("tts_backend", "official"))
        idx_b = self.backend_combo.findData(saved_backend)
        self.backend_combo.setCurrentIndex(idx_b if idx_b >= 0 else 0)
        self.backend_combo.currentIndexChanged.connect(self._backend_changed)
        backend_row.addWidget(self.backend_combo, 1)
        model_layout.addLayout(backend_row)
        from ..services.gguf_model_manager import GgufModelManager
        _saved = str(context.settings.get("tts_backend", "official"))
        self.model_label = QLabel(
            GgufModelManager().status_text() if _saved == "gguf"
            else context.model_manager.status_text())
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
        from ..infra.subprocess_runner import run as _run_no_window
        from ..workers.launcher import apply_no_window, diagnostics_command
        try:
            program, dargs = diagnostics_command()
            r = _run_no_window([program, *dargs],
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


    def _current_manager(self):
        """콤보 선택에 맞는 모델 관리자 인스턴스(저장 전 미리보기용)."""
        from ..services.gguf_model_manager import GgufModelManager
        if self.backend_combo.currentData() == "gguf":
            return GgufModelManager()
        return ModelManager()

    def _backend_changed(self, *_):
        """백엔드 전환 시 상태 표시만 갱신한다. 실제 적용은 저장(close) 시 확정한다."""
        self.model_label.setText(self._current_manager().status_text())

    def _select_saved_quality(self):
        saved = self.context.settings.get("mp3_bitrate_kbps", 192)
        index = self.quality.findData(int(saved))
        self.quality.setCurrentIndex(index if index >= 0 else 1)

    def download_model(self, force=False):
        """모델 다운로드를 별도 스레드로 실행해 UI freeze를 막는다(P12.2-14, P17-H2).

        저장 전이라도 콤보에서 선택한 백엔드의 모델을 받는다(_current_manager).
        GGUF 선택 시 엔진도 함께 준비한다(_ModelDownloadThread 내 처리).
        """
        if getattr(self, "_dl_thread", None) is not None and self._dl_thread.isRunning():
            return  # 중복 클릭 무시
        manager = self._current_manager()
        self._dl_backend = self.backend_combo.currentData()
        self.dl_progress.setVisible(True)
        self.model_label.setText(manager.status_text())
        for b in self.download_buttons:
            b.setEnabled(False)
        self._dl_thread = _ModelDownloadThread(manager, force=force)
        self._dl_thread.done.connect(self._on_download_done)
        self._dl_thread.failed.connect(self._on_download_failed)
        self._dl_thread.start()

    def _on_download_done(self, path: str):
        self.dl_progress.setVisible(False)
        for b in self.download_buttons:
            b.setEnabled(True)
        # 방금 다운로드한 백엔드의 상태를 표시한다(저장 전 선택 기준, P17-H2).
        from ..services.gguf_model_manager import GgufModelManager
        if getattr(self, "_dl_backend", None) == "gguf":
            self.model_label.setText(GgufModelManager().status_text())
        else:
            self.model_label.setText(ModelManager().status_text())

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
            "tts_backend": str(self.backend_combo.currentData()),
        })
        # 백엔드 변경을 즉시 반영해 이후 생성 job이 새 백엔드로 진행되게 한다(P17-C).
        from ..services.gguf_model_manager import GgufModelManager
        backend = self.context.settings.get("tts_backend", "official")
        if backend == "gguf" and not isinstance(self.context.model_manager, GgufModelManager):
            self.context.model_manager = GgufModelManager()
        # P18-1: 단일 gguf - 어떤 저장값이든 manager 교체 없이 GgufModelManager 유지.
        if not isinstance(self.context.model_manager, GgufModelManager):
            self.context.model_manager = GgufModelManager()
        self.accept()
