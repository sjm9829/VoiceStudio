"""목소리 관리 대화상자: 목록/추가/편집/삭제."""

from __future__ import annotations
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
                               QPushButton, QMessageBox, QLabel)
from ..core.errors import ProfileError

class VoiceManagerDialog(QDialog):
    def __init__(self, context, parent=None):
        super().__init__(parent)
        self.context = context
        self.setWindowTitle("목소리 관리")
        self.resize(520, 420)
        layout = QVBoxLayout(self)
        self.list = QListWidget()
        layout.addWidget(self.list)
        buttons = QHBoxLayout()
        add = QPushButton("+ 목소리 등록")
        add.clicked.connect(self.add_voice)
        edit = QPushButton("편집")
        edit.clicked.connect(self.edit_voice)
        delete = QPushButton("삭제")
        delete.clicked.connect(self.delete_voice)
        close = QPushButton("닫기")
        close.clicked.connect(self.accept)
        buttons.addWidget(add); buttons.addWidget(edit); buttons.addWidget(delete)
        buttons.addStretch(); buttons.addWidget(close)
        layout.addLayout(buttons)
        self.refresh()

    def refresh(self):
        self.list.clear()
        for p in self.context.profile_service.list_profiles():
            days = p.reference_duration_ms / 1000
            item = QListWidgetItem(f"{p.name}  ·  등록 {p.created_at[:10]}  ·  참조 {days:.1f}초")
            item.setData(Qt.UserRole, p.uuid)
            self.list.addItem(item)

    def _selected_uuid(self):
        item = self.list.currentItem()
        return item.data(Qt.UserRole) if item else None

    def add_voice(self):
        from .voice_editor_dialog import VoiceEditorDialog
        VoiceEditorDialog(self.context, self).exec()
        self.refresh()

    def edit_voice(self):
        uuid = self._selected_uuid()
        if not uuid:
            return
        from .voice_editor_dialog import VoiceEditorDialog
        VoiceEditorDialog(self.context, self, uuid).exec()
        self.refresh()

    def delete_voice(self):
        uuid = self._selected_uuid()
        if not uuid:
            return
        name = self.context.profile_service.get(uuid).name
        if QMessageBox.question(self, "삭제", f"'{name}' 목소리를 삭제할까요?") != QMessageBox.Yes:
            return
        try:
            self.context.profile_service.delete(uuid)
        except ProfileError as exc:
            QMessageBox.warning(self, "삭제 실패", exc.user_message)
        self.refresh()
