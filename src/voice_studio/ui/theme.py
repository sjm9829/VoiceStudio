"""P17-B 공통 디자인 시스템: 밝고 차분한 기본 테마(QSS).

색상·여백·모서리 반경·버튼 위계를 이 모듈 하나로 통일한다. 개별 위젯마다
스타일을 하드코딩하지 않고 objectName/속성 선택자로만 꾸민다.
"""

from __future__ import annotations

# 팔레트
BG = "#f4f5f7"          # 앱 배경
CARD = "#ffffff"        # 카드 배경
BORDER = "#e2e5ea"      # 카드/입력 테두리
TEXT = "#22262e"        # 기본 텍스트
MUTED = "#697180"       # 보조 텍스트
PRIMARY = "#2f6fed"     # 주요 동작(파랑)
PRIMARY_HOVER = "#245ad0"
DANGER = "#d64545"      # 취소/삭제
SUCCESS = "#2f9e5f"     # 완료 상태
SELECT_BG = "#e8f0fe"   # 선택 강조

FONT_STACK = '"맑은 고딕", "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR", "Segoe UI", sans-serif'

_RADIUS = "8px"
_PAD = "12px"

_APP_QSS = f"""
* {{
    font-family: {FONT_STACK};
    font-size: 10pt;
    color: {TEXT};
}}
QMainWindow, QDialog {{ background: {BG}; }}

/* ---- 카드 ---- */
QFrame#card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: {_RADIUS};
}}

/* ---- 단계 라벨 ---- */
QLabel#stepLabel {{
    font-size: 10pt;
    font-weight: 600;
    color: {TEXT};
}}
QLabel#hintLabel, QLabel#mutedLabel {{
    color: {MUTED};
    font-size: 9pt;
}}
QLabel#headerTitle {{
    font-size: 14pt;
    font-weight: 700;
    color: {TEXT};
}}
QLabel#headerSubtitle {{
    color: {MUTED};
    font-size: 9pt;
}}
QLabel#statusLabel {{
    color: {TEXT};
    font-size: 10pt;
}}
QLabel#statusDone {{ color: {SUCCESS}; font-weight: 600; }}
QLabel#statusError {{ color: {DANGER}; font-weight: 600; }}

/* ---- 버튼 위계 ---- */
QPushButton {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 14px;
}}
QPushButton:hover {{ background: #f0f2f5; }}
QPushButton:disabled {{ color: #a9b0ba; background: #f7f8fa; }}
QPushButton#primary {{
    background: {PRIMARY};
    color: #ffffff;
    border: none;
    font-weight: 600;
    padding: 8px 18px;
}}
QPushButton#primary:hover {{ background: {PRIMARY_HOVER}; }}
QPushButton#primary:disabled {{ background: #b9cdf6; color: #ffffff; }}
QPushButton#danger {{
    color: {DANGER};
    border: 1px solid #f0c4c4;
    background: #fff7f7;
}}
QPushButton#danger:hover {{ background: #fdeaea; }}

/* ---- 입력 ---- */
QPlainTextEdit, QLineEdit, QComboBox {{
    background: #ffffff;
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: {SELECT_BG};
    selection-color: {TEXT};
}}
QPlainTextEdit:focus, QLineEdit:focus, QComboBox:focus {{ border: 1px solid {PRIMARY}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}

/* ---- 진행/슬라이더 ---- */
QProgressBar {{
    background: #e9ecf1;
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{ background: {PRIMARY}; border-radius: 4px; }}
QSlider::groove:horizontal {{
    height: 6px; background: #e2e6ec; border-radius: 3px;
}}
QSlider::sub-page:horizontal {{ background: {PRIMARY}; border-radius: 3px; }}
QSlider::handle:horizontal {{
    width: 14px; height: 14px; margin: -4px 0;
    background: {CARD}; border: 2px solid {PRIMARY}; border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background: {SELECT_BG}; }}

/* ---- 목록 ---- */
QListWidget {{
    background: #ffffff;
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px;
}}
QListWidget::item {{ padding: 8px 10px; border-radius: 6px; }}
QListWidget::item:selected {{ background: {SELECT_BG}; color: {TEXT}; }}
QListWidget::item:hover:!selected {{ background: #f2f4f8; }}

/* ---- 기타 ---- */
QCheckBox {{ spacing: 6px; }}
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: {_RADIUS};
    margin-top: 10px;
    background: {CARD};
    font-weight: 600;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; top: 2px; padding: 0 4px; }}
"""


def apply_app_style(app) -> None:
    """애플리케이션 전체에 기본 테마를 적용한다(P17-B)."""
    app.setStyleSheet(_APP_QSS)


def make_card(parent=None) -> "QFrame":
    """공통 카드 프레임. 내용은 카드 안에 QVBoxLayout으로 배치한다."""
    from PySide6.QtWidgets import QFrame, QVBoxLayout
    card = QFrame(parent)
    card.setObjectName("card")
    inner = QVBoxLayout(card)
    inner.setContentsMargins(16, 12, 16, 12)
    inner.setSpacing(8)
    return card


def step_label(text: str) -> "QLabel":
    from PySide6.QtWidgets import QLabel
    label = QLabel(text)
    label.setObjectName("stepLabel")
    return label


def hint_label(text: str) -> "QLabel":
    from PySide6.QtWidgets import QLabel
    label = QLabel(text)
    label.setObjectName("hintLabel")
    label.setWordWrap(True)
    return label


def primary_button(text: str, parent=None) -> "QPushButton":
    from PySide6.QtWidgets import QPushButton
    button = QPushButton(text, parent)
    button.setObjectName("primary")
    return button
