"""Shared styling and reusable widgets for the NeuroFence desktop app.

Professional security-console theme:
- Dark navy application shell
- High-contrast readable content
- Blue/cyan security accent
- Clear severity labels
- Existing widget APIs preserved
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Application palette
# ---------------------------------------------------------------------------

INK = "#E8EEF7"
MUTED = "#91A0B5"
SUBTLE = "#607086"

BG = "#0B1220"
PANEL = "#111B2E"
PANEL_ALT = "#162238"
BORDER = "#263750"

ACCENT = "#36A3FF"
ACCENT_HOVER = "#55B2FF"
ACCENT_SOFT = "#183A59"

SUCCESS = "#31C48D"
WARNING = "#F2B84B"
DANGER = "#EF6262"

SEVERITY_COLORS = {
    "critical": "#FF5C5C",
    "high": "#FF875C",
    "medium": "#F2B84B",
    "low": "#4EA1FF",
    "info": MUTED,
    "none": MUTED,
}

MONO = "Cascadia Mono, Consolas, Menlo, DejaVu Sans Mono, monospace"


# ---------------------------------------------------------------------------
# Global stylesheet
# ---------------------------------------------------------------------------

STYLESHEET = f"""
/* =========================================================
   Main application
   ========================================================= */

QMainWindow, QWidget {{
    background: {BG};
    color: {INK};
    font-size: 13px;
}}

QMainWindow {{
    border: none;
}}

QLabel {{
    color: {INK};
}}


/* =========================================================
   Tabs
   ========================================================= */

QTabWidget::pane {{
    border: 1px solid {BORDER};
    background: {BG};
    border-radius: 10px;
    top: -1px;
}}

QTabBar {{
    background: {BG};
}}

QTabBar::tab {{
    background: transparent;
    color: {MUTED};
    padding: 10px 18px;
    margin-right: 3px;
    border: none;
    border-bottom: 2px solid transparent;
}}

QTabBar::tab:selected {{
    color: {ACCENT};
    border-bottom: 2px solid {ACCENT};
    font-weight: 700;
}}

QTabBar::tab:hover:!selected {{
    color: {INK};
    background: {PANEL_ALT};
}}


/* =========================================================
   Buttons
   ========================================================= */

QPushButton {{
    background: {PANEL_ALT};
    color: {INK};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 8px 17px;
    min-height: 18px;
}}

QPushButton:hover {{
    background: #1B2B45;
    border-color: {ACCENT};
}}

QPushButton:pressed {{
    background: {ACCENT_SOFT};
}}

QPushButton:disabled {{
    color: {SUBTLE};
    background: #101827;
    border-color: #1D2A3D;
}}

QPushButton#primary {{
    background: {ACCENT};
    color: #07111F;
    border: 1px solid {ACCENT};
    font-weight: 700;
}}

QPushButton#primary:hover {{
    background: {ACCENT_HOVER};
    border-color: {ACCENT_HOVER};
}}

QPushButton#primary:disabled {{
    background: #35506B;
    border-color: #35506B;
    color: #8295A9;
}}


/* =========================================================
   Inputs
   ========================================================= */

QLineEdit,
QSpinBox,
QDoubleSpinBox,
QComboBox {{
    background: {PANEL};
    color: {INK};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 8px 10px;
    selection-background-color: {ACCENT};
    selection-color: #07111F;
}}

QLineEdit:hover,
QSpinBox:hover,
QDoubleSpinBox:hover,
QComboBox:hover {{
    border-color: #385171;
}}

QLineEdit:focus,
QSpinBox:focus,
QDoubleSpinBox:focus,
QComboBox:focus {{
    border-color: {ACCENT};
}}

QLineEdit[invalid="true"] {{
    border-color: {DANGER};
    background: #24151A;
}}

QComboBox QAbstractItemView {{
    background: {PANEL};
    color: {INK};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT_SOFT};
    selection-color: {INK};
}}


/* =========================================================
   Group boxes
   ========================================================= */

QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-top: 18px;
    padding: 14px;
    background: {PANEL};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 7px;
    color: {MUTED};
    background: {PANEL};
    font-weight: 700;
    font-size: 11px;
}}


/* =========================================================
   Tables
   ========================================================= */

QTableWidget {{
    background: {PANEL};
    color: {INK};
    border: 1px solid {BORDER};
    border-radius: 9px;
    gridline-color: #1D2A3D;
    alternate-background-color: {PANEL_ALT};
}}

QTableWidget::item {{
    padding: 8px;
}}

QTableWidget::item:selected {{
    background: {ACCENT_SOFT};
    color: {INK};
}}

QHeaderView::section {{
    background: #18263C;
    color: {MUTED};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 9px;
    font-weight: 700;
}}


/* =========================================================
   Text areas / logs
   ========================================================= */

QTextEdit,
QPlainTextEdit {{
    background: #0A111D;
    color: #D7E2F0;
    border: 1px solid {BORDER};
    border-radius: 9px;
    padding: 8px;
    font-family: {MONO};
    font-size: 12px;
}}

QTextEdit:focus,
QPlainTextEdit:focus {{
    border-color: {ACCENT};
}}


/* =========================================================
   Progress bar
   ========================================================= */

QProgressBar {{
    border: 1px solid {BORDER};
    border-radius: 7px;
    background: {PANEL};
    color: {INK};
    text-align: center;
    height: 20px;
}}

QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 6px;
}}


/* =========================================================
   Scroll areas
   ========================================================= */

QScrollArea {{
    border: none;
    background: {BG};
}}

QScrollBar:vertical {{
    background: {BG};
    width: 10px;
    margin: 2px;
}}

QScrollBar::handle:vertical {{
    background: #344761;
    border-radius: 5px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background: #49617F;
}}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0px;
}}


/* =========================================================
   Status bar
   ========================================================= */

QStatusBar {{
    background: #0A111D;
    border-top: 1px solid {BORDER};
    color: {MUTED};
    padding: 4px 8px;
}}


/* =========================================================
   Tooltips
   ========================================================= */

QToolTip {{
    background: #18263C;
    color: {INK};
    border: 1px solid {BORDER};
    padding: 6px 8px;
}}


/* =========================================================
   Checkboxes
   ========================================================= */

QCheckBox {{
    color: {INK};
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {BORDER};
    border-radius: 4px;
    background: {PANEL};
}}

QCheckBox::indicator:hover {{
    border-color: {ACCENT};
}}

QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}


/* =========================================================
   Radio buttons
   ========================================================= */

QRadioButton {{
    color: {INK};
    spacing: 8px;
}}


/* =========================================================
   Splitters
   ========================================================= */

QSplitter::handle {{
    background: {BORDER};
}}

QSplitter::handle:hover {{
    background: {ACCENT};
}}
"""


# ---------------------------------------------------------------------------
# Reusable Card
# ---------------------------------------------------------------------------

class Card(QFrame):
    """A modern titled panel."""

    def __init__(
        self,
        title: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.setObjectName("Card")

        self.setStyleSheet(
            f"""
            QFrame#Card {{
                background: {PANEL};
                border: 1px solid {BORDER};
                border-radius: 10px;
            }}
            """
        )

        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(18, 16, 18, 16)
        self.layout_.setSpacing(10)

        if title:
            label = QLabel(title.upper())

            label.setStyleSheet(
                f"""
                color: {MUTED};
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1px;
                border: none;
                """
            )

            self.layout_.addWidget(label)

    def add(self, widget: QWidget) -> None:
        self.layout_.addWidget(widget)


# ---------------------------------------------------------------------------
# Dashboard statistic tile
# ---------------------------------------------------------------------------

class StatTile(QFrame):
    """A large dashboard statistic tile."""

    def __init__(
        self,
        caption: str,
        value: str = "—",
        color: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)

        self.setObjectName("StatTile")

        self.setStyleSheet(
            f"""
            QFrame#StatTile {{
                background: {PANEL};
                border: 1px solid {BORDER};
                border-radius: 10px;
            }}

            QFrame#StatTile:hover {{
                border-color: #385171;
                background: {PANEL_ALT};
            }}
            """
        )

        self.setMinimumHeight(96)

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        layout = QVBoxLayout(self)

        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(5)

        self.caption_label = QLabel(caption.upper())

        self.caption_label.setStyleSheet(
            f"""
            color: {MUTED};
            font-size: 9.5px;
            font-weight: 700;
            letter-spacing: 0.9px;
            border: none;
            """
        )

        self.value_label = QLabel(value)

        font = QFont()
        font.setPointSize(21)
        font.setWeight(QFont.Weight.DemiBold)

        self.value_label.setFont(font)

        self.value_label.setStyleSheet(
            f"""
            color: {color or INK};
            border: none;
            """
        )

        self.value_label.setWordWrap(False)

        layout.addWidget(self.caption_label)
        layout.addWidget(self.value_label)
        layout.addStretch(1)

    def set_value(
        self,
        value: str,
        color: str | None = None,
    ) -> None:
        self.value_label.setText(value)

        self.value_label.setStyleSheet(
            f"""
            color: {color or INK};
            border: none;
            """
        )


# ---------------------------------------------------------------------------
# Key/value information list
# ---------------------------------------------------------------------------

class KeyValueList(QWidget):
    """Aligned label/value rows used across inspection screens."""

    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._layout = QVBoxLayout(self)

        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)

        self._rows: dict[str, QLabel] = {}

    def add_row(
        self,
        key: str,
        value: str = "—",
        mono: bool = False,
    ) -> QLabel:

        row = QWidget()

        layout = QHBoxLayout(row)

        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        key_label = QLabel(key)

        key_label.setFixedWidth(170)

        key_label.setStyleSheet(
            f"""
            color: {MUTED};
            border: none;
            font-size: 12px;
            """
        )

        key_label.setAlignment(
            Qt.AlignmentFlag.AlignRight
            | Qt.AlignmentFlag.AlignTop
        )

        value_label = QLabel(value)

        value_label.setWordWrap(True)

        value_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        style = f"""
        color: {INK};
        border: none;
        font-size: 12px;
        """

        if mono:
            style += f"""
            font-family: {MONO};
            font-size: 11.5px;
            """

        value_label.setStyleSheet(style)

        layout.addWidget(key_label)
        layout.addWidget(value_label, 1)

        self._layout.addWidget(row)

        self._rows[key] = value_label

        return value_label

    def set_value(
        self,
        key: str,
        value: str,
    ) -> None:

        if key in self._rows:
            self._rows[key].setText(value)

    def clear_values(
        self,
        placeholder: str = "—",
    ) -> None:

        for label in self._rows.values():
            label.setText(placeholder)


# ---------------------------------------------------------------------------
# Severity helper
# ---------------------------------------------------------------------------

def severity_color(severity: str) -> str:
    """Return the configured colour for a severity label."""

    return SEVERITY_COLORS.get(
        severity.lower(),
        MUTED,
    )