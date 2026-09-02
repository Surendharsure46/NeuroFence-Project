"""Application screens.

Each screen is a self-contained widget with an ``update_from_result`` method,
so the main window only has to broadcast a scan result and never needs to know
how any individual screen renders it.

All displayed values come from the scan. Before a scan runs, fields show an
em-dash rather than zeros — "0 anomalies" and "not yet measured" mean very
different things, and a security tool must not blur them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..reporting.schemas import VERDICT_LABELS
from .theme import (
    ACCENT,
    BORDER,
    INK,
    MUTED,
    Card,
    KeyValueList,
    StatTile,
    severity_color,
)
from .worker import STAGES, ScanRequest


def _scrollable(inner: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(inner)
    return area


class DashboardScreen(QWidget):
    """Top-level summary of the most recent scan."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        title = QLabel("NeuroFence")
        title.setStyleSheet(
            f"font-size: 27px; font-weight: 700; color: {INK}; border: none;"
        )
        subtitle = QLabel("LLM Weight Poisoning & Backdoor Scanner — offline forensic tool")
        subtitle.setStyleSheet(f"color: {MUTED}; border: none;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self.verdict_label = QLabel("No scan has been run yet.")
        self.verdict_label.setStyleSheet(
            f"font-size: 16px; font-weight: 600; color: {MUTED}; border: none; padding: 4px 0;"
        )
        layout.addWidget(self.verdict_label)

        grid = QGridLayout()
        grid.setSpacing(10)
        self.tiles = {
            "risk": StatTile("Experimental risk score"),
            "level": StatTile("Risk level"),
            "prompts": StatTile("Prompts scanned"),
            "anomalies": StatTile("Anomalous readings"),
            "layers": StatTile("Suspicious layers"),
            "consistency": StatTile("Reproducibility"),
        }
        for index, tile in enumerate(self.tiles.values()):
            grid.addWidget(tile, index // 3, index % 3)
        layout.addLayout(grid)

        card = Card("Scan context")
        self.details = KeyValueList()
        for key in ("Model", "Model hash", "Scan timestamp", "Candidate trigger", "Determinism"):
            self.details.add_row(key, "—", mono=key == "Model hash")
        card.add(self.details)
        layout.addWidget(card)

        self.caveat_card = Card("Caveats")
        self.caveat_label = QLabel("—")
        self.caveat_label.setWordWrap(True)
        self.caveat_label.setStyleSheet(f"color: {MUTED}; border: none;")
        self.caveat_card.add(self.caveat_label)
        layout.addWidget(self.caveat_card)

        note = QLabel(
            "The risk score expresses evidence strength for the trigger hypothesis, "
            "not probability of compromise. NeuroFence measures activation anomalies only."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {MUTED}; font-size: 11px; border: none;")
        layout.addWidget(note)
        layout.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scrollable(inner))

    def update_from_result(self, result: Any, report: Any) -> None:
        risk = result.risk
        trigger = result.trigger_result
        color = severity_color(risk.severity)

        self.verdict_label.setText(VERDICT_LABELS.get(trigger.verdict, trigger.verdict))
        self.verdict_label.setStyleSheet(
            f"font-size: 16px; font-weight: 600; color: {color}; border: none; padding: 4px 0;"
        )

        self.tiles["risk"].set_value(f"{risk.score:.1f}", color)
        self.tiles["level"].set_value(risk.severity.upper(), color)
        self.tiles["prompts"].set_value(str(len(result.prompt_set)))
        self.tiles["anomalies"].set_value(str(report.anomaly_count))
        self.tiles["layers"].set_value(
            ", ".join(map(str, report.suspicious_layers)) or "none"
        )
        self.tiles["consistency"].set_value(f"{trigger.consistency:.0%}")

        self.details.set_value("Model", result.loaded.metadata.model_name)
        self.details.set_value("Model hash", result.loaded.fingerprint.model_sha256)
        self.details.set_value("Scan timestamp", report.scan_timestamp)
        self.details.set_value("Candidate trigger", trigger.trigger_token)
        self.details.set_value(
            "Determinism",
            "not measured" if trigger.determinism is None else f"{trigger.determinism:.2f}",
        )
        self.caveat_label.setText(
            "\n".join(f"• {c}" for c in risk.caveats) if risk.caveats else "None recorded."
        )


class ModelInspectionScreen(QWidget):
    """Model identity, fingerprint, and environment."""

    FIELDS = (
        "Model identifier",
        "Local path",
        "SHA-256 (full)",
        "SHA-256 (weights)",
        "File size",
        "Architecture",
        "Model type",
        "Layer count",
        "Hidden size",
        "Attention heads",
        "Parameter count",
        "Dtype / device",
        "Safetensors",
        "PyTorch version",
        "Transformers version",
        "Python version",
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        card = Card("Model inspection")
        self.details = KeyValueList()
        for field in self.FIELDS:
            self.details.add_row(field, "—", mono="SHA-256" in field or field == "Local path")
        card.add(self.details)
        layout.addWidget(card)

        files_card = Card("Model files")
        self.files_table = QTableWidget(0, 3)
        self.files_table.setHorizontalHeaderLabels(["File", "Size", "SHA-256"])
        self.files_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.files_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.files_table.setMinimumHeight(180)
        files_card.add(self.files_table)
        layout.addWidget(files_card)
        layout.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scrollable(inner))

    def update_from_result(self, result: Any, report: Any) -> None:
        meta = result.loaded.metadata
        fingerprint = result.loaded.fingerprint

        def show(value: Any) -> str:
            return "—" if value is None else str(value)

        values = {
            "Model identifier": meta.model_name,
            "Local path": meta.local_path,
            "SHA-256 (full)": fingerprint.model_sha256,
            "SHA-256 (weights)": fingerprint.weights_sha256,
            "File size": report.model.size_human,
            "Architecture": show(meta.architecture),
            "Model type": show(meta.model_type),
            "Layer count": show(meta.layers),
            "Hidden size": show(meta.hidden_size),
            "Attention heads": show(meta.attention_heads),
            "Parameter count": f"{meta.parameters:,}" if meta.parameters else "—",
            "Dtype / device": f"{show(meta.dtype)} / {show(meta.device)}",
            "Safetensors": str(meta.safetensors_present),
            "PyTorch version": show(meta.torch_version),
            "Transformers version": show(meta.transformers_version),
            "Python version": show(meta.python_version),
        }
        for key, value in values.items():
            self.details.set_value(key, value)

        self.files_table.setRowCount(len(fingerprint.files))
        for row, digest in enumerate(fingerprint.files):
            size = f"{digest.size_bytes / 1024:.1f} KB"
            for column, text in enumerate(
                (digest.relative_path, size, digest.sha256[:32] + "…")
            ):
                self.files_table.setItem(row, column, QTableWidgetItem(text))
        self.files_table.resizeColumnsToContents()
        self.files_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )


class ScanConfigScreen(QWidget):
    """Scan configuration with input validation."""

    scan_requested = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        # --- model
        model_box = QGroupBox("Model")
        model_layout = QGridLayout(model_box)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Path to a local Hugging Face model directory")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        model_layout.addWidget(QLabel("Model directory"), 0, 0)
        model_layout.addWidget(self.path_edit, 0, 1)
        model_layout.addWidget(browse, 0, 2)
        layout.addWidget(model_box)

        # --- fuzzing
        fuzz_box = QGroupBox("Fuzzing")
        fuzz_layout = QGridLayout(fuzz_box)
        self.trigger_edit = QLineEdit("PINEAPPLE")
        self.controls_edit = QLineEdit("APPLE, BANANA, ORANGE, MANGO")
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(42)
        self.normal_spin = self._count_spin(100)
        self.trigger_spin = self._count_spin(50)
        self.control_spin = self._count_spin(25)

        rows = [
            ("Candidate trigger", self.trigger_edit),
            ("Control tokens", self.controls_edit),
            ("Seed", self.seed_spin),
            ("Baseline (normal) prompts", self.normal_spin),
            ("Trigger prompts", self.trigger_spin),
            ("Control prompts per token", self.control_spin),
        ]
        for index, (label, widget) in enumerate(rows):
            fuzz_layout.addWidget(QLabel(label), index, 0)
            fuzz_layout.addWidget(widget, index, 1)
        layout.addWidget(fuzz_box)

        # --- detection
        detect_box = QGroupBox("Detection")
        detect_layout = QGridLayout(detect_box)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["robust", "zscore"])
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 100.0)
        self.threshold_spin.setSingleStep(0.5)
        self.threshold_spin.setValue(3.0)
        self.layers_edit = QLineEdit("all")
        self.layers_edit.setPlaceholderText("all, or a spec like 0-5,10")

        for index, (label, widget) in enumerate(
            [
                ("Method", self.method_combo),
                ("Anomaly threshold", self.threshold_spin),
                ("Layers", self.layers_edit),
            ]
        ):
            detect_layout.addWidget(QLabel(label), index, 0)
            detect_layout.addWidget(widget, index, 1)
        layout.addWidget(detect_box)

        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #B4541F; border: none;")
        layout.addWidget(self.error_label)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.start_button = QPushButton("Start scan")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self._submit)
        buttons.addWidget(self.start_button)
        layout.addLayout(buttons)

        note = QLabel(
            "Control tokens are what separate a real trigger from ordinary rare-token "
            "novelty. Removing them means no positive verdict is reachable."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {MUTED}; font-size: 11px; border: none;")
        layout.addWidget(note)
        layout.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scrollable(inner))

    @staticmethod
    def _count_spin(value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(0, 10_000)
        spin.setValue(value)
        return spin

    def _browse(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select model directory")
        if directory:
            self.path_edit.setText(directory)

    def set_busy(self, busy: bool) -> None:
        self.start_button.setEnabled(not busy)
        self.start_button.setText("Scanning…" if busy else "Start scan")

    def validate(self) -> tuple[ScanRequest | None, list[str]]:
        """Validate every input, returning a request or a list of problems."""
        errors: list[str] = []

        path_text = self.path_edit.text().strip()
        if not path_text:
            errors.append("Model directory is required.")
        else:
            path = Path(path_text)
            if not path.is_dir():
                errors.append(f"Not a directory: {path}")
            elif not (path / "config.json").is_file():
                errors.append(f"No config.json in {path}; this is not a model directory.")

        trigger = self.trigger_edit.text().strip()
        if not trigger:
            errors.append("A candidate trigger is required.")
        elif " " in trigger:
            errors.append("Trigger must be a single token with no spaces.")

        controls = [t.strip() for t in self.controls_edit.text().split(",") if t.strip()]
        if not controls:
            errors.append(
                "At least one control token is required; without controls no positive "
                "verdict can be reached."
            )
        if trigger and trigger in controls:
            errors.append("The trigger must not also be listed as a control token.")

        if self.normal_spin.value() < 20:
            errors.append(
                "Baseline needs at least 20 normal prompts for its statistics to be usable."
            )
        if self.trigger_spin.value() < 2:
            errors.append("At least 2 trigger prompts are required to measure consistency.")

        layers = self.layers_edit.text().strip()
        if not layers:
            errors.append("Layer selection is required (use 'all' for every layer).")

        self._mark(self.path_edit, any("directory" in e or "config.json" in e for e in errors))
        self._mark(self.trigger_edit, any("rigger must" in e or "trigger is" in e for e in errors))
        self._mark(self.controls_edit, any("control token" in e for e in errors))

        if errors:
            return None, errors

        return (
            ScanRequest(
                model_path=path_text,
                trigger=trigger,
                control_tokens=controls,
                seed=self.seed_spin.value(),
                normal_prompts=self.normal_spin.value(),
                trigger_prompts=self.trigger_spin.value(),
                control_prompts_per_token=self.control_spin.value(),
                threshold=self.threshold_spin.value(),
                method=self.method_combo.currentText(),
                layers=layers,
            ),
            [],
        )

    @staticmethod
    def _mark(widget: QWidget, invalid: bool) -> None:
        widget.setProperty("invalid", "true" if invalid else "false")
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _submit(self) -> None:
        request, errors = self.validate()
        if errors:
            self.error_label.setText("\n".join(f"• {e}" for e in errors))
            return
        self.error_label.setText("")
        self.scan_requested.emit(request)


class ScanExecutionScreen(QWidget):
    """Live progress. Never blocks — all work happens on the worker thread."""

    cancel_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        card = Card("Scan progress")
        self.progress = QProgressBar()
        self.progress.setRange(0, len(STAGES))
        self.progress.setValue(0)
        self.progress.setFormat("Idle")
        card.add(self.progress)

        self.stage_labels: list[QLabel] = []
        for stage in STAGES:
            label = QLabel(f"○  {stage}")
            label.setStyleSheet(f"color: {MUTED}; border: none; padding: 1px 0;")
            self.stage_labels.append(label)
            card.add(label)
        layout.addWidget(card)

        log_card = Card("Log")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(220)
        log_card.add(self.log)
        layout.addWidget(log_card, 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.cancel_button = QPushButton("Cancel scan")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)

    def reset(self) -> None:
        self.progress.setValue(0)
        self.progress.setFormat("Starting…")
        self.log.clear()
        for index, stage in enumerate(STAGES):
            self.stage_labels[index].setText(f"○  {stage}")
            self.stage_labels[index].setStyleSheet(
                f"color: {MUTED}; border: none; padding: 1px 0;"
            )
        self.cancel_button.setEnabled(True)

    def set_stage(self, index: int, label: str) -> None:
        for i in range(index):
            self.stage_labels[i].setText(f"●  {STAGES[i]}")
            self.stage_labels[i].setStyleSheet(
                f"color: {MUTED}; border: none; padding: 1px 0;"
            )
        self.stage_labels[index].setText(f"▶  {label}")
        self.stage_labels[index].setStyleSheet(
            f"color: {ACCENT}; font-weight: 600; border: none; padding: 1px 0;"
        )
        self.progress.setValue(index)
        self.progress.setFormat(f"{label} (%p%)")

    def complete(self, message: str = "Scan complete") -> None:
        self.progress.setValue(len(STAGES))
        self.progress.setFormat(message)
        for index, stage in enumerate(STAGES):
            self.stage_labels[index].setText(f"●  {stage}")
        self.cancel_button.setEnabled(False)

    def append_log(self, message: str) -> None:
        self.log.appendPlainText(message)

    def set_failed(self, message: str) -> None:
        self.progress.setFormat("Scan failed")
        self.cancel_button.setEnabled(False)
        self.append_log(f"ERROR: {message}")


class FindingsScreen(QWidget):
    """Findings table plus the detail of the selected row."""

    COLUMNS = (
        "ID",
        "Severity",
        "Type",
        "Trigger",
        "Layer",
        "Anomaly score",
        "Reproducibility",
        "Confidence",
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        self.summary = QLabel("No scan has been run yet.")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet(f"color: {MUTED}; border: none;")
        layout.addWidget(self.summary)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(list(self.COLUMNS))
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._show_detail)
        self.table.setMinimumHeight(220)
        layout.addWidget(self.table, 1)

        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setMinimumHeight(180)
        self.detail.setPlaceholderText("Select a finding to view its evidence.")
        layout.addWidget(self.detail, 1)

        self._findings: list[Any] = []

    def update_from_result(self, result: Any, report: Any) -> None:
        self._findings = list(report.findings)
        self.summary.setText(
            f"{report.verdict_label} — {len(self._findings)} finding(s). "
            f"Reproducibility {report.consistency:.0%}, separation from controls "
            f"{report.separation:.2f}."
        )
        self.table.setRowCount(len(self._findings))
        for row, finding in enumerate(self._findings):
            cells = [
                finding.finding_id,
                finding.severity.upper(),
                finding.finding_type,
                finding.trigger or "—",
                "—" if finding.layer is None else str(finding.layer),
                "—" if finding.anomaly_score is None else f"{finding.anomaly_score:.2f}",
                "—"
                if finding.reproducibility is None
                else f"{finding.reproducibility:.0%}",
                finding.confidence.upper(),
            ]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column == 1:
                    item.setForeground(Qt.GlobalColor.black)
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        if self._findings:
            self.table.selectRow(0)
        else:
            self.detail.setPlainText("This scan raised no findings.")

    def _show_detail(self) -> None:
        rows = {index.row() for index in self.table.selectedIndexes()}
        if not rows or not self._findings:
            return
        finding = self._findings[min(rows)]
        lines = [
            f"{finding.finding_id} — {finding.title}",
            "",
            f"Severity     : {finding.severity.upper()}",
            f"Type         : {finding.finding_type}",
            f"Confidence   : {finding.confidence.upper()}",
            "",
            finding.description,
        ]
        if finding.evidence:
            lines += ["", "Evidence:"]
            lines += [f"  {k} = {v}" for k, v in sorted(finding.evidence.items())]
        self.detail.setPlainText("\n".join(lines))


class VisualizationScreen(QWidget):
    """Charts rendered from the scan, with their captions."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.inner = QWidget()
        self.layout_ = QVBoxLayout(self.inner)
        self.layout_.setContentsMargins(18, 18, 18, 18)
        self.layout_.setSpacing(16)

        self.placeholder = QLabel(
            "No visualisations yet. Charts are rendered only from measured scan data."
        )
        self.placeholder.setStyleSheet(f"color: {MUTED}; border: none;")
        self.layout_.addWidget(self.placeholder)
        self.layout_.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scrollable(self.inner))

    def set_charts(self, charts: list[Any]) -> None:
        while self.layout_.count():
            item = self.layout_.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not charts:
            label = QLabel(
                "No charts could be rendered from this scan. Figures are omitted rather "
                "than substituted when the underlying data is absent."
            )
            label.setWordWrap(True)
            label.setStyleSheet(f"color: {MUTED}; border: none;")
            self.layout_.addWidget(label)
            self.layout_.addStretch(1)
            return

        for chart in charts:
            card = Card(chart.title)
            image = QLabel()
            pixmap = QPixmap()
            pixmap.loadFromData(chart.png)
            image.setPixmap(
                pixmap.scaledToWidth(
                    820, Qt.TransformationMode.SmoothTransformation
                )
            )
            image.setStyleSheet(f"border: 1px solid {BORDER}; border-radius: 4px;")
            card.add(image)

            caption = QLabel(chart.caption)
            caption.setWordWrap(True)
            caption.setStyleSheet(f"color: {MUTED}; font-size: 11px; border: none;")
            card.add(caption)
            self.layout_.addWidget(card)
        self.layout_.addStretch(1)
