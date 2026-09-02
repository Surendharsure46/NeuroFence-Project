"""Main window and application entry point.

Wires the screens to the worker. The window is a coordinator: it holds no
analysis logic, and every value it displays arrives from a completed scan.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QTabWidget,
)

from ..core.config import Config, load_config
from ..core.logging import get_logger, setup_logging
from .screens import (
    DashboardScreen,
    FindingsScreen,
    ModelInspectionScreen,
    ScanConfigScreen,
    ScanExecutionScreen,
    VisualizationScreen,
)
from .theme import STYLESHEET
from .worker import ScanController, ScanRequest

logger = get_logger(__name__)

APP_NAME = "NeuroFence"


class MainWindow(QMainWindow):
    """The NeuroFence forensic desktop application."""

    def __init__(self, config: Config | None = None) -> None:
        super().__init__()
        self.config = config or load_config()
        self.controller = ScanController(self)
        self.result: Any = None
        self.report: Any = None
        self.charts: list[Any] = []

        self.setWindowTitle(f"{APP_NAME} — LLM Weight Poisoning & Backdoor Scanner")
        self.resize(1180, 840)
        self.setStyleSheet(STYLESHEET)

        self.tabs = QTabWidget()
        self.dashboard = DashboardScreen()
        self.inspection = ModelInspectionScreen()
        self.scan_config = ScanConfigScreen()
        self.execution = ScanExecutionScreen()
        self.findings = FindingsScreen()
        self.visualization = VisualizationScreen()

        for widget, label in (
            (self.dashboard, "Dashboard"),
            (self.inspection, "Model"),
            (self.scan_config, "Configure"),
            (self.execution, "Scan"),
            (self.findings, "Findings"),
            (self.visualization, "Visualisations"),
        ):
            self.tabs.addTab(widget, label)
        self.setCentralWidget(self.tabs)

        self.scan_config.scan_requested.connect(self.start_scan)
        self.execution.cancel_requested.connect(self.controller.cancel)

        self._build_menu()
        self.statusBar().showMessage("Ready — offline. No network access, no downloads.")

        # Pre-fill from config so the default flow needs no typing.
        self.scan_config.path_edit.setText(self.config.model.local_path)
        self.scan_config.trigger_edit.setText(self.config.fuzzer.trigger)
        self.scan_config.controls_edit.setText(", ".join(self.config.fuzzer.control_tokens))

    # --- menu -------------------------------------------------------------

    def _build_menu(self) -> None:
        menu = self.menuBar()

        file_menu = menu.addMenu("&File")
        self.export_pdf_action = QAction("Export PDF report…", self)
        self.export_pdf_action.triggered.connect(self.export_pdf)
        self.export_pdf_action.setEnabled(False)
        file_menu.addAction(self.export_pdf_action)

        self.export_json_action = QAction("Export JSON report…", self)
        self.export_json_action.triggered.connect(self.export_json)
        self.export_json_action.setEnabled(False)
        file_menu.addAction(self.export_json_action)

        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        help_menu = menu.addMenu("&Help")
        about = QAction("About NeuroFence", self)
        about.triggered.connect(self.show_about)
        help_menu.addAction(about)

    def show_about(self) -> None:
        from .. import __version__

        QMessageBox.information(
            self,
            f"About {APP_NAME}",
            f"{APP_NAME} v{__version__}\n\n"
            "Offline LLM weight-poisoning and backdoor scanner.\n\n"
            "NeuroFence measures activation-level anomalies. It cannot establish "
            "intent, confirm a functional backdoor, or certify a model as safe. "
            "Findings require review by a qualified analyst.\n\n"
            "The sandbox is an in-process guardrail, not a security boundary. "
            "Scan untrusted weights inside an isolated VM.",
        )

    # --- scanning ---------------------------------------------------------

    def start_scan(self, request: ScanRequest) -> None:
        if self.controller.busy:
            QMessageBox.warning(self, APP_NAME, "A scan is already running.")
            return

        self.tabs.setCurrentWidget(self.execution)
        self.execution.reset()
        self.scan_config.set_busy(True)
        self.statusBar().showMessage("Scanning…")

        worker = self.controller.start(request, self.config)
        worker.stage_changed.connect(self.execution.set_stage)
        worker.log_message.connect(self.execution.append_log)
        worker.finished.connect(self.on_finished)
        worker.failed.connect(self.on_failed)
        worker.cancelled.connect(self.on_cancelled)
        self.controller.launch()

    def on_finished(self, result: Any) -> None:
        from ..reporting import build_report
        from ..visualization import build_all

        self.result = result
        self.execution.complete()
        self.scan_config.set_busy(False)

        try:
            self.report = build_report(result)
            self.charts = build_all(
                result, threshold=self.scan_config.threshold_spin.value()
            )
        except Exception as exc:
            logger.exception("post-processing failed")
            self.execution.append_log(f"Report generation failed: {exc}")
            QMessageBox.warning(
                self,
                APP_NAME,
                f"The scan completed but the report could not be built:\n{exc}",
            )
            return

        self.dashboard.update_from_result(result, self.report)
        self.inspection.update_from_result(result, self.report)
        self.findings.update_from_result(result, self.report)
        self.visualization.set_charts(self.charts)

        self.export_pdf_action.setEnabled(True)
        self.export_json_action.setEnabled(True)
        self.statusBar().showMessage(
            f"Scan complete — {self.report.verdict_label} "
            f"(risk {result.risk.score:.1f}/100)"
        )
        self.tabs.setCurrentWidget(self.dashboard)

    def on_failed(self, message: str) -> None:
        self.execution.set_failed(message)
        self.scan_config.set_busy(False)
        self.statusBar().showMessage("Scan failed")
        QMessageBox.critical(self, APP_NAME, f"The scan failed:\n\n{message}")

    def on_cancelled(self) -> None:
        self.execution.append_log("Scan cancelled.")
        self.execution.complete("Cancelled")
        self.scan_config.set_busy(False)
        self.statusBar().showMessage("Scan cancelled")

    # --- export -----------------------------------------------------------

    def export_pdf(self) -> None:
        if self.report is None:
            return
        default = f"neurofence-report-{self.report.model.sha256[:8] or 'scan'}.pdf"
        path, _ = QFileDialog.getSaveFileName(self, "Export PDF report", default, "PDF (*.pdf)")
        if not path:
            return
        from ..reporting import generate_pdf

        try:
            generate_pdf(self.report, path, self.charts)
        except Exception as exc:
            QMessageBox.critical(self, APP_NAME, f"Could not write the PDF:\n{exc}")
            return
        self.statusBar().showMessage(f"Report written to {path}")

    def export_json(self) -> None:
        if self.report is None:
            return
        import json

        path, _ = QFileDialog.getSaveFileName(
            self, "Export JSON report", "neurofence-report.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            Path(path).write_text(
                json.dumps(self.report.to_dict(), indent=2, default=str), encoding="utf-8"
            )
        except OSError as exc:
            QMessageBox.critical(self, APP_NAME, f"Could not write the report:\n{exc}")
            return
        self.statusBar().showMessage(f"Report written to {path}")

    # --- lifecycle --------------------------------------------------------

    def closeEvent(self, event: Any) -> None:
        if self.controller.busy:
            answer = QMessageBox.question(
                self,
                APP_NAME,
                "A scan is still running. Cancel it and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self.controller.shutdown()
        event.accept()


def main(argv: list[str] | None = None) -> int:
    """Launch the desktop application."""
    argv = sys.argv if argv is None else argv

    config = load_config()
    setup_logging(
        level=config.logging.level,
        log_dir=config.run.log_dir,
        run_name="desktop",
        console=False,
        force=True,
    )

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontShowIconsInMenus, False)
    app = QApplication(argv)
    app.setApplicationName(APP_NAME)

    window = MainWindow(config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
