"""Tests for the desktop application.

Qt runs under the ``offscreen`` platform so these execute headlessly in CI. The
focus is on the two things most likely to go wrong in a GUI wrapped around slow
work: input validation letting a bad scan start, and the scan blocking the
event loop.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6", reason="PyQt6 not installed; GUI extras are optional")

from PyQt6.QtWidgets import QApplication

from neurofence.desktop.worker import STAGES, ScanController, ScanRequest


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qapp, base_config):
    from neurofence.desktop.app import MainWindow

    win = MainWindow(base_config)
    yield win
    win.controller.shutdown()
    win.close()


class TestWindowConstruction:
    def test_all_screens_present(self, window) -> None:
        labels = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        assert labels == [
            "Dashboard",
            "Model",
            "Configure",
            "Scan",
            "Findings",
            "Visualisations",
        ]

    def test_export_disabled_before_a_scan(self, window) -> None:
        """Nothing to export until something has actually been measured."""
        assert not window.export_pdf_action.isEnabled()
        assert not window.export_json_action.isEnabled()

    def test_dashboard_shows_no_data_placeholders(self, window) -> None:
        """'Not yet measured' must not look like 'zero anomalies'."""
        assert window.dashboard.tiles["risk"].value_label.text() == "—"
        assert "No scan" in window.dashboard.verdict_label.text()

    def test_renders_offscreen(self, window) -> None:
        window.resize(1100, 800)
        pixmap = window.grab()
        assert pixmap.width() > 0 and pixmap.height() > 0


class TestConfigValidation:
    def test_rejects_missing_model_path(self, window) -> None:
        screen = window.scan_config
        screen.path_edit.setText("")
        request, errors = screen.validate()
        assert request is None
        assert any("required" in e for e in errors)

    def test_rejects_non_model_directory(self, window, tmp_path: Path) -> None:
        screen = window.scan_config
        screen.path_edit.setText(str(tmp_path))
        request, errors = screen.validate()
        assert request is None
        assert any("config.json" in e for e in errors)

    def test_rejects_empty_trigger(self, window, tiny_model_dir: Path) -> None:
        screen = window.scan_config
        screen.path_edit.setText(str(tiny_model_dir))
        screen.trigger_edit.setText("   ")
        request, errors = screen.validate()
        assert request is None
        assert any("trigger is required" in e for e in errors)

    def test_rejects_trigger_with_spaces(self, window, tiny_model_dir: Path) -> None:
        screen = window.scan_config
        screen.path_edit.setText(str(tiny_model_dir))
        screen.trigger_edit.setText("TWO WORDS")
        request, errors = screen.validate()
        assert request is None
        assert any("single token" in e for e in errors)

    def test_rejects_empty_controls(self, window, tiny_model_dir: Path) -> None:
        """Without controls no positive verdict is reachable, so refuse early."""
        screen = window.scan_config
        screen.path_edit.setText(str(tiny_model_dir))
        screen.controls_edit.setText("")
        request, errors = screen.validate()
        assert request is None
        assert any("control token" in e for e in errors)

    def test_rejects_trigger_listed_as_control(self, window, tiny_model_dir: Path) -> None:
        screen = window.scan_config
        screen.path_edit.setText(str(tiny_model_dir))
        screen.trigger_edit.setText("APPLE")
        screen.controls_edit.setText("APPLE, BANANA")
        request, errors = screen.validate()
        assert request is None
        assert any("must not also be listed" in e for e in errors)

    def test_rejects_undersized_baseline(self, window, tiny_model_dir: Path) -> None:
        screen = window.scan_config
        screen.path_edit.setText(str(tiny_model_dir))
        screen.normal_spin.setValue(5)
        request, errors = screen.validate()
        assert request is None
        assert any("at least 20 normal prompts" in e for e in errors)

    def test_accepts_valid_configuration(self, window, tiny_model_dir: Path) -> None:
        screen = window.scan_config
        screen.path_edit.setText(str(tiny_model_dir))
        screen.trigger_edit.setText("PINEAPPLE")
        screen.controls_edit.setText("APPLE, BANANA")
        screen.normal_spin.setValue(24)
        request, errors = screen.validate()
        assert errors == []
        assert request is not None
        assert request.trigger == "PINEAPPLE"
        assert request.control_tokens == ["APPLE", "BANANA"]

    def test_submit_does_not_emit_when_invalid(self, window) -> None:
        emitted = []
        window.scan_config.scan_requested.connect(emitted.append)
        window.scan_config.path_edit.setText("")
        window.scan_config._submit()
        assert emitted == []
        assert window.scan_config.error_label.text()


class TestScanRequest:
    def test_builds_a_valid_config(self, base_config, tiny_model_dir: Path) -> None:
        request = ScanRequest(
            model_path=str(tiny_model_dir),
            trigger="PINEAPPLE",
            normal_prompts=24,
            threshold=4.0,
        )
        cfg = request.to_config(base_config)
        assert cfg.fuzzer.trigger == "PINEAPPLE"
        assert cfg.detection.threshold == 4.0
        assert cfg.fuzzer.normal_prompts == 24

    def test_trigger_removed_from_controls(self, base_config, tiny_model_dir: Path) -> None:
        """Config validation would otherwise reject the run."""
        request = ScanRequest(
            model_path=str(tiny_model_dir),
            trigger="APPLE",
            control_tokens=["APPLE", "BANANA"],
        )
        cfg = request.to_config(base_config)
        assert "APPLE" not in cfg.fuzzer.control_tokens
        assert "BANANA" in cfg.fuzzer.control_tokens


class TestProgressScreen:
    def test_stage_labels_match_worker_stages(self, window) -> None:
        assert len(window.execution.stage_labels) == len(STAGES)

    def test_progress_advances(self, window) -> None:
        screen = window.execution
        screen.reset()
        assert screen.progress.value() == 0
        screen.set_stage(3, STAGES[3])
        assert screen.progress.value() == 3
        screen.complete()
        assert screen.progress.value() == len(STAGES)

    def test_cancel_enabled_only_while_running(self, window) -> None:
        screen = window.execution
        assert not screen.cancel_button.isEnabled()
        screen.reset()
        assert screen.cancel_button.isEnabled()
        screen.complete()
        assert not screen.cancel_button.isEnabled()

    def test_failure_is_reported(self, window) -> None:
        window.execution.set_failed("boom")
        assert "boom" in window.execution.log.toPlainText()


class TestController:
    def test_not_busy_initially(self, qapp) -> None:
        controller = ScanController()
        assert not controller.busy
        controller.shutdown()

    def test_shutdown_is_safe_when_idle(self, qapp) -> None:
        controller = ScanController()
        controller.shutdown()
        controller.shutdown()  # idempotent
        assert not controller.busy


@pytest.mark.slow
@pytest.mark.integration
class TestLiveScan:
    def test_scan_runs_on_worker_thread_without_blocking(
        self, window, detection_model_dir: Path, qapp
    ) -> None:
        """The spec's explicit failure mode: a frozen GUI."""
        from PyQt6.QtCore import QElapsedTimer, QEventLoop, QTimer

        request = ScanRequest(
            model_path=str(detection_model_dir),
            trigger="PINEAPPLE",
            normal_prompts=24,
            trigger_prompts=8,
            control_prompts_per_token=4,
            random_prompts=4,
            edge_case_prompts=4,
            security_prompts=4,
            paraphrase_prompts=2,
        )

        loop = QEventLoop()
        outcome: dict[str, object] = {}
        ticks = {"count": 0}

        window.start_scan(request)
        worker = window.controller.worker
        worker.finished.connect(lambda r: (outcome.update(result=r), loop.quit()))
        worker.failed.connect(lambda m: (outcome.update(error=m), loop.quit()))

        # If the event loop were blocked, this timer would never fire.
        heartbeat = QTimer()
        heartbeat.timeout.connect(lambda: ticks.__setitem__("count", ticks["count"] + 1))
        heartbeat.start(50)

        timer = QElapsedTimer()
        timer.start()
        QTimer.singleShot(600_000, loop.quit)
        loop.exec()
        heartbeat.stop()

        assert "error" not in outcome, outcome.get("error")
        assert "result" in outcome, "scan did not complete"
        assert ticks["count"] > 5, "event loop was blocked during the scan"

        window.on_finished(outcome["result"])
        assert window.report is not None
        assert window.export_pdf_action.isEnabled()
        assert window.dashboard.tiles["risk"].value_label.text() != "—"

    def test_end_to_end_pdf_export(self, window, detection_model_dir: Path, tmp_path: Path):
        from neurofence.core.config import Config
        from neurofence.phase2 import run_phase2
        from neurofence.reporting import build_report, generate_pdf
        from neurofence.visualization import build_all

        cfg = Config()
        cfg.model.local_path = str(detection_model_dir)
        cfg.model.device = "cpu"
        cfg.logging.console = False
        cfg.fuzzer.normal_prompts = 24
        cfg.fuzzer.trigger_prompts = 8
        cfg.fuzzer.control_prompts_per_token = 4
        cfg.fuzzer.random_prompts = 4
        cfg.fuzzer.edge_case_prompts = 4
        cfg.fuzzer.security_prompts = 4
        cfg.fuzzer.paraphrase_prompts = 2
        cfg.run.output_dir = str(tmp_path / "out")
        cfg.run.log_dir = str(tmp_path / "logs")
        cfg.validate()

        result = run_phase2(cfg=cfg, write_output=False)
        charts = build_all(result, threshold=cfg.detection.threshold)
        report = build_report(result)
        path = generate_pdf(report, tmp_path / "report.pdf", charts)

        assert path.read_bytes().startswith(b"%PDF")
        assert charts, "no charts were rendered from a real scan"
        window.visualization.set_charts(charts)
