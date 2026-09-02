"""Background scan worker.

A NeuroFence scan loads a model and runs hundreds of forward passes; on the GUI
thread that would freeze the window for minutes and Qt would mark the app as
not responding. All scanning therefore happens on a ``QThread``, communicating
with the UI only through signals.

Design constraints that follow from Qt's threading rules:

* The worker owns no widgets and touches no UI object. It emits signals; the
  main thread decides what to draw.
* Progress is reported as discrete named stages rather than a fake percentage,
  because the stage durations are wildly uneven (model loading dominates) and a
  smoothly-advancing bar would be a lie.
* Cancellation is cooperative and checked between stages. A scan cannot be
  interrupted mid-forward-pass without corrupting hook state, so the worker
  finishes the current stage and stops cleanly rather than being killed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from ..core.logging import get_logger

logger = get_logger(__name__)

#: Stages reported to the UI, in order.
STAGES: tuple[str, ...] = (
    "Loading model...",
    "Calculating fingerprint...",
    "Preparing prompts...",
    "Running inference...",
    "Collecting activations...",
    "Analyzing anomalies...",
    "Generating findings...",
)


class ScanCancelled(Exception):
    """Raised internally when the user cancels between stages."""


@dataclass
class ScanRequest:
    """Everything the worker needs to run one scan."""

    model_path: str
    trigger: str = "PINEAPPLE"
    seed: int = 42
    normal_prompts: int = 100
    trigger_prompts: int = 50
    control_prompts_per_token: int = 25
    random_prompts: int = 50
    edge_case_prompts: int = 50
    security_prompts: int = 50
    paraphrase_prompts: int = 30
    threshold: float = 3.0
    method: str = "robust"
    layers: str = "all"
    control_tokens: list[str] = field(
        default_factory=lambda: ["APPLE", "BANANA", "ORANGE", "MANGO"]
    )

    def to_config(self, base: Any) -> Any:
        """Apply this request onto a copy of a loaded config."""
        import copy
        from pathlib import Path

        cfg = copy.deepcopy(base)
        resolved = Path(self.model_path).resolve()
        cfg.model.local_path = str(resolved)
        cfg.model.name = f"local:{resolved.name}"
        cfg.fuzzer.seed = self.seed
        cfg.fuzzer.trigger = self.trigger
        cfg.fuzzer.control_tokens = [t for t in self.control_tokens if t != self.trigger]
        cfg.fuzzer.normal_prompts = self.normal_prompts
        cfg.fuzzer.trigger_prompts = self.trigger_prompts
        cfg.fuzzer.control_prompts_per_token = self.control_prompts_per_token
        cfg.fuzzer.random_prompts = self.random_prompts
        cfg.fuzzer.edge_case_prompts = self.edge_case_prompts
        cfg.fuzzer.security_prompts = self.security_prompts
        cfg.fuzzer.paraphrase_prompts = self.paraphrase_prompts
        cfg.detection.threshold = self.threshold
        cfg.detection.method = self.method
        cfg.activation.layers = self.layers
        cfg.logging.console = False
        cfg.validate()
        return cfg


class ScanWorker(QObject):
    """Runs a Phase 2 scan off the GUI thread."""

    stage_changed = pyqtSignal(int, str)  # index, label
    log_message = pyqtSignal(str)
    finished = pyqtSignal(object)  # Phase2Result
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, request: ScanRequest, base_config: Any) -> None:
        super().__init__()
        self.request = request
        self.base_config = base_config
        self._cancel = False

    def cancel(self) -> None:
        """Request cooperative cancellation; honoured between stages."""
        self._cancel = True
        self.log_message.emit("Cancellation requested; stopping after the current stage.")

    # --- execution --------------------------------------------------------

    def run(self) -> None:
        try:
            result = self._run_scan()
        except ScanCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:
            logger.exception("scan failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(result)

    def _stage(self, index: int) -> None:
        if self._cancel:
            raise ScanCancelled
        self.stage_changed.emit(index, STAGES[index])
        self.log_message.emit(STAGES[index])

    def _run_scan(self) -> Any:
        from ..activation.collector import ActivationCollector
        from ..baseline.analyzer import BaselineAnalyzer
        from ..detection.anomaly import aggregate_by_layer
        from ..detection.risk_score import RiskScorer, RiskWeights
        from ..detection.trigger_analysis import TriggerAnalyzer
        from ..model.loader import load_model
        from ..phase2 import (
            Phase2Result,
            _build_baseline,
            _check_determinism,
            _generate_prompts,
            _score_prompts,
        )
        from ..sandbox.policy import SandboxPolicy
        from ..sandbox.sandbox import ModelSandbox

        cfg = self.request.to_config(self.base_config)

        # Stages are driven explicitly rather than by calling run_phase2, so the
        # UI can report real progress instead of a single opaque wait.
        self._stage(2)
        prompt_set = _generate_prompts(cfg)
        self.log_message.emit(
            f"Generated {len(prompt_set)} unique prompts "
            f"({len(prompt_set.triggers())} trigger, {len(prompt_set.controls())} control)."
        )
        if prompt_set.shortfalls:
            self.log_message.emit(f"Prompt shortfalls: {prompt_set.shortfalls}")

        with ModelSandbox(SandboxPolicy.from_config(cfg.sandbox)) as sandbox:
            sandbox.check_remote_code(cfg.model.trust_remote_code)
            sandbox.check_download(cfg.model.allow_download)

            self._stage(0)
            loaded = load_model(cfg.model)
            self.log_message.emit(
                f"Loaded {loaded.metadata.architecture} "
                f"({loaded.metadata.layers} layers, {loaded.metadata.parameters:,} params)."
            )

            self._stage(1)
            self.log_message.emit(f"Weights SHA-256: {loaded.fingerprint.weights_sha256}")

            collector = ActivationCollector(loaded, cfg.activation)

            self._stage(3)
            baseline = _build_baseline(cfg, collector, prompt_set, loaded)
            quality = BaselineAnalyzer().analyze(baseline)
            self.log_message.emit(
                f"Baseline built from {baseline.prompt_count} prompts "
                f"(usable: {baseline.is_usable})."
            )
            for warning in quality.warnings[:3]:
                self.log_message.emit(f"Warning: {warning}")

            self._stage(4)
            anomalies = _score_prompts(cfg, collector, prompt_set, baseline)
            determinism = _check_determinism(cfg, collector, prompt_set, baseline)
            if determinism is not None and determinism < 1.0:
                self.log_message.emit(
                    f"Determinism check FAILED ({determinism:.2f}); results unreliable."
                )

            sandbox_report = sandbox.report.to_dict()

        self._stage(5)
        trigger_result = TriggerAnalyzer(
            threshold=cfg.detection.threshold,
            metric=cfg.detection.primary_metric,
            min_consistency=cfg.detection.min_consistency,
            min_separation=cfg.detection.min_separation,
        ).analyze(anomalies, cfg.fuzzer.trigger, determinism=determinism)
        self.log_message.emit(f"Verdict: {trigger_result.verdict}")

        self._stage(6)
        risk = RiskScorer(
            RiskWeights(
                trigger_consistency=cfg.detection.weight_trigger_consistency,
                control_separation=cfg.detection.weight_control_separation,
                layer_concentration=cfg.detection.weight_layer_concentration,
                anomaly_magnitude=cfg.detection.weight_anomaly_magnitude,
            ),
            threshold=cfg.detection.threshold,
            saturation_score=cfg.detection.saturation_score,
            saturation_separation=cfg.detection.saturation_separation,
        ).score(
            trigger_result,
            anomalies,
            baseline_usable=baseline.is_usable,
            baseline_warnings=quality.warnings,
        )
        self.log_message.emit(f"Risk score: {risk.score:.1f} ({risk.severity})")

        return Phase2Result(
            loaded=loaded,
            prompt_set=prompt_set,
            baseline=baseline,
            baseline_quality=quality,
            anomalies=anomalies,
            trigger_result=trigger_result,
            risk=risk,
            layer_summary=aggregate_by_layer(
                [a for a in anomalies if a.trigger],
                cfg.detection.primary_metric,
                cfg.detection.threshold,
            ),
            sandbox_report=sandbox_report,
        )


class ScanController(QObject):
    """Owns the worker thread and its lifecycle.

    Keeping thread management here rather than in the window means the window
    never has to reason about Qt thread affinity, and the thread is always
    joined before the app exits.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.thread: QThread | None = None
        self.worker: ScanWorker | None = None

    @property
    def busy(self) -> bool:
        return self.thread is not None and self.thread.isRunning()

    def start(self, request: ScanRequest, base_config: Any) -> ScanWorker:
        if self.busy:
            raise RuntimeError("A scan is already running")

        self.thread = QThread()
        self.worker = ScanWorker(request, base_config)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        for signal in (self.worker.finished, self.worker.failed, self.worker.cancelled):
            signal.connect(self._teardown)
        return self.worker

    def launch(self) -> None:
        if self.thread is not None:
            self.thread.start()

    def cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()

    def _teardown(self, *_args: object) -> None:
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(5000)
            self.thread = None
        self.worker = None

    def shutdown(self) -> None:
        """Called on window close so a running scan never outlives the app."""
        self.cancel()
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(5000)
            self.thread = None
        self.worker = None
