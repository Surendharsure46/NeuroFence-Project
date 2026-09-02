"""Validation of the detector against known ground truth.

Runs the full scan against both arms of the controlled experiment and reports
detection performance:

* **clean arm** — a known-negative. A finding here is a false positive, and
  false positives are what make a scanner unusable in practice.
* **poisoned arm** — a known-positive. No finding here is a false negative.

The output is a plain statement of what the detector did, including the
uncomfortable outcomes. A validation harness that can only report success is a
marketing artefact, not a test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.logging import get_logger
from .controlled_backdoor import BackdoorGroundTruth

logger = get_logger(__name__)

POSITIVE_VERDICTS = {"trigger_behaviour_detected"}


@dataclass
class ArmResult:
    """Scan outcome for one arm of the experiment."""

    label: str
    expected_positive: bool
    verdict: str
    risk_score: float
    consistency: float
    separation: float
    top_layers: list[int] = field(default_factory=list)

    @property
    def detected(self) -> bool:
        return self.verdict in POSITIVE_VERDICTS

    @property
    def outcome(self) -> str:
        if self.expected_positive:
            return "true_positive" if self.detected else "false_negative"
        return "false_positive" if self.detected else "true_negative"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "expected_positive": self.expected_positive,
            "detected": self.detected,
            "outcome": self.outcome,
            "verdict": self.verdict,
            "risk_score": round(self.risk_score, 2),
            "consistency": round(self.consistency, 4),
            "separation": round(self.separation, 4),
            "top_layers": self.top_layers[:5],
        }


@dataclass
class ValidationReport:
    """Detection performance across both arms."""

    ground_truth: dict[str, Any] = field(default_factory=dict)
    arms: list[ArmResult] = field(default_factory=list)

    @property
    def true_positives(self) -> int:
        return sum(1 for a in self.arms if a.outcome == "true_positive")

    @property
    def false_positives(self) -> int:
        return sum(1 for a in self.arms if a.outcome == "false_positive")

    @property
    def false_negatives(self) -> int:
        return sum(1 for a in self.arms if a.outcome == "false_negative")

    @property
    def true_negatives(self) -> int:
        return sum(1 for a in self.arms if a.outcome == "true_negative")

    @property
    def passed(self) -> bool:
        """Validation passes only with no false positives and no false negatives."""
        return self.false_positives == 0 and self.false_negatives == 0 and bool(self.arms)

    @property
    def summary(self) -> str:
        if not self.arms:
            return "no arms were run"
        if self.passed:
            return "detector distinguished the poisoned model from the clean model"
        problems = []
        if self.false_negatives:
            problems.append(f"{self.false_negatives} false negative(s): planted backdoor missed")
        if self.false_positives:
            problems.append(f"{self.false_positives} false positive(s): clean model flagged")
        return "; ".join(problems)

    @property
    def risk_gap(self) -> float:
        """Risk-score separation between the poisoned and clean arms."""
        poisoned = [a.risk_score for a in self.arms if a.expected_positive]
        clean = [a.risk_score for a in self.arms if not a.expected_positive]
        if not poisoned or not clean:
            return 0.0
        return max(poisoned) - max(clean)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "summary": self.summary,
            "risk_gap": round(self.risk_gap, 2),
            "confusion": {
                "true_positive": self.true_positives,
                "false_positive": self.false_positives,
                "true_negative": self.true_negatives,
                "false_negative": self.false_negatives,
            },
            "ground_truth": self.ground_truth,
            "arms": [a.to_dict() for a in self.arms],
        }


def build_report(
    truth: BackdoorGroundTruth,
    clean_result: Any,
    poisoned_result: Any,
) -> ValidationReport:
    """Assemble a validation report from two Phase 2 scan results."""
    report = ValidationReport(ground_truth=truth.to_dict())
    for label, result, expected in (
        ("clean", clean_result, False),
        ("poisoned", poisoned_result, True),
    ):
        trigger = result.trigger_result
        profile = trigger.trigger_profile
        report.arms.append(
            ArmResult(
                label=label,
                expected_positive=expected,
                verdict=trigger.verdict,
                risk_score=result.risk.score,
                consistency=trigger.consistency,
                separation=trigger.separation,
                top_layers=list(profile.top_layers) if profile else [],
            )
        )
    logger.info(
        "validation complete",
        extra={
            "extra_fields": {
                "passed": report.passed,
                "risk_gap": round(report.risk_gap, 2),
                "summary": report.summary,
            }
        },
    )
    return report
